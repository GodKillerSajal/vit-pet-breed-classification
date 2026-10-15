"""Download, audit, and prepare the official Oxford-IIIT Pet split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
import urllib.request
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError


PARQUET_URL = (
    "https://huggingface.co/datasets/enterprise-explorers/oxford-pets/resolve/main/"
    "data/train-00000-of-00001-ecc2afb43dedd5e0.parquet"
)
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "portfolio-project-data-prep/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        copied = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            copied += len(chunk)
            if total:
                print(f"\r{path.name}: {copied / total:.1%}", end="", flush=True)
    print()


def safe_extract_annotations(archive_path: Path, destination: Path) -> None:
    if (destination / "annotations" / "list.txt").exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in annotation archive: {member.name}")
        archive.extractall(destination, filter="data")


def read_annotation_list(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            image_id, breed_id, species_id, _breed_number = parts[:4]
            result[Path(image_id).stem.casefold()] = {
                "breed_id": int(breed_id),
                "species_id": int(species_id),
                "breed": Path(image_id).stem.rsplit("_", 1)[0],
                "species": "cat" if int(species_id) == 1 else "dog",
            }
    return result


def read_split(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if parts:
                names.add(Path(parts[0]).stem.casefold())
    return names


def dataset_image(row: dict[str, object]) -> tuple[str, Image.Image]:
    image_id = None
    for key in ("image_id", "path", "file_name", "filename"):
        value = row.get(key)
        if value:
            image_id = Path(str(value)).name
            break
    if image_id is None:
        raise ValueError("The image mirror row has no filename/id column")
    value = row["image"]
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    elif isinstance(value, dict) and value.get("bytes"):
        image = Image.open(BytesIO(value["bytes"])).convert("RGB")
    elif isinstance(value, dict) and value.get("path"):
        image = Image.open(value["path"]).convert("RGB")
    else:
        raise ValueError(f"Could not decode mirrored image row {image_id}")
    return Path(image_id).stem.casefold(), image


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[2] / "data" / "oxford-iiit-pet"
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument("--force", action="store_true", help="Regenerate manifests and audits")
    args = parser.parse_args()
    root = args.data_root.resolve()
    raw, images = root / "raw", root / "images"
    parquet = raw / "train.parquet"
    annotations_archive = raw / "annotations.tar.gz"
    download(PARQUET_URL, parquet)
    download(ANNOTATIONS_URL, annotations_archive)
    safe_extract_annotations(annotations_archive, root)

    annotation_dir = root / "annotations"
    labels = read_annotation_list(annotation_dir / "list.txt")
    official_test = read_split(annotation_dir / "test.txt")
    official_trainval = read_split(annotation_dir / "trainval.txt")
    if not labels or not official_test or not official_trainval:
        raise RuntimeError("Oxford-IIIT Pet annotation/split files were missing or empty")

    images.mkdir(parents=True, exist_ok=True)
    parquet_ds = load_dataset("parquet", data_files={"train": str(parquet)}, split="train")
    written: set[str] = set()
    duplicates = 0
    unmatched = 0
    mirror_label_mismatches = 0
    dimensions: list[tuple[int, int]] = []
    corrupt: list[str] = []
    rows_by_breed: dict[int, list[str]] = defaultdict(list)
    class_labels = parquet_ds.features.get("label")
    for row in parquet_ds:
        image_id, image = dataset_image(row)
        if image_id not in labels:
            unmatched += 1
            continue
        if image_id in written:
            duplicates += 1
            continue
        annotation = labels[image_id]
        try:
            width, height = image.size
            if width < 32 or height < 32:
                raise ValueError("Image dimensions are unexpectedly small")
            output_path = images / f"{image_id}.jpg"
            image.save(output_path, format="JPEG", quality=95)
            with Image.open(output_path) as check:
                check.verify()
        except (OSError, UnidentifiedImageError, ValueError):
            corrupt.append(image_id)
            continue
        written.add(image_id)
        dimensions.append((width, height))
        rows_by_breed[int(annotation["breed_id"])].append(image_id)
        row_label = row.get("label")
        if row_label is not None and class_labels is not None:
            if hasattr(class_labels, "int2str"):
                mirror_name = class_labels.int2str(int(row_label))
            else:
                mirror_name = str(row_label)
            if mirror_name.casefold().replace(" ", "_") != str(annotation["breed"]).casefold():
                mirror_label_mismatches += 1

    valid_trainval = official_trainval & written
    valid_test = official_test & written
    assignment: dict[str, str] = {name: "test" for name in valid_test}
    for breed_id, names in rows_by_breed.items():
        train_names = sorted(set(names) & valid_trainval)
        # Stable hash makes the validation sample reproducible and stratified by breed.
        ranked = sorted(train_names, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
        validation_count = max(1, round(len(ranked) * 0.15)) if len(ranked) > 1 else 0
        for name in ranked[:validation_count]:
            assignment[name] = "validation"
        for name in ranked[validation_count:]:
            assignment[name] = "train"

    records: list[dict[str, object]] = []
    for name, split in sorted(assignment.items()):
        annotation = labels[name]
        records.append(
            {
                "image_id": name,
                "image_path": str((images / f"{name}.jpg").resolve()),
                "breed": annotation["breed"],
                "breed_id": int(annotation["breed_id"]) - 1,
                "species": annotation["species"],
                "species_id": int(annotation["species_id"]) - 1,
                "split": split,
                "xml_path": str((annotation_dir / "xmls" / f"{name}.xml").resolve()),
            }
        )
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_dir / "manifest.csv", index=False)
    for split in ("train", "validation", "test"):
        manifest[manifest["split"] == split].to_csv(manifest_dir / f"{split}.csv", index=False)

    breed_counts = manifest.groupby(["split", "breed"], sort=True).size().unstack(fill_value=0)
    plt.figure(figsize=(15, 5))
    manifest["breed"].value_counts().sort_values().plot(kind="barh", color="#2f6b9a")
    plt.title("Oxford-IIIT Pet: images per breed after annotation matching")
    plt.xlabel("Image count")
    plt.tight_layout()
    plt.savefig(manifest_dir / "breed_distribution.png", dpi=160)
    plt.close()

    audit = {
        "dataset": "Oxford-IIIT Pet",
        "annotation_rows": len(labels),
        "mirror_rows": len(parquet_ds),
        "unique_annotated_images_written": len(written),
        "manifest_rows": len(manifest),
        "duplicate_mirror_rows_skipped": duplicates,
        "unmatched_mirror_rows_skipped": unmatched,
        "corrupt_images_skipped": corrupt,
        "mirror_label_mismatches": mirror_label_mismatches,
        "trainval_annotation_names": len(official_trainval),
        "official_test_annotation_names": len(official_test),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "class_count": int(manifest["breed_id"].nunique()),
        "species_counts": manifest["species"].value_counts().to_dict(),
        "image_width_range": [int(min(w for w, _ in dimensions)), int(max(w for w, _ in dimensions))],
        "image_height_range": [int(min(h for _, h in dimensions)), int(max(h for _, h in dimensions))],
        "original_annotation_bbox_type": "Pascal VOC tight head-region boxes",
    }
    (manifest_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

