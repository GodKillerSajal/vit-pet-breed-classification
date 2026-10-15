"""Train and compare a ResNet-18 baseline and a fine-tuned ViT on Oxford-IIIT Pet."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SEED = 42
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PetDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform):
        self.rows = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), int(row.breed_id), str(row.image_id)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(name: str, n_classes: int) -> nn.Module:
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return model
    if name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, n_classes)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for block in list(model.encoder.layers)[-3:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in model.heads.parameters():
            parameter.requires_grad = True
        return model
    raise ValueError(f"Unknown model: {name}")


def _loader(frame, training: bool, batch_size: int) -> DataLoader:
    if training:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.72, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    return DataLoader(
        PetDataset(frame, transform),
        batch_size=batch_size,
        shuffle=training,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, max_batches=0):
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    labels_all: list[int] = []
    predictions_all: list[int] = []
    for batch_number, (images, labels, _ids) in enumerate(loader):
        if max_batches and batch_number >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        predictions_all.extend(logits.detach().argmax(dim=1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(labels_all, predictions_all)) if labels_all else 0.0,
        "macro_f1": float(f1_score(labels_all, predictions_all, average="macro", zero_division=0)) if labels_all else 0.0,
    }


@torch.no_grad()
def predict(model, loader, device, max_batches=0):
    model.eval()
    y_true, y_pred, y_prob, image_ids = [], [], [], []
    for batch_number, (images, labels, ids) in enumerate(loader):
        if max_batches and batch_number >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        probs = logits.float().softmax(dim=1).cpu().numpy()
        y_true.extend(labels.tolist())
        y_pred.extend(probs.argmax(axis=1).tolist())
        y_prob.extend(probs.tolist())
        image_ids.extend(ids)
    return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob), image_ids


def evaluate_model(model, loader, device, class_names, output_dir: Path, model_name: str, max_batches=0):
    y_true, y_pred, y_prob, image_ids = predict(model, loader, device, max_batches=max_batches)
    labels = np.arange(len(class_names))
    top5 = np.argsort(y_prob, axis=1)[:, -min(5, len(class_names)) :]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top5_accuracy": float(np.mean([label in candidates for label, candidates in zip(y_true, top5)])),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "class_metrics": report,
        "test_examples": int(len(y_true)),
    }
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels), index=class_names, columns=class_names
    ).to_csv(model_dir / "confusion_matrix.csv")
    prediction_rows = []
    for true_id, pred_id, probs, image_id in zip(y_true, y_pred, y_prob, image_ids):
        prediction_rows.append(
            {
                "image_id": image_id,
                "true_breed": class_names[true_id],
                "predicted_breed": class_names[pred_id],
                "confidence": float(probs[pred_id]),
                "correct": bool(true_id == pred_id),
            }
        )
    pd.DataFrame(prediction_rows).to_csv(model_dir / "test_predictions.csv", index=False)
    pd.DataFrame([row for row in prediction_rows if not row["correct"]]).sort_values(
        "confidence", ascending=False
    ).to_csv(model_dir / "high_confidence_errors.csv", index=False)
    return {key: value for key, value in metrics.items() if key != "class_metrics"}


def train_one(
    name: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    class_names: list[str],
    output_dir: Path,
    device,
    epochs: int,
    batch_size: int,
    max_batches: int,
) -> dict[str, object]:
    train_loader = _loader(train_frame, True, batch_size)
    validation_loader = _loader(validation_frame, False, max(2, batch_size))
    test_loader = _loader(test_frame, False, max(2, batch_size))
    model = make_model(name, len(class_names)).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4 if name == "vit_b_16" else 2e-4,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.08)
    model_output = output_dir / name
    model_output.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_path = model_output / "best_state.pt"
    history: list[dict[str, float]] = []

    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params(
            {
                "architecture": name,
                "train_samples": len(train_frame),
                "validation_samples": len(validation_frame),
                "test_samples": len(test_frame),
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": SEED,
                "split": "original trainval/test; deterministic 15% trainval validation split",
                "device": str(device),
            }
        )
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            train_metrics = run_epoch(
                model, train_loader, loss_fn, device, optimizer, scaler, max_batches=max_batches
            )
            val_metrics = run_epoch(
                model, validation_loader, loss_fn, device, max_batches=max_batches
            )
            scheduler.step()
            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": val_metrics["loss"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "seconds": time.perf_counter() - started,
            }
            history.append(record)
            mlflow.log_metrics({k: float(v) for k, v in record.items() if k != "epoch"}, step=epoch)
            print(f"{name} epoch {epoch}/{epochs}: {record}")
            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                torch.save(model.state_dict(), best_path)
        (model_output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        test_metrics = evaluate_model(
            model, test_loader, device, class_names, output_dir, name, max_batches=max_batches
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_artifact(str(model_output / "test_metrics.json"))
        mlflow.log_artifact(str(best_path))
    return {"model": name, **test_metrics, "best_validation_macro_f1": best_f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    project = Path(__file__).resolve().parent
    data_default = project.parents[1] / "data" / "oxford-iiit-pet"
    parser.add_argument("--data-root", type=Path, default=data_default)
    parser.add_argument("--output-dir", type=Path, default=project / "outputs")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0, help="Optional short debug run; 0 uses all data")
    args = parser.parse_args()
    seed_everything(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.90)
    else:
        torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}; CUDA available: {torch.cuda.is_available()}")

    manifest_dir = args.data_root / "manifests"
    train_frame = pd.read_csv(manifest_dir / "train.csv")
    validation_frame = pd.read_csv(manifest_dir / "validation.csv")
    test_frame = pd.read_csv(manifest_dir / "test.csv")
    class_names = (
        train_frame[["breed_id", "breed"]]
        .drop_duplicates()
        .sort_values("breed_id")["breed"]
        .tolist()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(args.output_dir / 'mlflow.db').resolve().as_posix()}")
    mlflow.set_experiment("oxford-pet-vit-vs-resnet")
    comparisons = []
    with mlflow.start_run(run_name="transfer-learning-comparison"):
        for model_name in ("resnet18", "vit_b_16"):
            comparisons.append(
                train_one(
                    model_name,
                    train_frame,
                    validation_frame,
                    test_frame,
                    class_names,
                    args.output_dir,
                    device,
                    args.epochs,
                    args.batch_size,
                    args.max_batches,
                )
            )
        (args.output_dir / "metrics.json").write_text(
            json.dumps({"device": str(device), "comparison": comparisons}, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(args.output_dir / "metrics.json"))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()

