# Fine-grained pet breed classification with a Vision Transformer

Fine-tune a PyTorch ViT on 37 cat and dog breeds and compare it with a pretrained ResNet-18 baseline. The experiment reports accuracy, top-5 accuracy, macro-F1, per-breed precision/recall/F1, confusion matrices, and misclassified test examples.

## Data and provenance

`prepare_data.py` downloads the Oxford-IIIT Pet image mirror and the original Oxford annotations, then uses the original `list.txt`, `trainval.txt`, and `test.txt` assignments. It matches images to original annotations, removes unannotated mirror rows and duplicates, validates decodability, creates a deterministic breed-stratified validation set from trainval, and writes an audit plus distribution chart. Original Oxford annotations provide breed/species and Pascal VOC head-box/trimap labels; this classifier uses the 37-breed labels. See the [Oxford dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/) and [image mirror dataset card](https://huggingface.co/datasets/enterprise-explorers/oxford-pets). Dataset files are not committed.

## Run

From the workspace root:

```powershell
portfolio_projects\.venv\Scripts\python.exe portfolio_projects\expanded\vit-pet-breed-classification\prepare_data.py
portfolio_projects\.venv\Scripts\python.exe portfolio_projects\expanded\vit-pet-breed-classification\train.py --epochs 4
```

The default data root is `portfolio_projects/data/oxford-iiit-pet`, shared with the Faster R-CNN project. The training script uses CUDA and mixed precision when available, with batch size 4 suited to a 4 GB laptop GPU. Outputs and checkpoints are kept under `outputs/` and are excluded from Git.

## Measured evaluation

Run on an RTX 3050 Laptop GPU with four training epochs per model. The official 3,669-image Oxford test split was held out from training and checkpoint selection; checkpoint selection used the 551-image validation split. Metrics below are from `outputs/metrics.json`.

| Model | Test accuracy | Top-5 accuracy | Macro-F1 | Best validation macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| ResNet-18 | 87.16% | 98.80% | 0.8693 | 0.8929 |
| ViT-B/16 | 94.44% | 99.32% | 0.9435 | 0.9492 |

The ViT performed better on this split, but the result is specific to the Oxford pet-breed task and does not establish that it will outperform ResNet on other image domains. Both runs save per-breed metrics, confusion matrices, test predictions, and high-confidence mistakes under their model-specific output directories for inspection.

