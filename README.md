# OPCM Reproduction & Experiments

This repository reproduces **OPCM (Orthogonal Projection-based Continual Merging)**, published at NeurIPS 2025, and conducts additional experiments on top of it.

Official repository: [tanganke/opcm](https://github.com/tanganke/opcm) | Project page: [tanganke.github.io/opcm](https://tanganke.github.io/opcm/)

## Overview

OPCM is a model merging method that uses orthogonal projection to mitigate task interference when merging multiple fine-tuned models. At each step, an incoming task vector is projected onto the null space of the accumulated merged task vector, then rescaled to maintain consistent norm. This repo reimplements the core method and explores further experimental settings beyond those in the original paper.

## Structure

```
OPCM/
├── opcm.py                        # OPCM class + main experiment runner
├── src/
│   ├── model.py                   # SingleTask/MultiTask model definitions (ViT & CLIP)
│   ├── task_vector.py             # TaskVector class
│   ├── utils.py                   # SVD, evaluation, task vector loading utilities
│   └── csv_logger.py              # CSV-based experiment logger
├── scripts/
│   ├── train_single_task_vit.py   # Fine-tune ViT models per task (multi-GPU)
│   ├── train_single_task_clip.py  # Fine-tune CLIP models per task (multi-GPU)
│   └── evaluate_model.py          # Evaluate saved single-task checkpoints
├── dataset/
│   ├── dataloader.py              # Dataset loaders
│   └── classnames.py              # Class names & templates for CLIP zero-shot heads
└── results/                       # CSV logs and single-task accuracy baselines
```

## Supported Backbones

| Model | Architectures |
|-------|--------------|
| ViT (timm) | `vit_base_patch32_224`, `vit_base_patch16_224`, `vit_large_patch16_224` |
| CLIP (open_clip) | `ViT-B-32`, `ViT-B-16`, `ViT-L-14` |

CLIP supports two classification head types:
- **zeroshot** — class prototypes built from text embeddings (no extra parameters)
- **linear** — task-specific linear layers trained on top of frozen visual features

## Supported Tasks

18 image classification datasets:
MNIST, FashionMNIST, EMNIST, CIFAR10, CIFAR100, STL10, SVHN, GTSRB, EuroSAT, RESISC45, PCAM, RenderedSST2, Flowers102, OxfordIIITPet, Food101, Cars, DTD, SUN397

Three preset task subsets are available via `--num_tasks`:

| `--num_tasks` | Tasks |
|---------------|-------|
| `8` | SUN397, Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD |
| `14` | above 8 + Flowers102, PCAM, OxfordIIITPet, STL10, CIFAR100, FashionMNIST |
| `all` | all 18 tasks (default) |

## Usage

### 1. Fine-tune single-task models

```bash
# ViT
python scripts/train_single_task_vit.py --vit_arch vit_base_patch16_224

# CLIP (zero-shot head, default)
python scripts/train_single_task_clip.py --clip_arch ViT-B-32 --head_type zeroshot

# CLIP (linear head)
python scripts/train_single_task_clip.py --clip_arch ViT-B-32 --head_type linear

# Train specific tasks only
python scripts/train_single_task_clip.py --clip_arch ViT-B-32 --tasks CIFAR10 SUN397
```

Checkpoints are saved to `models/{vit,clip_zeroshot,clip_linear}/{arch}/`.

### 2. Evaluate single-task checkpoints

```bash
python scripts/evaluate_model.py --model clip --clip_arch ViT-B-32 --head_type zeroshot
python scripts/evaluate_model.py --model vit  --vit_arch vit_base_patch16_224
```

Results are appended to `results/single_task_accuracy/{model_type}/result_*.txt`.

### 3. Run OPCM merging

```bash
# CLIP with zero-shot head (default)
python opcm.py --model clip --clip_arch ViT-B-32 --head_type zeroshot --alpha 0.5

# CLIP with linear head
python opcm.py --model clip --clip_arch ViT-B-32 --head_type linear --alpha 0.5

# ViT
python opcm.py --model vit --vit_arch vit_base_patch16_224 --alpha 0.5

# Use only 8-task subset
python opcm.py --model clip --clip_arch ViT-B-32 --num_tasks 8

# Use 14-task subset with random task order
python opcm.py --model clip --clip_arch ViT-B-32 --num_tasks 14 --shuffle
```

#### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--alpha` | `0.5` | SVD energy threshold for null-space split rank |
| `--model` | `clip` | Backbone type: `vit` or `clip` |
| `--clip_arch` | `ViT-B-32` | CLIP architecture (used when `--model clip`) |
| `--vit_arch` | `vit_base_patch16_224` | ViT architecture (used when `--model vit`) |
| `--head_type` | `zeroshot` | CLIP head type: `zeroshot` or `linear` |
| `--num_tasks` | `all` | Task subset: `8`, `14`, or `all` |
| `--shuffle` | `False` | Randomly shuffle task merge order |
| `--monitor` | `csv` | Logging backend: `csv`, `mlflow`, or `both` |

## Logging & Results

CSV results are saved to `results/{timestamp}_{model}_{head}_{arch}_tasks{num_tasks}_alpha{alpha}[_shuffled]/`:

| File | Contents |
|------|----------|
| `accuracy.csv` | Per-task accuracy after each merge step |
| `drop_vs_single.csv` | Accuracy drop relative to single-task baseline |
| `forgetting.csv` | Forgetting since each task's first merge |
| `projection_metrics.csv` | Frobenius inner product, approximation error, average split rank |
| `layer_ranks.csv` | Cumulative split rank per linear layer |
| `config.json` | Run configuration: model, arch, alpha, num_tasks, shuffle, task_order, single-task accuracy baselines |

MLflow logging is also supported (`--monitor mlflow` or `--monitor both`).

## Multi-GPU Support

Both training and evaluation automatically detect available GPUs. When multiple GPUs are present:
- **Training**: tasks are distributed across GPUs via a thread-safe queue (one task per GPU at a time).
- **Evaluation (OPCM)**: tasks are evaluated in parallel using GPU replicas, with results aggregated after each merge step.

## Reference

> **OPCM: Orthogonal Projection-based Continual Merging**
> NeurIPS 2025
> [https://github.com/tanganke/opcm](https://github.com/tanganke/opcm)
