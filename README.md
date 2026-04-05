# OPCM Reproduction & Experiments

This repository reproduces **OPCM (Orthogonal Projection for Continual Merging)**, published at NeurIPS 2025, and conducts additional experiments on top of it.

Official repository: [tanganke/opcm](https://github.com/tanganke/opcm) | Project page: [tanganke.github.io/opcm](https://tanganke.github.io/opcm/)

## Overview

OPCM is a model merging method that uses orthogonal projection to mitigate task interference when merging multiple fine-tuned models. This repo reimplements the core method and explores further experimental settings beyond those in the original paper.

## Structure

```
OPCM/
├── opcm.py                 # OPCM class + main experiment runner
├── src/
│   ├── model.py            # SingleTaskViT / MultiTaskViT definitions
│   ├── task_vector.py      # TaskVector class
│   └── utils.py            # SVD, evaluation, task vector loading utilities
├── scripts/
│   ├── train_single_task_model.py  # Single-task fine-tuning
│   └── evaluate_model.py           # Per-task accuracy evaluation
└── dataset/                # Dataset directory (raw data excluded)
```

## Usage

```bash
# 1. Train single-task models
python scripts/train_single_task_model.py

# 2. Evaluate single-task models
python scripts/evaluate_model.py

# 3. Run OPCM merging experiment
python opcm.py --alpha 0.5
```

## Reference

> **OPCM: Orthogonal Projection for Continual Merging**
> NeurIPS 2025
> [https://github.com/tanganke/opcm](https://github.com/tanganke/opcm)
