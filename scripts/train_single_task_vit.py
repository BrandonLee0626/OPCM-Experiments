import os
import sys
import copy
import threading
import numpy as np
from queue import Queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import cast

from tqdm import tqdm
from timm.models.vision_transformer import VisionTransformer
from src.model import SingleTaskViT
from dataset.dataloader import get_train_dataloader, get_test_dataloader

_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)

# ── Per-dataset hyperparameters ───────────────────────────────────────────────
DATASET_CONFIGS = {
    "MNIST":         {"lr": 1e-4, "epochs": 10,  "bs": 128, "warmup": 1,  "patience": 5,  "mixup": 0.0, "lr_decay": 0.9},
    "FashionMNIST":  {"lr": 1e-4, "epochs": 10,  "bs": 128, "warmup": 1,  "patience": 5,  "mixup": 0.0, "lr_decay": 0.9},
    "EMNIST":        {"lr": 5e-5, "epochs": 20,  "bs": 128, "warmup": 2,  "patience": 8,  "mixup": 0.0, "lr_decay": 0.85},
    "CIFAR10":       {"lr": 2e-5, "epochs": 20,  "bs": 64,  "warmup": 2,  "patience": 7,  "mixup": 0.2, "lr_decay": 0.85},
    "STL10":         {"lr": 2e-5, "epochs": 20,  "bs": 64,  "warmup": 2,  "patience": 7,  "mixup": 0.2, "lr_decay": 0.85},
    "SVHN":          {"lr": 1e-4, "epochs": 15,  "bs": 64,  "warmup": 1,  "patience": 5,  "mixup": 0.0, "lr_decay": 0.9},
    "GTSRB":         {"lr": 1e-4, "epochs": 15,  "bs": 64,  "warmup": 1,  "patience": 5,  "mixup": 0.2, "lr_decay": 0.9},
    "EuroSAT":       {"lr": 2e-5, "epochs": 20,  "bs": 64,  "warmup": 2,  "patience": 7,  "mixup": 0.2, "lr_decay": 0.85},
    "RESISC45":      {"lr": 2e-5, "epochs": 20,  "bs": 64,  "warmup": 2,  "patience": 7,  "mixup": 0.2, "lr_decay": 0.85},
    "PCAM":          {"lr": 5e-5, "epochs": 20,  "bs": 128, "warmup": 2,  "patience": 8,  "mixup": 0.0, "lr_decay": 0.85},
    "RenderedSST2":  {"lr": 2e-5, "epochs": 30,  "bs": 64,  "warmup": 3,  "patience": 10, "mixup": 0.0, "lr_decay": 0.85},
    "Flowers102":    {"lr": 1e-5, "epochs": 60,  "bs": 32,  "warmup": 5,  "patience": 15, "mixup": 0.4, "lr_decay": 0.75},
    "OxfordIIITPet": {"lr": 1e-5, "epochs": 40,  "bs": 64,  "warmup": 3,  "patience": 12, "mixup": 0.2, "lr_decay": 0.8},
    "Food101":       {"lr": 5e-5, "epochs": 30,  "bs": 64,  "warmup": 3,  "patience": 10, "mixup": 0.4, "lr_decay": 0.8},
    "Cars":          {"lr": 1e-5, "epochs": 60,  "bs": 32,  "warmup": 5,  "patience": 15, "mixup": 0.4, "lr_decay": 0.75},
    "CIFAR100":      {"lr": 2e-5, "epochs": 50,  "bs": 64,  "warmup": 5,  "patience": 12, "mixup": 0.4, "lr_decay": 0.8},
    "DTD":           {"lr": 1e-5, "epochs": 80,  "bs": 32,  "warmup": 5,  "patience": 20, "mixup": 0.4, "lr_decay": 0.75},
    "SUN397":        {"lr": 1e-5, "epochs": 50,  "bs": 64,  "warmup": 5,  "patience": 15, "mixup": 0.4, "lr_decay": 0.75},
}


# ── Layer-wise LR decay ───────────────────────────────────────────────────────
def get_param_groups(model: SingleTaskViT, lr: float, weight_decay: float, lr_decay: float):
    """
    ViT layer-wise LR decay:
      head                    → lr
      blocks[N-1] .. blocks[0] → lr * decay^1 .. lr * decay^N
      embeddings / other      → lr * decay^(N+1)
    """
    backbone = cast(VisionTransformer, model.backbone)
    blocks = list(cast(nn.Sequential, backbone.blocks))
    num_layers = len(blocks)

    no_decay = {"bias", "norm"}
    param_groups: list[dict] = []

    def add_group(params: list[tuple[str, nn.Parameter]], group_lr: float) -> None:
        decay_p   = [p for n, p in params if not any(nd in n for nd in no_decay)]
        nodecay_p = [p for n, p in params if     any(nd in n for nd in no_decay)]
        if decay_p:
            param_groups.append({"params": decay_p,   "lr": group_lr, "weight_decay": weight_decay})
        if nodecay_p:
            param_groups.append({"params": nodecay_p, "lr": group_lr, "weight_decay": 0.0})

    # Head
    add_group(list(model.head.named_parameters()), lr)

    # Transformer blocks (later → higher lr)
    for i, block in enumerate(reversed(blocks)):
        block_lr = lr * (lr_decay ** (i + 1))
        add_group(list(block.named_parameters()), block_lr)

    # Everything else in backbone (patch embed, pos embed, norm, ...)
    block_param_ids = {id(p) for block in blocks for p in block.parameters()}
    other = [(n, p) for n, p in backbone.named_parameters() if id(p) not in block_param_ids]
    add_group(other, lr * (lr_decay ** (num_layers + 1)))

    return param_groups


# ── Mixup / CutMix ───────────────────────────────────────────────────────────
def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    cut_rat = (1 - lam) ** 0.5
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0);  x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0);  y2 = min(cy + cut_h // 2, H)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, y, y[idx], lam

def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Training ─────────────────────────────────────────────────────────────────
def train_and_evaluate(model, train_loader, test_loader, device,
                       epochs, lr, warmup_epochs, patience, mixup_alpha, lr_decay, save_path,
                       gpu_id=0, task_name=''):
    use_amp = device.type == 'cuda'
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    param_groups = get_param_groups(model, lr, weight_decay=0.01, lr_decay=lr_decay)
    optimizer = optim.AdamW(param_groups)

    warmup   = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine   = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    scaler = GradScaler('cuda', enabled=use_amp)

    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        correct_train = total_train = 0

        pbar = tqdm(train_loader, desc=f"[cuda:{gpu_id}|{task_name}] {epoch+1}/{epochs}",
                    leave=False, position=gpu_id, dynamic_ncols=True)
        for inputs, labels in pbar:
            inputs  = inputs.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            # Randomly choose Mixup or CutMix
            use_mix = mixup_alpha > 0 and np.random.rand() < 0.5
            if use_mix:
                if np.random.rand() < 0.5:
                    inputs, y_a, y_b, lam = mixup_data(inputs, labels, mixup_alpha)
                else:
                    inputs, y_a, y_b, lam = cutmix_data(inputs, labels, mixup_alpha)

            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                if use_mix:
                    loss = mixed_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(outputs.data, 1)
            total_train  += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct_train/total_train:.3f}")

        scheduler.step()

        # ── Evaluate ──────────────────────────────────────────────────────
        model.eval()
        correct_test = total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast('cuda', enabled=use_amp):
                    outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test  += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc   = correct_test / total_test if total_test > 0 else 0
        current_lr = scheduler.get_last_lr()[0]
        msg = f"[cuda:{gpu_id}|{task_name}] Epoch {epoch+1:>3}/{epochs} | lr={current_lr:.2e} | test acc={test_acc:.4f}"

        if test_acc > best_acc:
            best_acc   = test_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)
            no_improve = 0
            tprint(msg + f"  ✓ best={best_acc:.4f}")
        else:
            no_improve += 1
            tprint(msg)
            if no_improve >= patience:
                tprint(f"[cuda:{gpu_id}|{task_name}] Early stopping (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_acc


# ── Main ─────────────────────────────────────────────────────────────────────
def run_single_task_experiments(vit_arch='vit_base_patch16_224', tasks=None):
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print('Device: CPU')
        devices = [torch.device('cpu')]
    elif n_gpus == 1:
        print(f'Device: {torch.cuda.get_device_name(0)} (1 GPU)')
        devices = [torch.device('cuda:0')]
    else:
        print(f'Device: {n_gpus} GPUs available')
        for i in range(n_gpus):
            print(f'  cuda:{i}  {torch.cuda.get_device_name(i)}')
        devices = [torch.device(f'cuda:{i}') for i in range(n_gpus)]

    print(f'ViT arch: {vit_arch}')

    save_dir = os.path.join('models', 'vit', vit_arch)
    os.makedirs(save_dir, exist_ok=True)

    # Only retrain datasets that did not reach 95% accuracy
    default_targets = {
        "EMNIST", "PCAM", "RenderedSST2", "OxfordIIITPet",
        "Food101", "Cars", "CIFAR100", "DTD", "SUN397",
    }
    target_tasks = [(name, DATASET_CONFIGS[name]) for name in DATASET_CONFIGS
                    if name in (tasks if tasks else default_targets)]

    print(f'\nTraining {len(target_tasks)} tasks across {len(devices)} device(s).\n')

    gpu_queue = Queue()
    for dev in devices:
        gpu_queue.put(dev)

    results = {}
    results_lock = threading.Lock()

    def train_task(task_name, cfg):
        device = gpu_queue.get()
        gpu_id = device.index if device.type == 'cuda' else 0
        save_path = os.path.join(save_dir, f"{vit_arch}_{task_name}.pt")
        try:
            tprint(f"[cuda:{gpu_id}|{task_name}] Starting  arch={vit_arch}  lr={cfg['lr']}  epochs={cfg['epochs']}  bs={cfg['bs']}")
            model = SingleTaskViT(task_name=task_name, vit_arch=vit_arch).to(device)
            if os.path.exists(save_path):
                model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
                tprint(f"[cuda:{gpu_id}|{task_name}] Loaded existing checkpoint")
            train_loader = get_train_dataloader(task_name, batch_size=cfg['bs'])
            test_loader  = get_test_dataloader(task_name,  batch_size=cfg['bs'])

            _, best_acc = train_and_evaluate(
                model, train_loader, test_loader, device,
                epochs        = cfg['epochs'],
                lr            = cfg['lr'],
                warmup_epochs = cfg['warmup'],
                patience      = cfg['patience'],
                mixup_alpha   = cfg['mixup'],
                lr_decay      = cfg['lr_decay'],
                save_path     = save_path,
                gpu_id        = gpu_id,
                task_name     = task_name,
            )
            with results_lock:
                results[task_name] = best_acc
            tprint(f"[cuda:{gpu_id}|{task_name}] Done  best={best_acc:.4f}")
        except Exception as e:
            tprint(f"[cuda:{gpu_id}|{task_name}] ERROR: {e}")
        finally:
            gpu_queue.put(device)

    threads = [threading.Thread(target=train_task, args=(name, cfg), daemon=True)
               for name, cfg in target_tasks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\n" + "="*60)
    print("  Final Results")
    print("="*60)
    for name in [n for n, _ in target_tasks]:
        if name in results:
            flag = "" if results[name] >= 0.95 else "  <- below 95%"
            print(f"  {name:<20} {results[name]:.4f}{flag}")
        else:
            print(f"  {name:<20} FAILED")

    result_path = os.path.join('results', 'single_task_accuracy', 'vit',
                               f'result_vit_{vit_arch}.txt')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'a') as f:
        for name, acc in results.items():
            f.write(f'{name} {acc}\n')
    print(f"\nResults appended to {result_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit_arch',
                        choices=['vit_base_patch32_224', 'vit_base_patch16_224', 'vit_large_patch16_224'],
                        default='vit_base_patch16_224')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Specific tasks to train (default: below-95%% targets)')
    args = parser.parse_args()
    run_single_task_experiments(vit_arch=args.vit_arch, tasks=args.tasks)
