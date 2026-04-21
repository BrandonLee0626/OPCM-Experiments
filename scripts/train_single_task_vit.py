import os
import sys
import copy
import itertools
import random
import threading
import numpy as np
from queue import Queue

import mlflow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parallel import get_devices, init_cuda_contexts
from scripts.train_utils import SUPPORTED_DATASETS, tprint, get_config, save_results

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

MODEL_ROOT = '/data/leeminjae0626/model/OPCM'

# lp: backbone frozen, high lr, large batch
# ft/lp-ft: full fine-tuning with layer-wise lr decay
_DEFAULTS = {
    "lp": {"lr": 5e-3, "bs": 256, "warmup": 1, "patience": 10, "mixup": 0.0, "lr_decay": 1.0},
    "ft": {"lr": 2e-5, "bs": 64,  "warmup": 2, "patience": 10, "mixup": 0.2, "lr_decay": 0.85},
}


# ── Layer-wise LR decay ───────────────────────────────────────────────────────
def get_param_groups(model: SingleTaskViT, lr: float, weight_decay: float, lr_decay: float):
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

    add_group(list(model.head.named_parameters()), lr)

    for i, block in enumerate(reversed(blocks)):
        add_group(list(block.named_parameters()), lr * (lr_decay ** (i + 1)))

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
                       lr, warmup_epochs, patience, mixup_alpha, lr_decay, save_path,
                       gpu_id=0, task_name='', mode='ft'):
    use_amp = device.type == 'cuda'
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if mode == 'lp':
        optimizer = optim.AdamW(model.head.parameters(), lr=lr, weight_decay=0.01)
    else:  # ft / lp-ft
        param_groups = get_param_groups(model, lr, weight_decay=0.01, lr_decay=lr_decay)
        optimizer = optim.AdamW(param_groups)

    warmup   = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine   = CosineAnnealingLR(optimizer, T_max=1000, eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    scaler = GradScaler('cuda', enabled=use_amp)

    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in itertools.count():
        model.train()
        if mode == 'lp':
            model.backbone.eval()
        correct_train = total_train = 0
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[cuda:{gpu_id}|{task_name}] {epoch+1}/∞",
                    leave=False, position=gpu_id, dynamic_ncols=True)
        for inputs, labels in pbar:
            inputs  = inputs.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

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

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train  += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct_train/total_train:.3f}")

        scheduler.step()

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
        mlflow.log_metrics({
            "train_acc": float(correct_train / total_train) if total_train > 0 else 0.0,
            "test_acc":  float(test_acc),
            "lr":        float(current_lr),
            "loss":      float(total_loss / len(train_loader)),
        }, step=epoch)
        msg = f"[cuda:{gpu_id}|{task_name}] Epoch {epoch+1:>3}/∞ | lr={current_lr:.2e} | test acc={test_acc*100:.2f}%"

        if test_acc > best_acc:
            best_acc   = test_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)
            no_improve = 0
            tprint(msg + f"  ✓ best={best_acc*100:.2f}%")
        else:
            no_improve += 1
            tprint(msg)
            if no_improve >= patience:
                tprint(f"[cuda:{gpu_id}|{task_name}] Early stopping (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    return model, best_acc


# ── Main ─────────────────────────────────────────────────────────────────────
def run_single_task_experiments(vit_arch='vit_base_patch16_224', tasks=None, mode='ft',
                                num_workers: int | None = None):
    devices = get_devices()

    # Cap total DataLoader worker processes to ~cpu_count so augmentation doesn't saturate CPU.
    # Each concurrent task spawns 2 loaders (train+test), each with num_workers processes.
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) // (2 * len(devices)))
    print(f'DataLoader num_workers={num_workers} (cpu_count={os.cpu_count()}, devices={len(devices)})')

    mlflow.set_experiment(f"single_task_vit_{vit_arch}")

    print(f'ViT arch: {vit_arch}, mode: {mode}')

    save_root = os.path.join(MODEL_ROOT, 'vit')
    save_dir  = os.path.join(save_root, vit_arch, mode)
    os.makedirs(save_dir, exist_ok=True)

    all_datasets = tasks if tasks else SUPPORTED_DATASETS
    candidate_tasks = [(name, get_config(_DEFAULTS, mode)) for name in all_datasets if name in SUPPORTED_DATASETS]

    if mode == 'lp-ft':
        missing = [name for name, _ in candidate_tasks
                   if not os.path.exists(os.path.join(save_root, vit_arch, 'lp', f'{vit_arch}_{name}.pt'))]
        if missing:
            print(f'[Warning] No LP checkpoint found for {len(missing)} task(s) — skipping: {missing}')
        target_tasks = [(name, cfg) for name, cfg in candidate_tasks if name not in missing]
    else:
        target_tasks = candidate_tasks

    if not target_tasks:
        print('No tasks to run. Exiting.')
        return

    print(f'\nTraining {len(target_tasks)} tasks across {len(devices)} device(s).\n')

    init_cuda_contexts(devices)

    gpu_queue = Queue()
    for dev in devices:
        gpu_queue.put(dev)

    results = {}
    results_lock = threading.Lock()
    model_init_lock = threading.Lock()

    def train_task(task_name, cfg):
        device = gpu_queue.get()
        gpu_id = device.index if device.type == 'cuda' else 0
        save_path = os.path.join(save_dir, f"{vit_arch}_{task_name}.pt")
        try:
            with mlflow.start_run(run_name=task_name):
                mlflow.log_params({
                    "arch":      vit_arch,
                    "mode":      mode,
                    "task":      task_name,
                    "lr":        cfg['lr'],
                    "bs":        cfg['bs'],
                    "warmup":    cfg['warmup'],
                    "patience":  cfg['patience'],
                    "mixup":     cfg['mixup'],
                    "lr_decay":  cfg['lr_decay'],
                })
                tprint(f"[cuda:{gpu_id}|{task_name}] Starting  arch={vit_arch}  mode={mode}  lr={cfg['lr']:.2e}  bs={cfg['bs']}  patience={cfg['patience']}")
                with model_init_lock:
                    model = SingleTaskViT(task_name=task_name, vit_arch=vit_arch).to(device)
                    if mode == 'lp':
                        model.backbone.requires_grad_(False)
                    if mode == 'lp-ft':
                        lp_path = os.path.join(save_root, vit_arch, 'lp', f"{vit_arch}_{task_name}.pt")
                        if not os.path.exists(lp_path):
                            raise FileNotFoundError(f"LP checkpoint not found: {lp_path}. Run --mode lp first.")
                        model.load_state_dict(torch.load(lp_path, map_location=device, weights_only=True), strict=False)
                        tprint(f"[cuda:{gpu_id}|{task_name}] Loaded LP checkpoint from {lp_path}")
                    elif os.path.exists(save_path):
                        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True), strict=False)
                        tprint(f"[cuda:{gpu_id}|{task_name}] Loaded existing checkpoint")
                train_loader = get_train_dataloader(task_name, batch_size=cfg['bs'], num_workers=num_workers)
                test_loader  = get_test_dataloader(task_name,  batch_size=cfg['bs'], num_workers=num_workers)

                _, best_acc = train_and_evaluate(
                    model, train_loader, test_loader, device,
                    lr            = cfg['lr'],
                    warmup_epochs = cfg['warmup'],
                    patience      = cfg['patience'],
                    mixup_alpha   = cfg['mixup'],
                    lr_decay      = cfg['lr_decay'],
                    save_path     = save_path,
                    gpu_id        = gpu_id,
                    task_name     = task_name,
                    mode          = mode,
                )
                mlflow.log_metric("best_acc", best_acc)
                with results_lock:
                    results[task_name] = best_acc
                tprint(f"[cuda:{gpu_id}|{task_name}] Done  best={best_acc*100:.2f}%")
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
            print(f"  {name:<20} {results[name]*100:.2f}%{flag}")
        else:
            print(f"  {name:<20} FAILED")

    result_path = os.path.join('results', 'single_task_accuracy', 'vit', mode,
                               f'result_vit_linear_{mode}_{vit_arch}.json')
    save_results(result_path, results)


if __name__ == '__main__':
    import argparse
    import multiprocessing
    # DataLoader workers are forked inside threading.Thread workers.
    # On Linux, 'fork' inherits locked mutexes from sibling threads → deadlock.
    # 'forkserver' spawns workers via a clean server process that has no thread state.
    multiprocessing.set_start_method('forkserver', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--vit_arch',
                        choices=['vit_base_patch32_224', 'vit_base_patch16_224', 'vit_large_patch16_224'],
                        default='vit_base_patch16_224')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Specific tasks to train (default: below-95%% targets)')
    parser.add_argument('--mode', choices=['ft', 'lp', 'lp-ft'], default='lp-ft',
                        help='Training mode: ft (full fine-tuning), lp (linear probe), '
                             'lp-ft (LP then FT) (default: lp-ft)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers per loader (default: cpu_count / (2 * num_gpus))')
    args = parser.parse_args()
    run_single_task_experiments(vit_arch=args.vit_arch, tasks=args.tasks, mode=args.mode,
                                num_workers=args.num_workers)
