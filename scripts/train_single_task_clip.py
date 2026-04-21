import os
import sys
import json
import copy
import math
import random
import itertools
import threading
from queue import Queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parallel import get_devices, init_cuda_contexts

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from tqdm import tqdm
from src.model import SingleTaskCLIP, SingleTaskCLIPLinear
from dataset.dataloader import get_train_dataloader, get_test_dataloader

_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


SUPPORTED_DATASETS = [
    "MNIST", "FashionMNIST", "EMNIST", "CIFAR10", "STL10", "SVHN",
    "GTSRB", "EuroSAT", "RESISC45", "PCAM", "RenderedSST2",
    "Flowers102", "OxfordIIITPet", "Food101", "Cars", "CIFAR100", "DTD", "SUN397",
    "Country211", "Aircraft",
]

# lp: backbone frozen, high lr, large batch
# ft/lp-ft: full fine-tuning with layer-wise lr decay
_DEFAULTS = {
    "lp": {"lr": 5e-3, "bs": 256, "warmup": 1, "patience": 10},
    "ft": {"lr": 1e-5, "bs": 64,  "warmup": 2, "patience": 7},
}

def get_config(mode: str) -> dict:
    base = _DEFAULTS["lp" if mode == "lp" else "ft"]
    lr = base["lr"] * (10 ** random.uniform(-0.5, 0.5))
    return {**base, "lr": lr}

# ── Layer-wise LR decay for CLIP visual encoder ───────────────────────────────
def get_param_groups(model, lr: float, weight_decay: float, lr_decay: float = 0.9):
    backbone = model.backbone
    blocks = list(backbone.transformer.resblocks)
    num_layers = len(blocks)

    no_decay = {"bias", "norm", "ln"}
    param_groups: list[dict] = []

    def add_group(params, group_lr):
        decay_p   = [p for n, p in params if not any(nd in n for nd in no_decay)]
        nodecay_p = [p for n, p in params if     any(nd in n for nd in no_decay)]
        if decay_p:
            param_groups.append({"params": decay_p,   "lr": group_lr, "weight_decay": weight_decay})
        if nodecay_p:
            param_groups.append({"params": nodecay_p, "lr": group_lr, "weight_decay": 0.0})

    if hasattr(model, 'head'):
        param_groups.append({"params": list(model.head.parameters()), "lr": lr, "weight_decay": weight_decay})

    for i, block in enumerate(reversed(blocks)):
        add_group(list(block.named_parameters()), lr * (lr_decay ** (i + 1)))

    block_param_ids = {id(p) for block in blocks for p in block.parameters()}
    other = [(n, p) for n, p in backbone.named_parameters() if id(p) not in block_param_ids]
    add_group(other, lr * (lr_decay ** (num_layers + 1)))

    return param_groups


# ── Training ──────────────────────────────────────────────────────────────────
def train_and_evaluate(model, train_loader, test_loader, device,
                       lr, warmup_epochs, patience, save_path,
                       gpu_id=0, task_name='', mode='ft'):
    use_amp = device.type == 'cuda'
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if mode == 'lp':
        optimizer = optim.AdamW(model.head.parameters(), lr=lr, weight_decay=0.01)
    else:  # ft / lp-ft
        param_groups = get_param_groups(model, lr, weight_decay=0.01)
        optimizer = optim.AdamW(param_groups)

    warmup    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine    = CosineAnnealingLR(optimizer, T_max=1000, eta_min=lr * 0.01)
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

        pbar = tqdm(train_loader, desc=f"[cuda:{gpu_id}|{task_name}] {epoch+1}/∞",
                    leave=False, position=gpu_id, dynamic_ncols=True)
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
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


# ── Main ──────────────────────────────────────────────────────────────────────
def run_single_task_experiments(clip_arch='ViT-B-32', tasks=None, head_type='zeroshot', mode='ft'):
    devices = get_devices()

    if head_type == 'zeroshot' and mode != 'ft':
        print(f'[Warning] --mode={mode} is not applicable to head_type=zeroshot; using ft.')
        mode = 'ft'

    save_root = f'models/clip_{head_type}'
    print(f'CLIP arch: {clip_arch}, head_type: {head_type}, mode: {mode}')

    save_dir = os.path.join(save_root, clip_arch, mode)
    os.makedirs(save_dir, exist_ok=True)

    all_datasets = tasks if tasks else SUPPORTED_DATASETS
    candidate_tasks = [(name, get_config(mode)) for name in all_datasets if name in SUPPORTED_DATASETS]

    if mode == 'lp-ft':
        missing = [name for name, _ in candidate_tasks
                   if not os.path.exists(os.path.join(save_root, clip_arch, 'lp', f'clip_{clip_arch}_{name}.pt'))]
        if missing:
            print(f'[Warning] No LP checkpoint found for {len(missing)} task(s) — skipping: {missing}')
        target_tasks = [(name, c) for name, c in candidate_tasks if name not in missing]
    else:
        target_tasks = candidate_tasks

    if not target_tasks:
        print('No tasks to run. Exiting.')
        return

    print(f'\nTraining {len(target_tasks)} tasks across {len(devices)} device(s).\n')

    # Without this, threads racing to initialize CUDA while another thread is
    # inside AMP/GradScaler causes currentStreamCaptureStatusMayInitCtx errors.
    init_cuda_contexts(devices)

    gpu_queue = Queue()
    for dev in devices:
        gpu_queue.put(dev)

    results = {}
    results_lock = threading.Lock()
    # Serialize model creation: open_clip text encoding + .to(device) must not
    # race with another thread's CUDA graph capture.
    model_init_lock = threading.Lock()

    def train_task(task_name, cfg):
        device = gpu_queue.get()
        gpu_id = device.index if device.type == 'cuda' else 0
        save_path = os.path.join(save_dir, f'clip_{clip_arch}_{task_name}.pt')
        try:
            tprint(f"[cuda:{gpu_id}|{task_name}] Starting  arch={clip_arch}  head={head_type}  mode={mode}  lr={cfg['lr']}  bs={cfg['bs']}  patience={cfg['patience']}")
            with model_init_lock:
                if head_type == 'linear':
                    model = SingleTaskCLIPLinear(task_name=task_name, clip_arch=clip_arch).to(device)
                else:
                    model = SingleTaskCLIP(task_name=task_name, clip_arch=clip_arch).to(device)
                if mode == 'lp':
                    model.backbone.requires_grad_(False)
                if mode == 'lp-ft':
                    lp_path = os.path.join(save_root, clip_arch, 'lp', f'clip_{clip_arch}_{task_name}.pt')
                    if not os.path.exists(lp_path):
                        raise FileNotFoundError(f"LP checkpoint not found: {lp_path}. Run --mode lp first.")
                    model.load_state_dict(torch.load(lp_path, map_location=device, weights_only=True), strict=False)
                    tprint(f"[cuda:{gpu_id}|{task_name}] Loaded LP checkpoint from {lp_path}")
                elif os.path.exists(save_path):
                    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True), strict=False)
                    tprint(f"[cuda:{gpu_id}|{task_name}] Loaded existing checkpoint")
            train_loader = get_train_dataloader(task_name, batch_size=cfg['bs'], model_type='clip')
            test_loader  = get_test_dataloader(task_name,  batch_size=cfg['bs'], model_type='clip')

            _, best_acc = train_and_evaluate(
                model, train_loader, test_loader, device,
                lr            = cfg['lr'],
                warmup_epochs = cfg['warmup'],
                patience      = cfg['patience'],
                save_path     = save_path,
                gpu_id        = gpu_id,
                task_name     = task_name,
                mode          = mode,
            )
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
    for name, _ in target_tasks:
        if name in results:
            print(f"  {name:<20} {results[name]*100:.2f}%")
        else:
            print(f"  {name:<20} FAILED")

    result_path = os.path.join('results', 'single_task_accuracy', 'clip', mode,
                               f'result_clip_{head_type}_{mode}_{clip_arch}.json')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    existing = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
    existing.update({name: round(acc * 100, 4) for name, acc in results.items()})
    with open(result_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {result_path}")


if __name__ == '__main__':
    import argparse
    import multiprocessing
    # DataLoader workers are forked inside threading.Thread workers.
    # On Linux, 'fork' inherits locked mutexes from sibling threads → deadlock.
    # 'forkserver' spawns workers via a clean server process that has no thread state.
    multiprocessing.set_start_method('forkserver', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_arch', choices=['ViT-B-32', 'ViT-B-16', 'ViT-L-14'], default='ViT-B-32')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Specific tasks to train (default: all)')
    parser.add_argument('--head_type', choices=['zeroshot', 'linear'], default='linear',
                        help='Classification head type: zeroshot (text embeddings) or linear '
                             '(default: linear)')
    parser.add_argument('--mode', choices=['ft', 'lp', 'lp-ft'], default='lp-ft',
                        help='Training mode: ft (full fine-tuning), lp (linear probe), '
                             'lp-ft (LP then FT). Ignored when head_type=zeroshot. (default: lp-ft)')
    args = parser.parse_args()

    run_single_task_experiments(clip_arch=args.clip_arch, tasks=args.tasks,
                                head_type=args.head_type, mode=args.mode)
