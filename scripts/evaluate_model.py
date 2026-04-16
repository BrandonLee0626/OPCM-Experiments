import os
import sys
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse

from src.model import SingleTaskViT, SingleTaskCLIP, SingleTaskCLIPLinear
from src.parallel import get_devices, init_cuda_contexts
from dataset.dataloader import get_test_dataloader

_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            # non_blocking=True: overlaps CPU→GPU transfer with computation when pin_memory=True
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0


def evaluate_saved_models(model_type='vit', clip_arch='ViT-B-32', vit_arch='vit_base_patch16_224',
                          head_type='zeroshot', mode='ft'):
    devices = get_devices()
    n_parallel = len(devices)

    if model_type == 'clip':
        arch = clip_arch
        if head_type == 'zeroshot' and mode != 'ft':
            print(f'[Warning] --mode={mode} is not applicable to head_type=zeroshot; using ft.')
            mode = 'ft'
        models_dir = os.path.join('models', f'clip_{head_type}', arch, mode)
        prefix = f'clip_{arch}_'
        def make_model(task_name, device):
            if head_type == 'linear':
                return SingleTaskCLIPLinear(task_name=task_name, clip_arch=arch).to(device)
            return SingleTaskCLIP(task_name=task_name, clip_arch=arch).to(device)
    else:
        arch = vit_arch
        head_type = 'linear'
        models_dir = os.path.join('models', 'vit', arch, mode)
        prefix = f'{arch}_'
        def make_model(task_name, device):
            return SingleTaskViT(task_name=task_name, vit_arch=arch).to(device)

    result_path = os.path.join(
        'results', 'single_task_accuracy', model_type, mode,
        f'result_{model_type}_{head_type}_{mode}_{arch}.json',
    )
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    print(f'Model: {model_type} ({arch}), head_type: {head_type}, mode: {mode}')
    print(f'Models dir: {models_dir}\n{"="*50}')

    tasks = [
        (file_name.removeprefix(prefix).removesuffix('.pt'),
         os.path.join(models_dir, file_name))
        for file_name in sorted(os.listdir(models_dir))
        if file_name.startswith(prefix) and file_name.endswith('.pt')
    ]

    # Pre-initialize CUDA contexts before any threading
    init_cuda_contexts(devices)

    # ------------------------------------------------------------------ #
    # [1] Pre-load DataLoaders in parallel                                   #
    #   - Dataset initialization is done concurrently before GPU evaluation  #
    #   - num_workers: CPU cores distributed evenly across concurrent GPUs   #
    # ------------------------------------------------------------------ #
    num_workers = min(4, max(1, (os.cpu_count() or 4) // n_parallel))
    print(f'Loading {len(tasks)} test dataloaders in parallel (num_workers={num_workers} each)...')

    task_loaders: dict = {}
    loader_lock = threading.Lock()

    def _load_dataloader(name):
        loader = get_test_dataloader(name, batch_size=64, num_workers=num_workers, model_type=model_type)
        with loader_lock:
            task_loaders[name] = loader

    loader_threads = [threading.Thread(target=_load_dataloader, args=(name,), daemon=True)
                      for name, _ in tasks]
    for t in loader_threads: t.start()
    for t in loader_threads: t.join()
    print(f'Dataloaders ready.\n{"="*50}')

    # ------------------------------------------------------------------ #
    # [2] GPU device pool                                                    #
    # ------------------------------------------------------------------ #
    device_queue: Queue = Queue()
    for dev in devices:
        device_queue.put(dev)

    results: dict = {}
    results_lock = threading.Lock()

    def eval_task(task_name, model_path):
        device = device_queue.get()
        try:
            # [3] No model_init_lock: each thread uses a distinct device, so
            #     parallel init is safe (CUDA contexts are pre-initialized)
            model = make_model(task_name, device)
            model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True), strict=False)
            acc = evaluate_model(model, task_loaders[task_name], device)
            with results_lock:
                results[task_name] = acc
            tprint(f'  {task_name:<20} {acc*100:.4f}%')
        except Exception as e:
            tprint(f'  {task_name:<20} ERROR: {e}')
        finally:
            device_queue.put(device)

    # ------------------------------------------------------------------ #
    # [4] Limit concurrent threads to the number of GPUs via ThreadPoolExecutor  #
    #   - Before: all task threads created at once → heavy lock contention       #
    #   - After: at most n_parallel threads run; others wait in the pool queue   #
    # ------------------------------------------------------------------ #
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = [executor.submit(eval_task, name, path) for name, path in tasks]
        for f in as_completed(futures):
            f.result()  # re-raise any exception from the worker

    print(f'\n{"="*50}')
    existing = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
    existing.update({task_name: round(results[task_name] * 100, 4)
                     for task_name, _ in tasks if task_name in results})
    with open(result_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'Results saved to {result_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['vit', 'clip'], default='clip')
    parser.add_argument('--clip_arch', choices=['ViT-B-32', 'ViT-B-16', 'ViT-L-14'], default='ViT-B-32')
    parser.add_argument('--vit_arch',
                        choices=['vit_base_patch32_224', 'vit_base_patch16_224', 'vit_large_patch16_224'],
                        default='vit_base_patch16_224')
    parser.add_argument('--head_type', choices=['zeroshot', 'linear'], default='linear',
                        help='Inference head type for CLIP (ignored for vit): '
                             'zeroshot uses text embeddings, linear uses a trained classification head '
                             '(default: linear)')
    parser.add_argument('--mode', choices=['ft', 'lp', 'lp-ft'], default='lp-ft',
                        help='Training mode of the checkpoints to evaluate: '
                             'ft (full fine-tuning), lp (linear probe), lp-ft (LP then FT) '
                             '(default: lp-ft)')
    args = parser.parse_args()

    evaluate_saved_models(
        model_type=args.model,
        clip_arch=args.clip_arch,
        vit_arch=args.vit_arch,
        head_type=args.head_type,
        mode=args.mode,
    )
