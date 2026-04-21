import os
import sys
import json
import threading
from queue import Queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse

from src.model import SingleTaskViT, SingleTaskCLIP, SingleTaskCLIPLinear
from src.parallel import get_devices
from dataset.dataloader import get_test_dataloader


def _load_and_eval(task_name, model_path, device, model_type, arch, head_type):
    if model_type == 'clip':
        if head_type == 'linear':
            model = SingleTaskCLIPLinear(task_name=task_name, clip_arch=arch).to(device)
        else:
            model = SingleTaskCLIP(task_name=task_name, clip_arch=arch).to(device)
    else:
        model = SingleTaskViT(task_name=task_name, vit_arch=arch).to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.eval()

    loader = get_test_dataloader(task_name, batch_size=64, num_workers=4, model_type=model_type)
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0


def _worker_loop(device, task_q, result_q, model_type, arch, head_type):
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    while True:
        item = task_q.get()
        if item is None:
            return
        task_name, model_path = item
        try:
            acc = _load_and_eval(task_name, model_path, device, model_type, arch, head_type)
            result_q.put((task_name, acc))
        except Exception as e:
            result_q.put((task_name, e))


def evaluate_saved_models(model_type='vit', clip_arch='ViT-B-32', vit_arch='vit_base_patch16_224',
                          head_type='zeroshot', mode='ft'):
    devices = get_devices()

    if model_type == 'clip':
        arch = clip_arch
        if head_type == 'zeroshot' and mode != 'ft':
            print(f'[Warning] --mode={mode} is not applicable to head_type=zeroshot; using ft.')
            mode = 'ft'
        models_dir = os.path.join('models', f'clip_{head_type}', arch, mode)
        prefix = f'clip_{arch}_'
    else:
        arch = vit_arch
        head_type = 'linear'
        models_dir = os.path.join('models', 'vit', arch, mode)
        prefix = f'{arch}_'

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

    task_q: Queue = Queue()
    result_q: Queue = Queue()

    # Create one persistent worker thread per GPU (before any CUDA work)
    for dev in devices:
        t = threading.Thread(
            target=_worker_loop,
            args=(dev, task_q, result_q, model_type, arch, head_type),
            daemon=True,
        )
        t.start()

    for item in tasks:
        task_q.put(item)

    results: dict = {}
    for _ in tasks:
        task_name, acc = result_q.get()
        if isinstance(acc, Exception):
            print(f'  {task_name:<20} ERROR: {acc}')
        else:
            results[task_name] = acc
            print(f'  {task_name:<20} {acc*100:.4f}%')

    # Signal workers to stop
    for _ in devices:
        task_q.put(None)

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
