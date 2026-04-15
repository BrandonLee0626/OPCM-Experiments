import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from queue import Queue

from src.model import SingleTaskViT, SingleTaskCLIP, SingleTaskCLIPLinear
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
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0


def evaluate_saved_models(model_type='vit', clip_arch='ViT-B-32', vit_arch='vit_base_patch16_224',
                          head_type='zeroshot', mode='ft'):
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        devices = [torch.device('cpu')]
        print('Device: CPU')
    elif n_gpus == 1:
        devices = [torch.device('cuda:0')]
        print(f'Device: {torch.cuda.get_device_name(0)} (1 GPU)')
    else:
        devices = [torch.device(f'cuda:{i}') for i in range(n_gpus)]
        print(f'Device: {n_gpus} GPUs')
        for i in range(n_gpus):
            print(f'  cuda:{i}  {torch.cuda.get_device_name(i)}')

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

    head_tag = head_type
    print(f'Model: {model_type} ({arch}), head_type: {head_tag}, mode: {mode}')
    print(f'Models dir: {models_dir}\n{"="*50}')

    tasks = [
        (file_name.removeprefix(prefix).removesuffix('.pt'),
         os.path.join(models_dir, file_name))
        for file_name in sorted(os.listdir(models_dir))
        if file_name.startswith(prefix) and file_name.endswith('.pt')
    ]

    gpu_queue = Queue()
    for dev in devices:
        gpu_queue.put(dev)

    # Pre-initialize CUDA contexts to avoid race conditions
    for dev in devices:
        if dev.type == 'cuda':
            torch.zeros(1, device=dev)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    model_init_lock = threading.Lock()
    results = {}
    results_lock = threading.Lock()

    def eval_task(task_name, model_path):
        device = gpu_queue.get()
        try:
            with model_init_lock:
                model = make_model(task_name, device)
                model.load_state_dict(
                    torch.load(model_path, map_location=device, weights_only=True), strict=False)
            test_loader = get_test_dataloader(task_name, batch_size=64, num_workers=0, model_type=model_type)
            acc = evaluate_model(model, test_loader, device)
            with results_lock:
                results[task_name] = acc
            tprint(f'  {task_name:<20} {acc*100:.4f}%')
        except Exception as e:
            tprint(f'  {task_name:<20} ERROR: {e}')
        finally:
            gpu_queue.put(device)

    threads = [threading.Thread(target=eval_task, args=(name, path), daemon=True)
               for name, path in tasks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f'\n{"="*50}')
    import json
    existing = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
    existing.update({task_name: round(results[task_name] * 100, 4) for task_name, _ in tasks if task_name in results})
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
