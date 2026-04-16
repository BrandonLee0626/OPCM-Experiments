import copy
import threading
from queue import Queue

import torch

from src.utils import evaluate_task


def get_devices() -> list:
    """Detect available GPUs (or CPU), print device info, and return a list of devices."""
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print('Device: CPU')
        return [torch.device('cpu')]
    elif n_gpus == 1:
        print(f'Device: {torch.cuda.get_device_name(0)} (1 GPU)')
        return [torch.device('cuda:0')]
    else:
        print(f'Device: {n_gpus} GPUs available')
        for i in range(n_gpus):
            print(f'  cuda:{i}  {torch.cuda.get_device_name(i)}')
        return [torch.device(f'cuda:{i}') for i in range(n_gpus)]


def init_cuda_contexts(devices: list) -> None:
    """Pre-initialize CUDA contexts on all devices to prevent race conditions in multi-threaded code."""
    for dev in devices:
        if dev.type == 'cuda':
            torch.zeros(1, device=dev)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_gpu_replicas(source_model, n_gpus):
    """Create one frozen model replica per GPU."""
    replicas = []
    for gpu_id in range(n_gpus):
        dev = torch.device(f'cuda:{gpu_id}')
        replica = copy.deepcopy(source_model).to(dev)
        # plain-dict tensors (e.g. CLIP zero-shot weights) are not moved by .to()
        if hasattr(replica, 'zeroshot_weights'):
            replica.zeroshot_weights = {
                k: v.to(dev) for k, v in replica.zeroshot_weights.items()
            }
        replica.eval()
        replicas.append((replica, dev))
    return replicas


def sync_replicas(source_model, replicas):
    """Copy updated backbone (and heads if present) from source_model to every replica."""
    backbone_cpu = {k: v.cpu() for k, v in source_model.backbone.state_dict().items()}
    heads_cpu = None
    if hasattr(source_model, 'heads'):
        heads_cpu = {
            name: {k: v.cpu() for k, v in head.state_dict().items()}
            for name, head in source_model.heads.items()
        }

    def _copy_to(replica, dev):
        replica.backbone.load_state_dict({k: v.to(dev) for k, v in backbone_cpu.items()})
        if heads_cpu is not None:
            for name, state in heads_cpu.items():
                replica.heads[name].load_state_dict({k: v.to(dev) for k, v in state.items()})

    threads = [
        threading.Thread(target=_copy_to, args=(replica, dev), daemon=True)
        for replica, dev in replicas
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def evaluate_parallel(replicas, tasks, model_type='vit'):
    """
    Evaluate tasks in parallel across GPUs using a GPU pool queue.
    Each GPU processes one task at a time; tasks are dynamically assigned.
    Returns dict with same key format as evaluate_model.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    gpu_queue = Queue()
    for replica, dev in replicas:
        gpu_queue.put((replica, dev))

    results = {}
    results_lock = threading.Lock()

    def _worker(task_idx, task):
        replica, dev = gpu_queue.get()
        try:
            acc = evaluate_task(replica, task, dev, model_type=model_type)
            with results_lock:
                results[f'task_{task_idx}_{task}'] = acc
        finally:
            gpu_queue.put((replica, dev))

    # Limit concurrent threads to the number of GPUs to avoid unnecessary contention
    with ThreadPoolExecutor(max_workers=len(replicas)) as executor:
        futures = [executor.submit(_worker, i, task) for i, task in enumerate(tasks)]
        for f in as_completed(futures):
            f.result()

    return results
