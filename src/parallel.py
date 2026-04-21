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


def _make_replica(source_model, gpu_id):
    dev = torch.device(f'cuda:{gpu_id}')
    replica = copy.deepcopy(source_model).to(dev)
    if hasattr(replica, 'zeroshot_weights'):
        replica.zeroshot_weights = {
            k: v.to(dev) for k, v in replica.zeroshot_weights.items()
        }
    replica.eval()
    return replica, dev


class EvalPool:
    """
    Persistent per-GPU worker threads for parallel evaluation.

    Threads are created once at construction time (before any CUDA work accumulates),
    then reused across all evaluate() calls. This avoids the CUDA context hang that
    occurs when new threads are spawned repeatedly after CUDA is initialized.
    """

    def __init__(self, source_model, n_gpus, model_type='vit'):
        self.model_type = model_type
        self.replicas = [_make_replica(source_model, i) for i in range(n_gpus)]

        self._task_q: Queue = Queue()
        self._result_q: Queue = Queue()

        for replica, dev in self.replicas:
            t = threading.Thread(target=self._worker_loop, args=(replica, dev), daemon=True)
            t.start()

    def _worker_loop(self, replica, dev):
        if dev.type == 'cuda':
            torch.cuda.set_device(dev)
        while True:
            item = self._task_q.get()
            if item is None:
                return
            idx, task = item
            try:
                acc = evaluate_task(replica, task, dev, model_type=self.model_type)
            except Exception as e:
                acc = e
            self._result_q.put((idx, task, acc))

    def sync(self, source_model):
        """Copy updated weights from source_model to every replica (sequential, main thread)."""
        backbone_state = source_model.backbone.state_dict()
        heads_state = None
        if hasattr(source_model, 'heads'):
            heads_state = {
                name: head.state_dict() for name, head in source_model.heads.items()
            }

        for replica, dev in self.replicas:
            replica.backbone.load_state_dict(
                {k: v.to(dev) for k, v in backbone_state.items()})
            if heads_state is not None:
                for name, state in heads_state.items():
                    replica.heads[name].load_state_dict(
                        {k: v.to(dev) for k, v in state.items()})

    def evaluate(self, tasks) -> dict:
        """Evaluate tasks across GPUs in parallel. Returns {task_N_name: accuracy}."""
        for i, task in enumerate(tasks):
            self._task_q.put((i, task))

        results = {}
        for _ in tasks:
            idx, task, acc = self._result_q.get()
            if isinstance(acc, Exception):
                raise acc
            results[f'task_{idx}_{task}'] = acc
        return results
