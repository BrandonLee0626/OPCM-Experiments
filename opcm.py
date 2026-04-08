import os
import torch
import copy, math, argparse, threading
from datetime import datetime
from queue import Queue

from src.model import MultiTaskViT, MultiTaskCLIP, MultiTaskCLIPLinear
from src.task_vector import TaskVector
from src.utils import load_task_vectors, evaluate_model, evaluate_task, frobenius_inner_product

TASKS_8 = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
TASKS_14 = TASKS_8 + ['Flowers102', 'PCAM', 'OxfordIIITPet', 'STL10', 'CIFAR100', 'FashionMNIST']


# ---------------------------------------------------------------------------
# Multi-GPU parallel evaluation
# ---------------------------------------------------------------------------

def _make_gpu_replicas(source_model, n_gpus):
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

def _sync_replicas(source_model, replicas):
    """Copy updated backbone (and heads if present) from source_model to every replica."""
    backbone_cpu = {k: v.cpu() for k, v in source_model.backbone.state_dict().items()}
    heads_cpu = None
    if hasattr(source_model, 'heads'):
        heads_cpu = {
            name: {k: v.cpu() for k, v in head.state_dict().items()}
            for name, head in source_model.heads.items()
        }
    for replica, dev in replicas:
        replica.backbone.load_state_dict({k: v.to(dev) for k, v in backbone_cpu.items()})
        if heads_cpu is not None:
            for name, state in heads_cpu.items():
                replica.heads[name].load_state_dict({k: v.to(dev) for k, v in state.items()})

def evaluate_parallel(replicas, tasks, model_type='vit'):
    """
    Evaluate tasks in parallel across GPUs using a GPU pool queue.
    Each GPU processes one task at a time; tasks are dynamically assigned.
    Returns dict with same key format as evaluate_model.
    """
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

    threads = [
        threading.Thread(target=_worker, args=(i, task), daemon=True)
        for i, task in enumerate(tasks)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results


class OPCM:
    def __init__(self, alpha, task_vector: TaskVector):
        self.alpha = alpha
        self.merged_task_vector = task_vector
        self.merged_task_number = 1
        self.avg_task_vector_norm = task_vector.linear_weight_norm()
        self.scaling_factor = 1
        self.previous_lambda_t = 1.0
        self.projected_task_vector_sum = task_vector
        self.first_tv = task_vector

        self.rank_count = {
            linear_weight_name: 0
            for linear_weight_name in task_vector.linear_weight_list
        }

    def get_split_rank(self, S):
        return (S.cumsum(dim=0) / S.sum() > self.alpha).float().argmax().item()

    def project_linear_weight(self, svd_result, linear_weight, split_rank):
        U, S, V = svd_result

        coeff_mat = U.T @ linear_weight @ V
        coeff_mat.fill_diagonal_(0)
        coeff_mat[:split_rank, :split_rank] = 0

        projected_linear_weight = U @ coeff_mat @ V.T

        return projected_linear_weight

    def project_task_vector(self, tv: TaskVector):
        """Project tv onto the null space of merged_task_vector.

        Returns:
            projected_task_vector: TaskVector
            metrics: dict with inner_product, inner_product_with_first, approx_error, rank
        """
        projected_task_vector = copy.deepcopy(tv)

        svd_result_dict = self.merged_task_vector.svd_linear_weight()

        total_fip = 0.0
        total_fip_w_first = 0.0
        total_error = 0.0
        total_rank = 0

        for linear_weight_name in tv.linear_weight_list:
            svd_result = svd_result_dict[linear_weight_name]

            split_rank = self.get_split_rank(svd_result[1])
            self.rank_count[linear_weight_name] += split_rank
            total_rank += split_rank

            projected_linear_weight = self.project_linear_weight(
                svd_result, tv.backbone[linear_weight_name], split_rank
            )
            projected_task_vector.backbone[linear_weight_name] = projected_linear_weight

            total_fip += frobenius_inner_product(
                self.merged_task_vector.backbone[linear_weight_name], projected_linear_weight
            )
            total_fip_w_first += frobenius_inner_product(
                self.first_tv.backbone[linear_weight_name], projected_linear_weight
            )
            total_error += torch.linalg.norm(
                projected_linear_weight - tv.backbone[linear_weight_name], ord='fro'
            )

        n = len(tv.linear_weight_list)
        metrics = {
            'inner_product': total_fip / n,
            'inner_product_with_first': total_fip_w_first / n,
            'approx_error': total_error / n,
            'rank': total_rank / n,
        }

        return projected_task_vector, metrics

    def merge_task_vector(self, tv: TaskVector):
        projected_task_vector, metrics = self.project_task_vector(tv)

        numerator_tv = self.previous_lambda_t * self.merged_task_vector + projected_task_vector

        tv_norm = tv.linear_weight_norm()
        merged_task_number = self.get_merged_task_number()
        self.avg_task_vector_norm = (
            (merged_task_number * self.avg_task_vector_norm + tv_norm) / (merged_task_number + 1)
        )

        new_norm = numerator_tv.linear_weight_norm()
        self.lambda_t = new_norm / self.avg_task_vector_norm

        self.merged_task_vector = (1 / self.lambda_t) * numerator_tv
        self.previous_lambda_t = self.lambda_t
        self.merged_task_number += 1

        return metrics

    def get_merged_task_vector(self):
        return self.merged_task_vector

    def get_merged_task_number(self):
        return self.merged_task_number


def main(args):
    alpha = args.alpha
    monitor = args.monitor
    model_type = args.model
    clip_arch = args.clip_arch
    vit_arch = args.vit_arch
    head_type = args.head_type if model_type == 'clip' else 'linear'

    use_mlflow = monitor in ('mlflow', 'both')
    use_csv = monitor in ('csv', 'both')

    if use_mlflow:
        import mlflow

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()

    if n_gpus == 0:
        print('Device: CPU')
    elif n_gpus == 1:
        print(f'Device: {torch.cuda.get_device_name(0)} (1 GPU)')
    else:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        print(f'Device: {n_gpus} GPUs available')
        for i, name in enumerate(gpu_names):
            print(f'  cuda:{i}  {name}')

    arch_str = clip_arch if model_type == 'clip' else vit_arch
    print(f'Model: {model_type} ({arch_str}), head_type: {head_type}')

    if model_type == 'clip':
        if head_type == 'linear':
            model = MultiTaskCLIPLinear(clip_arch=clip_arch)
        else:
            model = MultiTaskCLIP(clip_arch=clip_arch)
    else:
        model = MultiTaskViT(vit_arch=vit_arch)
    model.to(device)

    if args.num_tasks == '8':
        task_list = TASKS_8
    elif args.num_tasks == '14':
        task_list = TASKS_14
    else:
        task_list = None  # all tasks from num_classes_per_task.json

    if args.shuffle:
        import random
        if task_list is None:
            import json as _json
            with open(os.path.join('dataset', 'num_classes_per_task.json')) as _f:
                task_list = list(_json.load(_f).keys())
        random.shuffle(task_list)
        print(f'Task order (shuffled): {task_list}')

    task_vectors = load_task_vectors(device, model_type=model_type, clip_arch=clip_arch, vit_arch=vit_arch,
                                     head_type=head_type, task_list=task_list)
    tasks = [tv.trained_task_names[0] for tv in task_vectors]

    # --- Multi-GPU replicas ---
    gpu_replicas = None
    if n_gpus > 1:
        print(f'Parallel evaluation enabled across {n_gpus} GPUs.')
        gpu_replicas = _make_gpu_replicas(model, n_gpus)

    def _evaluate(tasks_to_eval):
        if gpu_replicas is not None:
            return evaluate_parallel(gpu_replicas, tasks_to_eval, model_type=model_type)
        return evaluate_model(model, tasks_to_eval, model_type=model_type)

    # --- CSV logger setup ---
    if model_type == 'clip':
        result_txt = os.path.join('results', 'single_task_accuracy', 'clip',
                                  f'result_clip_{head_type}_{arch_str}.txt')
    else:
        result_txt = os.path.join('results', 'single_task_accuracy', 'vit',
                                  f'result_vit_{arch_str}.txt')
    csv_logger = None
    if use_csv:
        from src.csv_logger import CSVLogger, load_single_task_accs
        single_task_accs = load_single_task_accs(result_txt)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        shuffle_suffix = '_shuffled' if args.shuffle else ''
        save_dir = f'results/{timestamp}_{model_type}_{head_type}_{arch_str}_tasks{args.num_tasks}_alpha{alpha}{shuffle_suffix}'
        csv_logger = CSVLogger(save_dir, tasks, single_task_accs, args)
        print(f'CSV results will be saved to: {save_dir}')

    n_tasks = len(tasks)
    opcm = OPCM(alpha, task_vectors[0])

    # --- Step 1: first task (no merge needed) ---
    print(f'[1/{n_tasks}] Evaluating initial task: {tasks[0]}')
    merged_task_vector = opcm.get_merged_task_vector()
    model.load_task_vector(merged_task_vector)
    if gpu_replicas is not None:
        _sync_replicas(model, gpu_replicas)

    raw_accs = _evaluate(tasks[:1])
    # raw_accs keys are like "task_0_SUN397"; map to plain task names
    accs = {key.split('_', 2)[2]: val for key, val in raw_accs.items()}
    print(f'  {tasks[0]}: {accs[tasks[0]]:.4f}')

    if use_mlflow:
        mlflow.log_metrics(raw_accs, step=1)
    if csv_logger:
        csv_logger.log_accuracies(step=1, merged_tasks=tasks[:1], accuracies=accs)

    # --- Steps 2..N: merge remaining tasks ---
    for tv in task_vectors[1:]:
        added_task = tv.trained_task_names[0]

        print(f'\n[{tasks.index(added_task) + 1}/{n_tasks}] Merging: {added_task}')
        print(f'  Projecting task vector...')
        metrics = opcm.merge_task_vector(tv)
        print(f'  inner_product={metrics["inner_product"]:.4f}  '
              f'approx_error={metrics["approx_error"]:.4f}  '
              f'avg_rank={metrics["rank"]:.1f}')

        merged_task_vector = opcm.get_merged_task_vector()
        merged_task_number = opcm.get_merged_task_number()

        model.load_task_vector(merged_task_vector)
        if gpu_replicas is not None:
            _sync_replicas(model, gpu_replicas)

        print(f'  Evaluating {merged_task_number} tasks...')
        raw_accs = _evaluate(tasks[:merged_task_number])
        accs = {key.split('_', 2)[2]: val for key, val in raw_accs.items()}
        for t, acc in accs.items():
            print(f'    {t}: {acc:.4f}')

        if use_mlflow:
            mlflow.log_metrics(raw_accs, step=merged_task_number)
            mlflow.log_metric('inner product', metrics['inner_product'], step=merged_task_number)
            mlflow.log_metric('inner product with first', metrics['inner_product_with_first'], step=merged_task_number)
            mlflow.log_metric('approx error', metrics['approx_error'], step=merged_task_number)
            mlflow.log_metric('rank', metrics['rank'], step=merged_task_number)
            for layer_name, cumulative_rank in opcm.rank_count.items():
                mlflow.log_metric(f'added_split_rank_{layer_name}', cumulative_rank, step=merged_task_number)

        if csv_logger:
            csv_logger.log_accuracies(
                step=merged_task_number,
                merged_tasks=tasks[:merged_task_number],
                accuracies=accs,
            )
            csv_logger.log_projection_metrics(
                step=merged_task_number,
                added_task=added_task,
                metrics=metrics,
            )
            csv_logger.log_layer_ranks(
                step=merged_task_number,
                added_task=added_task,
                rank_count_dict=opcm.rank_count,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument(
        '--num_tasks',
        choices=['8', '14', 'all'],
        default='all',
        help='Number of tasks to use: 8 (SUN397/Cars/RESISC45/EuroSAT/SVHN/GTSRB/MNIST/DTD), '
             '14 (8 + Flowers102/PCAM/OxfordIIITPet/STL10/CIFAR100/FashionMNIST), '
             'all (default)',
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=False,
        help='Randomly shuffle task merge order',
    )
    parser.add_argument(
        '--monitor',
        choices=['mlflow', 'csv', 'both'],
        default='csv',
        help='Logging backend (default: csv)',
    )
    parser.add_argument(
        '--model',
        choices=['vit', 'clip'],
        default='clip',
        help='Backbone model type (default: clip)',
    )
    parser.add_argument(
        '--clip_arch',
        choices=['ViT-B-32', 'ViT-B-16', 'ViT-L-14'],
        default='ViT-B-32',
        help='CLIP architecture (only used when --model clip)',
    )
    parser.add_argument(
        '--vit_arch',
        choices=['vit_base_patch32_224', 'vit_base_patch16_224', 'vit_large_patch16_224'],
        default='vit_base_patch16_224',
        help='ViT architecture (only used when --model vit)',
    )
    parser.add_argument(
        '--head_type',
        choices=['zeroshot', 'linear'],
        default='zeroshot',
        help='Inference head type for CLIP (ignored for vit): '
             'zeroshot uses text embeddings, linear uses a trained classification head '
             '(default: zeroshot)',
    )

    args = parser.parse_args()

    if args.monitor in ('mlflow', 'both'):
        import mlflow
        mlflow.set_experiment('OPCM_experiments')
        with mlflow.start_run():
            mlflow.log_param('alpha', args.alpha)
            mlflow.log_param('model', args.model)
            if args.model == 'clip':
                mlflow.log_param('clip_arch', args.clip_arch)
                mlflow.log_param('head_type', args.head_type)
            main(args)
    else:
        main(args)
