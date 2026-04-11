import os
import torch
import copy, argparse, threading
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

    def get_split_rank(self, S):
        return (S.cumsum(dim=0) / S.sum() > self.alpha).float().argmax().item()

    def project_linear_weight(self, svd_result, linear_weight, split_rank):
        U, _, V = svd_result

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
        layer_info = {}

        for linear_weight_name in tv.linear_weight_list:
            svd_result = svd_result_dict[linear_weight_name]
            weight = tv.backbone[linear_weight_name]
            min_dim = min(weight.shape)

            split_rank = self.get_split_rank(svd_result[1])
            total_rank += split_rank

            layer_info[linear_weight_name] = {
                'split_rank': split_rank,
                'min_dim':    min_dim,
            }

            projected_linear_weight = self.project_linear_weight(
                svd_result, weight, split_rank
            )
            projected_task_vector.backbone[linear_weight_name] = projected_linear_weight

            total_fip += frobenius_inner_product(
                self.merged_task_vector.backbone[linear_weight_name], projected_linear_weight
            )
            total_fip_w_first += frobenius_inner_product(
                self.first_tv.backbone[linear_weight_name], projected_linear_weight
            )
            total_error += torch.linalg.norm(
                projected_linear_weight - weight, ord='fro'
            )

        n = len(tv.linear_weight_list)
        metrics = {
            'inner_product': total_fip / n,
            'inner_product_with_first': total_fip_w_first / n,
            'approx_error': total_error / n,
            'rank': total_rank / n,
            'layer_info': layer_info,
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


def _run_once(args, tasks, task_vectors, model, gpu_replicas, csv_logger, use_mlflow, model_type, run_label=''):
    """Run one OPCM merge pass. Returns final accuracies dict {task_name: acc}."""
    alpha = args.alpha

    def _evaluate(tasks_to_eval):
        if gpu_replicas is not None:
            return evaluate_parallel(gpu_replicas, tasks_to_eval, model_type=model_type)
        return evaluate_model(model, tasks_to_eval, model_type=model_type)

    n_tasks = len(tasks)
    opcm = OPCM(alpha, task_vectors[0])

    prefix = f'[Run {run_label}] ' if run_label else ''

    # --- Step 1: first task (no merge needed) ---
    print(f'{prefix}[1/{n_tasks}] Evaluating initial task: {tasks[0]}')
    merged_task_vector = opcm.get_merged_task_vector()
    model.load_task_vector(merged_task_vector)
    if gpu_replicas is not None:
        _sync_replicas(model, gpu_replicas)

    raw_accs = _evaluate(tasks[:1])
    accs = {key.split('_', 2)[2]: val for key, val in raw_accs.items()}
    print(f'  {tasks[0]}: {accs[tasks[0]]*100:.2f}%')

    if use_mlflow:
        import mlflow
        mlflow.log_metrics(raw_accs, step=1)
    if csv_logger:
        csv_logger.log_accuracies(step=1, merged_tasks=tasks[:1], accuracies=accs)

    final_accs = accs

    # --- Steps 2..N: merge remaining tasks ---
    for tv in task_vectors[1:]:
        added_task = tv.trained_task_names[0]

        print(f'\n{prefix}[{tasks.index(added_task) + 1}/{n_tasks}] Merging: {added_task}')
        print(f'  Projecting task vector...')
        metrics = opcm.merge_task_vector(tv)
        print(f'  inner_product={metrics["inner_product"]:.4f}  '
              f'approx_error={metrics["approx_error"]:.4f}  '
              f'avg_rank={metrics["rank"]:.1f}')
        print(f'  {"Layer":<50} {"split_rank":>10} {"min_dim":>8} {"remaining":>10}')
        for layer_name, info in metrics['layer_info'].items():
            remaining = info['min_dim'] - info['split_rank']
            print(f'  {layer_name:<50} {info["split_rank"]:>10} {info["min_dim"]:>8} {remaining:>10}')

        merged_task_vector = opcm.get_merged_task_vector()
        merged_task_number = opcm.get_merged_task_number()

        model.load_task_vector(merged_task_vector)
        if gpu_replicas is not None:
            _sync_replicas(model, gpu_replicas)

        print(f'  Evaluating {merged_task_number} tasks...')
        raw_accs = _evaluate(tasks[:merged_task_number])
        accs = {key.split('_', 2)[2]: val for key, val in raw_accs.items()}
        for t, acc in accs.items():
            print(f'    {t}: {acc*100:.2f}%')

        if use_mlflow:
            mlflow.log_metrics(raw_accs, step=merged_task_number)
            mlflow.log_metric('inner product', metrics['inner_product'], step=merged_task_number)
            mlflow.log_metric('inner product with first', metrics['inner_product_with_first'], step=merged_task_number)
            mlflow.log_metric('approx error', metrics['approx_error'], step=merged_task_number)
            mlflow.log_metric('rank', metrics['rank'], step=merged_task_number)
            for layer_name, info in metrics['layer_info'].items():
                mlflow.log_metric(f'split_rank_{layer_name}', info['split_rank'], step=merged_task_number)

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
                split_ranks={name: info['split_rank'] for name, info in metrics['layer_info'].items()},
            )

        final_accs = accs

    return final_accs


def _save_average_results(all_run_accs, parent_save_dir):
    """Compute mean/std of final accuracies across runs and save to average/ subdir."""
    import csv as _csv
    import json as _json

    avg_dir = os.path.join(parent_save_dir, 'average')
    os.makedirs(avg_dir, exist_ok=True)

    # Collect per-task values across runs
    task_values = {}
    for run_accs in all_run_accs:
        for task, acc in run_accs.items():
            task_values.setdefault(task, []).append(acc)

    tasks_sorted = sorted(task_values.keys())

    # Compute mean and std (in % scale)
    summary = {}
    for task in tasks_sorted:
        vals = [v * 100 for v in task_values[task]]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        summary[task] = {'mean': round(mean, 4), 'std': round(std, 4), 'runs': vals}

    overall_means = [summary[t]['mean'] for t in tasks_sorted]
    overall_mean = sum(overall_means) / len(overall_means)

    summary['_overall'] = {
        'mean': round(overall_mean, 4),
        'num_runs': len(all_run_accs),
    }

    with open(os.path.join(avg_dir, 'summary.json'), 'w') as f:
        _json.dump(summary, f, indent=2)

    # Save average_accuracy.csv: one row per run + mean/std rows
    headers = ['run'] + tasks_sorted + ['avg_all_tasks']
    rows = []
    for i, run_accs in enumerate(all_run_accs):
        row_vals = [run_accs.get(t, '') for t in tasks_sorted]
        row_avg = sum(v for v in row_vals if isinstance(v, float)) / sum(1 for v in row_vals if isinstance(v, float))
        rows.append([f'run_{i}'] + [round(v * 100, 4) if isinstance(v, float) else v for v in row_vals] + [round(row_avg * 100, 4)])

    mean_row = ['mean'] + [summary[t]['mean'] for t in tasks_sorted] + [round(overall_mean, 6)]
    std_row = ['std'] + [summary[t]['std'] for t in tasks_sorted] + ['']
    rows += [mean_row, std_row]

    with open(os.path.join(avg_dir, 'average_accuracy.csv'), 'w', newline='') as f:
        w = _csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    print(f'\nAverage results saved to: {avg_dir}')
    print(f'Overall mean accuracy ({len(all_run_accs)} runs): {overall_mean:.2f}%')
    for task in tasks_sorted:
        print(f'  {task}: {summary[task]["mean"]:.2f}% ± {summary[task]["std"]:.2f}%')


def main(args):
    alpha = args.alpha
    monitor = args.monitor
    model_type = args.model
    clip_arch = args.clip_arch
    vit_arch = args.vit_arch
    head_type = args.head_type if model_type == 'clip' else 'linear'
    mode      = args.mode
    num_runs  = args.num_runs

    use_mlflow = monitor in ('mlflow', 'both')
    use_csv = monitor in ('csv', 'both')

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
    print(f'Model: {model_type} ({arch_str}), head_type: {head_type}, mode: {mode}')

    if model_type == 'clip':
        if head_type == 'linear':
            model = MultiTaskCLIPLinear(clip_arch=clip_arch)
        else:
            model = MultiTaskCLIP(clip_arch=clip_arch)
    else:
        model = MultiTaskViT(vit_arch=vit_arch)
    model.to(device)

    if args.num_tasks == '8':
        base_task_list = TASKS_8
    elif args.num_tasks == '14':
        base_task_list = TASKS_14
    else:
        base_task_list = None  # all tasks from num_classes_per_task.json

    # Resolve full task list for non-shuffle single run
    if base_task_list is None:
        import json as _json
        with open(os.path.join('dataset', 'num_classes_per_task.json')) as _f:
            base_task_list = list(_json.load(_f).keys())

    if args.shuffle and num_runs == 1:
        import random
        random.shuffle(base_task_list)
        print(f'Task order (shuffled): {base_task_list}')

    if num_runs > 1 and not args.shuffle:
        print(f'Note: num_runs={num_runs} with shuffle=False will repeat the same task order.')

    task_vectors = load_task_vectors(device, model_type=model_type, clip_arch=clip_arch, vit_arch=vit_arch,
                                     head_type=head_type, task_list=base_task_list, mode=mode)

    # --- Multi-GPU replicas ---
    gpu_replicas = None
    if n_gpus > 1:
        print(f'Parallel evaluation enabled across {n_gpus} GPUs.')
        gpu_replicas = _make_gpu_replicas(model, n_gpus)

    # --- CSV logger setup ---
    result_txt = os.path.join(
        'results', 'single_task_accuracy', model_type, mode,
        f'result_{model_type}_{head_type}_{arch_str}.json',
    )

    single_task_accs = None
    if use_csv:
        from src.csv_logger import CSVLogger, load_single_task_accs
        single_task_accs = load_single_task_accs(result_txt)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    shuffle_suffix = '_shuffled' if args.shuffle else ''
    runs_suffix = f'_runs{num_runs}' if num_runs > 1 else ''
    run_name = f'{timestamp}{shuffle_suffix}{runs_suffix}'
    mode_dir = mode if head_type == 'linear' else 'ft'
    parent_save_dir = os.path.join(
        'results', 'opcm',
        model_type, arch_str, head_type, mode_dir,
        f'alpha{alpha}',
        run_name,
    )

    all_run_accs = []

    for run_idx in range(num_runs):
        import random as _random

        # Shuffle task order per run when num_runs > 1
        if num_runs > 1 and args.shuffle:
            shuffled_tvs = task_vectors.copy()
            _random.shuffle(shuffled_tvs)
        else:
            shuffled_tvs = task_vectors

        run_tasks = [tv.trained_task_names[0] for tv in shuffled_tvs]

        if num_runs > 1:
            run_save_dir = os.path.join(parent_save_dir, f'run_{run_idx}')
            run_label = f'{run_idx + 1}/{num_runs}'
            print(f'\n{"=" * 60}')
            print(f'Run {run_idx + 1}/{num_runs}  |  Task order: {run_tasks}')
            print('=' * 60)
        else:
            run_save_dir = parent_save_dir
            run_label = ''

        csv_logger = None
        if use_csv:
            csv_logger = CSVLogger(run_save_dir, run_tasks, single_task_accs, args)
            print(f'CSV results will be saved to: {run_save_dir}')

        final_accs = _run_once(
            args, run_tasks, shuffled_tvs, model, gpu_replicas,
            csv_logger, use_mlflow, model_type, run_label=run_label,
        )
        all_run_accs.append(final_accs)

    if num_runs > 1 and use_csv:
        _save_average_results(all_run_accs, parent_save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument(
        '--num_tasks',
        choices=['8', '14', 'all'],
        default='8',
        help='Number of tasks to use: 8 (SUN397/Cars/RESISC45/EuroSAT/SVHN/GTSRB/MNIST/DTD), '
             '14 (8 + Flowers102/PCAM/OxfordIIITPet/STL10/CIFAR100/FashionMNIST), '
             'all (all tasks with checkpoints available, as listed in dataset/num_classes_per_task.json) ',
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
        default='linear',
        help='Inference head type for CLIP (ignored for vit): '
             'zeroshot uses text embeddings, linear uses a trained classification head '
             '(default: linear)',
    )
    parser.add_argument(
        '--mode',
        choices=['ft', 'lp-ft'],
        default='lp-ft',
        help='Checkpoint mode to load task vectors from: ft or lp-ft (default: lp-ft)',
    )
    parser.add_argument(
        '--num_runs',
        type=int,
        default=1,
        help='Number of times to repeat the experiment (default: 1). '
             'Use with --shuffle to average over different task orders.',
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
