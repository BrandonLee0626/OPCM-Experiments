import csv
import json
import os
from datetime import datetime


class CSVLogger:
    def __init__(self, save_dir, tasks, single_task_accs, args):
        """
        tasks: list of task names in merge order
        single_task_accs: dict {task_name: acc} loaded from result.txt
        args: argparse.Namespace from main
        """
        self.save_dir = save_dir
        self.tasks = tasks
        self.single_task_accs = single_task_accs
        self.alpha = args.alpha

        # task -> accuracy when first merged (for forgetting computation)
        self.first_merge_accs = {}

        os.makedirs(save_dir, exist_ok=True)

        arch_str = args.clip_arch if args.model == 'clip' else args.vit_arch
        head_type = args.head_type if args.model == 'clip' else 'linear'
        config = {
            'timestamp': datetime.now().isoformat(),
            'model': args.model,
            'arch': arch_str,
            'head_type': head_type,
            'alpha': args.alpha,
            'num_tasks': len(tasks),
            'shuffle': False if getattr(args, 'task_order_file', None) else args.shuffle,
            'task_order': tasks,
            'single_task_accs': single_task_accs,
        }
        if head_type == 'linear':
            config['mode'] = args.mode
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        task_headers = ['step', 'merged_tasks'] + tasks
        for filename in ['accuracy.csv', 'drop_vs_single.csv', 'forgetting.csv']:
            self._write_row(filename, task_headers, mode='w')

        self._write_row(
            'projection_metrics.csv',
            ['step', 'added_task', 'inner_product', 'inner_product_with_first', 'approx_error', 'avg_rank'],
            mode='w',
        )

        self._layer_ranks_initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_accuracies(self, step, merged_tasks, accuracies):
        """
        merged_tasks: list of task names merged so far (in order)
        accuracies: dict {task_name: acc}
        """
        for task in merged_tasks:
            if task not in self.first_merge_accs:
                acc = accuracies.get(task)
                self.first_merge_accs[task] = acc * 100 if acc is not None else None

        label = '+'.join(merged_tasks)

        # --- accuracy.csv ---
        self._write_row('accuracy.csv', [step, label] + [
            round(accuracies[t] * 100, 4) if t in accuracies else '' for t in self.tasks
        ])

        # --- drop_vs_single.csv ---
        self._write_row('drop_vs_single.csv', [step, label] + [
            round(self.single_task_accs[t] - accuracies[t] * 100, 4)
            if t in accuracies and t in self.single_task_accs else ''
            for t in self.tasks
        ])

        # --- forgetting.csv (drop from first merge) ---
        self._write_row('forgetting.csv', [step, label] + [
            round(self.first_merge_accs[t] - accuracies[t] * 100, 4)
            if t in accuracies and t in self.first_merge_accs and self.first_merge_accs[t] is not None else ''
            for t in self.tasks
        ])

    def log_projection_metrics(self, step, added_task, metrics):
        """
        metrics: dict with keys inner_product, inner_product_with_first, approx_error, rank
        """
        def to_float(v):
            return float(v) if hasattr(v, '__float__') else v

        self._write_row('projection_metrics.csv', [
            step,
            added_task,
            round(to_float(metrics.get('inner_product', 0)), 6),
            round(to_float(metrics.get('inner_product_with_first', 0)), 6),
            round(to_float(metrics.get('approx_error', 0)), 6),
            round(to_float(metrics.get('rank', 0)), 6),
        ])

    def log_layer_ranks(self, step, added_task, split_ranks):
        """
        split_ranks: {layer_name: split_rank}
        """
        layer_names = list(split_ranks.keys())

        if not self._layer_ranks_initialized:
            self._write_row(
                'layer_ranks.csv',
                ['step', 'added_task'] + layer_names,
                mode='w',
            )
            self._layer_ranks_initialized = True

        self._write_row('layer_ranks.csv', [step, added_task] + list(split_ranks.values()))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_row(self, filename, row, mode='a'):
        path = os.path.join(self.save_dir, filename)
        with open(path, mode, newline='') as f:
            csv.writer(f).writerow(row)


def load_single_task_accs(result_json_path):
    """Load result JSON into {task_name: accuracy} dict."""
    with open(result_json_path) as f:
        return json.load(f)
