import copy
import torch

from .utils import svd

class TaskVector():
    def __init__(self, pretrained, finetuned, finetuned_heads, task_names):
        self.backbone = {}
        self.head_weights = {}
        self.trained_task_names = task_names
        self.linear_weight_list = []

        for parameter_name in pretrained:
            self.backbone[parameter_name] = finetuned[parameter_name] - pretrained[parameter_name].to(finetuned[parameter_name].device)

            if any(k in parameter_name for k in ['weight', 'proj']) and self.backbone[parameter_name].dim() in [2, 4]:
                if 'embed' not in parameter_name and 'position' not in parameter_name:
                    self.linear_weight_list.append(parameter_name)

        for task_name in task_names:
            self.head_weights[task_name] = finetuned_heads

    def __add__(self, other):
        tv1 = copy.deepcopy(self)
        tv2 = copy.deepcopy(other)

        for parameter_name in tv1.backbone:
            tv1.backbone[parameter_name] += tv2.backbone[parameter_name]

        for task_name in tv2.trained_task_names:
            tv1.head_weights[task_name] = tv2.head_weights[task_name]

        tv1.trained_task_names = list(set(tv1.trained_task_names) | set(tv2.trained_task_names))

        return tv1

    def __mul__(self, scalar):
        tv1 = copy.deepcopy(self)
        for parameter_name in tv1.backbone:
            tv1.backbone[parameter_name] *= scalar
        return tv1

    def __rmul__(self, other):
        return copy.deepcopy(self) * other

    def svd_linear_weight(self):
        return {name: svd(self.backbone[name]) for name in self.linear_weight_list}

    def linear_weight_norm(self):
        sq_norms = torch.stack([self.backbone[name].norm() ** 2 for name in self.linear_weight_list])
        return sq_norms.sum().sqrt().item()
