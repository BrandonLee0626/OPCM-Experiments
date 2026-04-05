import torch
import torch.nn as nn
import timm
import copy

from .task_vector import TaskVector
from .utils import num_classes_per_task

class SingleTaskViT(nn.Module):
    def __init__(self, task_name, model_name='vit_base_patch16_224', feature_dim=768, pretrained=True):
        super().__init__()
        self.task_name = task_name
        self.backbone = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=0)
        self.pretrained_weight = copy.deepcopy(self.backbone.state_dict())

        self.head = nn.Linear(feature_dim, num_classes_per_task[task_name])

    def forward(self, x):
        return self.head(self.backbone(x))

    def get_task_vector(self):
        return TaskVector(
            pretrained=self.pretrained_weight,
            finetuned=self.backbone.state_dict(),
            finetuned_heads=self.head.state_dict(),
            task_names=[self.task_name],
        )

class MultiTaskViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', feature_dim=768, pretrained=True,
                 task_vector=None):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.backbone = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=0)
        self.backbone.to(device)
        self.pretrained_weight = copy.deepcopy(self.backbone.state_dict())

        self.heads = nn.ModuleDict({
            task_name: nn.Linear(feature_dim, num_classes_per_task[task_name])
            for task_name in num_classes_per_task
        })

        if task_vector is not None:
            self.load_task_vector(task_vector)

    def forward(self, x, task_name):
        return self.heads[task_name](self.backbone(x))

    def load_task_vector(self, task_vector):
        updated_weight = copy.deepcopy(self.pretrained_weight)
        for name in updated_weight:
            updated_weight[name] += task_vector.backbone[name]
        self.backbone.load_state_dict(updated_weight)

        for task_name in task_vector.trained_task_names:
            self.heads[task_name].load_state_dict(task_vector.head_weights[task_name])
