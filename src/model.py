import torch
import torch.nn as nn
import timm
import copy

from .task_vector import TaskVector
from .utils import num_classes_per_task

_CLIP_FEATURE_DIM = {
    'ViT-B-32': 512,
    'ViT-B-16': 512,
    'ViT-L-14': 768,
}

_VIT_CONFIGS = {
    'vit_base_patch32_224':  {'model_name': 'vit_base_patch32_224',  'feature_dim': 768},
    'vit_base_patch16_224':  {'model_name': 'vit_base_patch16_224',  'feature_dim': 768},
    'vit_large_patch16_224': {'model_name': 'vit_large_patch16_224', 'feature_dim': 1024},
}

class SingleTaskViT(nn.Module):
    def __init__(self, task_name, vit_arch='vit_base_patch16_224', pretrained=True):
        super().__init__()
        cfg = _VIT_CONFIGS[vit_arch]
        self.task_name = task_name
        self.vit_arch = vit_arch
        self.backbone = timm.create_model(model_name=cfg['model_name'], pretrained=pretrained, num_classes=0)
        self.pretrained_weight = copy.deepcopy(self.backbone.state_dict())

        self.head = nn.Linear(cfg['feature_dim'], num_classes_per_task[task_name])

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
    def __init__(self, vit_arch='vit_base_patch16_224', pretrained=True, task_vector=None):
        super().__init__()
        cfg = _VIT_CONFIGS[vit_arch]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.backbone = timm.create_model(model_name=cfg['model_name'], pretrained=pretrained, num_classes=0)
        self.backbone.to(device)
        self.pretrained_weight = copy.deepcopy(self.backbone.state_dict())

        self.heads = nn.ModuleDict({
            task_name: nn.Linear(cfg['feature_dim'], num_classes_per_task[task_name])
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


# ---------------------------------------------------------------------------
# CLIP-based models
# ---------------------------------------------------------------------------

class SingleTaskCLIP(nn.Module):
    def __init__(self, task_name, clip_arch='ViT-B-32'):
        super().__init__()
        import open_clip
        self.task_name = task_name
        self.clip_arch = clip_arch

        clip_model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained='openai')
        self.backbone = clip_model.visual.float()
        self.pretrained_weight = copy.deepcopy(self.backbone.state_dict())

        feature_dim = _CLIP_FEATURE_DIM[clip_arch]
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


class MultiTaskCLIP(nn.Module):
    def __init__(self, clip_arch='ViT-B-32'):
        super().__init__()
        import open_clip
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        clip_model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained='openai')
        self.backbone = clip_model.visual.float().to(device)
        self.pretrained_weight = copy.deepcopy(self.backbone.state_dict())

        feature_dim = _CLIP_FEATURE_DIM[clip_arch]
        self.heads = nn.ModuleDict({
            task_name: nn.Linear(feature_dim, num_classes_per_task[task_name])
            for task_name in num_classes_per_task
        })

    def forward(self, x, task_name):
        return self.heads[task_name](self.backbone(x))

    def load_task_vector(self, task_vector):
        updated_weight = copy.deepcopy(self.pretrained_weight)
        for name in updated_weight:
            updated_weight[name] += task_vector.backbone[name]
        self.backbone.load_state_dict(updated_weight)

        for task_name in task_vector.trained_task_names:
            self.heads[task_name].load_state_dict(task_vector.head_weights[task_name])
