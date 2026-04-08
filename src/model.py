import torch
import torch.nn as nn
import timm
import copy

from .task_vector import TaskVector
from .utils import num_classes_per_task


_VIT_CONFIGS = {
    'vit_base_patch32_224':  {'model_name': 'vit_base_patch32_224',  'feature_dim': 768},
    'vit_base_patch16_224':  {'model_name': 'vit_base_patch16_224',  'feature_dim': 768},
    'vit_large_patch16_224': {'model_name': 'vit_large_patch16_224', 'feature_dim': 1024},
}

_CLIP_CONFIGS = {
    'ViT-B-32': {'feature_dim': 512},
    'ViT-B-16': {'feature_dim': 512},
    'ViT-L-14': {'feature_dim': 768},
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

class SingleTaskCLIPLinear(nn.Module):
    def __init__(self, task_name, clip_arch='ViT-B-32'):
        super().__init__()
        import open_clip

        self.task_name = task_name
        self.clip_arch = clip_arch
        feature_dim = _CLIP_CONFIGS[clip_arch]['feature_dim']

        clip_model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained='openai')
        clip_model = clip_model.float()
        self.backbone = clip_model.visual
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


class MultiTaskCLIPLinear(nn.Module):
    def __init__(self, clip_arch='ViT-B-32'):
        super().__init__()
        import open_clip

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_dim = _CLIP_CONFIGS[clip_arch]['feature_dim']

        clip_model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained='openai')
        clip_model = clip_model.float()
        self.backbone = clip_model.visual.to(device)
        self.pretrained_weight = copy.deepcopy(self.backbone.state_dict())

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


class SingleTaskCLIP(nn.Module):
    def __init__(self, task_name, clip_arch='ViT-B-32'):
        super().__init__()
        import open_clip
        from dataset.classnames import CLASSNAMES, TEMPLATES

        self.task_name = task_name
        self.clip_arch = clip_arch

        clip_model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained='openai')
        clip_model = clip_model.float()
        self.backbone = clip_model.visual
        self.pretrained_weight = copy.deepcopy(self.backbone.state_dict())
        self.logit_scale = clip_model.logit_scale.exp().item()

        tokenizer = open_clip.get_tokenizer(clip_arch)
        class_names = CLASSNAMES[task_name]
        templates = TEMPLATES.get(task_name, ['a photo of a {}'])
        zs_weights = []
        with torch.no_grad():
            for classname in class_names:
                texts = tokenizer([t.format(classname) for t in templates])
                embs = clip_model.encode_text(texts)
                embs = embs / embs.norm(dim=-1, keepdim=True)
                emb = embs.mean(dim=0)
                emb = emb / emb.norm()
                zs_weights.append(emb)
        # persistent=False: moves with .to(device) but excluded from state_dict
        self.register_buffer('zeroshot_weights', torch.stack(zs_weights, dim=1), persistent=False)  # (D, C)

    def forward(self, x):
        image_features = self.backbone(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return self.logit_scale * (image_features @ self.zeroshot_weights)

    def get_task_vector(self):
        return TaskVector(
            pretrained=self.pretrained_weight,
            finetuned=self.backbone.state_dict(),
            finetuned_heads={},
            task_names=[self.task_name],
        )


class MultiTaskCLIP(nn.Module):
    def __init__(self, clip_arch='ViT-B-32'):
        super().__init__()
        import open_clip
        from dataset.classnames import CLASSNAMES, TEMPLATES

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        clip_model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained='openai')
        clip_model = clip_model.float()
        self.backbone = clip_model.visual.to(device)
        self.pretrained_weight = copy.deepcopy(self.backbone.state_dict())
        self.logit_scale = clip_model.logit_scale.exp().item()

        tokenizer = open_clip.get_tokenizer(clip_arch)
        self.zeroshot_weights: dict[str, torch.Tensor] = {}
        clip_model.to(device)
        with torch.no_grad():
            for task_name, class_names in CLASSNAMES.items():
                templates = TEMPLATES.get(task_name, ['a photo of a {}'])
                zs_weights = []
                for classname in class_names:
                    texts = tokenizer([t.format(classname) for t in templates]).to(device)
                    embs = clip_model.encode_text(texts)        # (T, D)
                    embs = embs / embs.norm(dim=-1, keepdim=True)
                    emb = embs.mean(dim=0)
                    emb = emb / emb.norm()
                    zs_weights.append(emb)
                # shape: (D, num_classes)
                self.zeroshot_weights[task_name] = torch.stack(zs_weights, dim=1)
        # keep text encoder off GPU after pre-computation
        clip_model.cpu()

    def forward(self, x, task_name):
        image_features = self.backbone(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        zs_weights = self.zeroshot_weights[task_name].to(image_features.device)
        return self.logit_scale * (image_features @ zs_weights)

    def load_task_vector(self, task_vector):
        updated_weight = copy.deepcopy(self.pretrained_weight)
        for name in updated_weight:
            updated_weight[name] += task_vector.backbone[name]
        self.backbone.load_state_dict(updated_weight)
