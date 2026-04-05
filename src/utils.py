import json
import torch
import os

from dataset.dataloader import get_test_dataloader

_json_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "num_classes_per_task.json")
with open(_json_path) as f:
    num_classes_per_task = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def svd(w):
    U, S, Vt = torch.linalg.svd(w, full_matrices=True, driver="gesvd")
    return U, S, Vt.T

def frobenius_inner_product(w1, w2):
    return torch.trace(w1.T @ w2).item()

def load_task_vector(task, device):
    from .model import SingleTaskViT
    path = os.path.join('models', f'{task}.pt')
    model = SingleTaskViT(task_name=task).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.get_task_vector()

def load_task_vectors(device):
    return [load_task_vector(task, device) for task in num_classes_per_task]

def evaluate_model(model, tasks):
    results = {}
    for task_idx, task in enumerate(tasks):
        model.eval()
        correct = total = 0
        test_loader = get_test_dataloader(task, batch_size=64)

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, task)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        results[f'task_{task_idx}_{task}'] = correct / total if total > 0 else 0

    return results
