import os
import json
import random
import threading

SUPPORTED_DATASETS = [
    "MNIST", "FashionMNIST", "EMNIST", "CIFAR10", "STL10", "SVHN",
    "GTSRB", "EuroSAT", "RESISC45", "PCAM", "RenderedSST2",
    "Flowers102", "OxfordIIITPet", "Food101", "Cars", "CIFAR100", "DTD", "SUN397",
    "Country211", "Aircraft",
]

_print_lock = threading.Lock()


def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def get_config(defaults: dict, mode: str) -> dict:
    base = defaults["lp" if mode == "lp" else "ft"]
    lr = base["lr"] * (10 ** random.uniform(-0.5, 0.5))
    return {**base, "lr": lr}


def save_results(result_path: str, results: dict) -> None:
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    existing = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
    existing.update({name: round(acc * 100, 4) for name, acc in results.items()})
    with open(result_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {result_path}")
