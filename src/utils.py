"""
Utility condivise per il caricamento di dataset e modelli.
"""
import random

import open_clip
import torch
from torch.utils.data import ConcatDataset, Subset
from torchvision import datasets

SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def balanced_subset(dataset, n_per_class, seed=SEED):
    """Restituisce un Subset con al massimo n_per_class immagini per classe.
    Se n_per_class e' None, usa tutte le immagini disponibili per classe."""
    random.seed(seed)
    indices_by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        indices_by_class.setdefault(label, []).append(idx)

    selected = []
    for indices in indices_by_class.values():
        if n_per_class is None:
            selected += indices
        else:
            selected += random.sample(indices, min(n_per_class, len(indices)))
    return Subset(dataset, selected)


def load_combined(split, n_per_class, data_dirs, transform):
    subsets = []
    class_names = None
    for data_dir in data_dirs:
        full = datasets.ImageFolder(str(data_dir / split), transform=transform)
        if class_names is None:
            class_names = full.classes
        else:
            assert full.classes == class_names, (
                f"Le classi di {data_dir} ({full.classes}) non coincidono con {class_names}"
            )
        subset = balanced_subset(full, n_per_class)
        print(f"  {data_dir}/{split}: {len(subset)} immagini")
        subsets.append(subset)
    return ConcatDataset(subsets), class_names


def load_clip_model(model_name="ViT-B-32-quickgelu", pretrained="openai"):
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    clip_model.eval().to(device)
    return clip_model, preprocess