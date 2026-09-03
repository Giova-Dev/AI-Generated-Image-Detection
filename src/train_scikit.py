"""
Classificazione REAL/FAKE con feature CLIP (congelato) + Logistic Regression.

Uso:
    python -m src.train_scikit
    python -m src.train_scikit --datasets CIFAKE AI-vs-Real --n_per_class_train 3000
"""
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import open_clip
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
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


def load_combined(split, n_per_class, data_dirs, preprocess):
    subsets = []
    class_names = None
    for data_dir in data_dirs:
        full = datasets.ImageFolder(str(data_dir / split), transform=preprocess)
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


def extract_features(loader, clip_model):
    """Forward pass di CLIP (nessun gradiente, backbone congelato): converte
    ogni immagine in un vettore a 512 dimensioni."""
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            batch_features = clip_model.encode_image(imgs.to(device)).cpu().numpy()
            features.append(batch_features)
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["CIFAKE"],
                         help="Sottocartelle di data/ da usare, es. --datasets CIFAKE AI-vs-Real")
    parser.add_argument("--n_per_class_train", type=int, default=5000,
                         help="Immagini per classe, per ciascun dataset, nel train")
    parser.add_argument("--n_per_class_test", type=int, default=1000,
                         help="Immagini per classe, per ciascun dataset, nel test")
    parser.add_argument("--all", action="store_true",
                         help="Usa tutte le immagini disponibili per classe (ignora --n_per_class_train/test)")
    parser.add_argument("--output_name", default=None,
                         help="Nome del checkpoint (default: generato da dataset e dimensione)")
    parser.add_argument("--num_workers", type=int, default=4,
                         help="Processi paralleli per il caricamento immagini")
    args = parser.parse_args()

    n_per_class_train = None if args.all else args.n_per_class_train
    n_per_class_test = None if args.all else args.n_per_class_test

    size_tag = "all" if args.all else str(args.n_per_class_train)
    output_name = args.output_name or f"clip_logreg_{'-'.join(args.datasets)}_{size_tag}"

    data_dirs = [Path("data") / name for name in args.datasets]
    for d in data_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Cartella dataset non trovata: {d}")

    print(f"Device: {device}")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai"
    )
    clip_model.eval().to(device)

    print("Caricamento train:")
    train_data, class_names = load_combined("train", n_per_class_train, data_dirs, preprocess)
    print("Caricamento test:")
    test_data, _ = load_combined("test", n_per_class_test, data_dirs, preprocess)

    print(f"Classi: {class_names}")
    print(f"Train totale: {len(train_data)} immagini, Test totale: {len(test_data)} immagini")

    train_loader = DataLoader(
        train_data, batch_size=256,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_data, batch_size=256,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    print("Estrazione feature CLIP (train)...")
    X_train, y_train = extract_features(train_loader, clip_model)
    print("Estrazione feature CLIP (test)...")
    X_test, y_test = extract_features(test_loader, clip_model)

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)

    train_accuracy = clf.score(X_train, y_train) * 100
    print(f"Train Accuracy: {train_accuracy:.2f}%")

    y_pred = clf.predict(X_test)
    test_accuracy = clf.score(X_test, y_test) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=class_names))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / f"{output_name}.pkl", "wb") as f:
        pickle.dump({"clf": clf, "class_names": class_names}, f)
    print(f"Modello salvato in {MODEL_DIR / f'{output_name}.pkl'}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "output_name": output_name,
        "data_dirs": [str(d) for d in data_dirs],
        "n_per_class_train": n_per_class_train if n_per_class_train is not None else "all",
        "n_per_class_test": n_per_class_test if n_per_class_test is not None else "all",
        "num_epochs": "N/A",
        "class_names": class_names,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "epoch_losses": [],
        "test_accuracy": test_accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }
    with open(REPORT_DIR / f"{output_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Risultati salvati in {REPORT_DIR / f'{output_name}_results.json'}")


if __name__ == "__main__":
    main()