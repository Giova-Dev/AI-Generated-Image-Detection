"""
Classificazione REAL/FAKE con feature CLIP (congelato) + Logistic Regression.

Uso:
    python -m src.train_scikit
    python -m src.train_scikit --datasets CIFAKE AI-vs-Real --n_per_class_train 3000
    python -m src.train_scikit --datasets AI-vs-Real --all --batch_size 64
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader

from src.utils import SEED, device, load_clip_model, load_combined

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
BATCH_SIZE = 16
NUM_WORKERS = 4


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
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                         help="Processi paralleli per il caricamento immagini")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                         help="Batch size per l'estrazione feature")
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

    clip_model, preprocess = load_clip_model()

    print("Caricamento train:")
    train_data, class_names = load_combined("train", n_per_class_train, data_dirs, preprocess)
    print("Caricamento test:")
    test_data, _ = load_combined("test", n_per_class_test, data_dirs, preprocess)

    print(f"Classi: {class_names}")
    print(f"Train totale: {len(train_data)} immagini, Test totale: {len(test_data)} immagini")

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    print("Estrazione feature CLIP (train)...")
    X_train, y_train = extract_features(train_loader, clip_model)
    print("Estrazione feature CLIP (test)...")
    X_test, y_test = extract_features(test_loader, clip_model)

    print("Ricerca del miglior C per la Logistic Regression (GridSearchCV)...")
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=SEED),
        param_grid, cv=5, scoring="accuracy", n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
    print(f"Miglior C: {grid_search.best_params_['C']} (CV accuracy: {grid_search.best_score_ * 100:.2f}%)")

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
        "best_params": grid_search.best_params_,
        "cv_accuracy": grid_search.best_score_ * 100,
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