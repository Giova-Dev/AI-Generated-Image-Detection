"""
Test di tutti i modelli salvati in MODEL_DIR sul dataset TEST_DIR.

Supporta sia modelli PyTorch (.pth, ResNet18) sia modelli scikit-learn (.pkl, CLIP+LogReg).

Uso:
    python -m src.global_test
    python -m src.global_test --models_dir models --test_dir data/GLOBAL_TEST --output_dir test_results
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from src.utils import device, load_clip_model

Image.MAX_IMAGE_PIXELS = None

MODEL_DIR = Path("models")
TEST_DIR = Path("data/GLOBAL_TEST")
OUTPUT_DIR = Path("test_results")
BATCH_SIZE = 32
NUM_WORKERS = 4

transform = transforms.Compose([
    transforms.Resize(256),                
    transforms.CenterCrop(224),            
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

clip_model = None
preprocess_clip = None


def is_valid_image(path):
    """Scarta i file non decodificabili da PIL prima che il DataLoader li carichi."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def safe_pil_loader(path):
    """Apre un'immagine e la converte in RGB. Le immagini in modalita' palette (P)
    con trasparenza vengono prima convertite in RGBA, per evitare il warning di PIL
    sulla conversione implicita."""
    with open(path, "rb") as f:
        img = Image.open(f)
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
        return img.convert("RGB")


def list_available_models(models_dir):
    """Restituisce la lista dei file .pth/.pkl nella cartella specificata."""
    return sorted(p.name for p in models_dir.iterdir() if p.suffix in (".pth", ".pkl"))


def load_torch_model(model_path):
    """Carica un checkpoint ResNet18 (.pth) e restituisce il modello in eval mode."""
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


def load_logreg_model(model_path):
    """Carica un modello CLIP+LogReg (.pkl) e restituisce il classificatore e le classi."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    clf = data["clf"]
    class_names = data["class_names"]
    return clf, class_names


def extract_clip_features(loader):
    """Estrae le feature CLIP da un dataloader (richiede clip_model già inizializzato)."""
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            batch_features = clip_model.encode_image(imgs.to(device)).cpu().numpy()
            features.append(batch_features)
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)


def evaluate_torch_model(model, test_loader, class_names):
    """Valuta un modello PyTorch (ResNet18) e restituisce metriche."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return compute_metrics(all_labels, all_preds, class_names)


def evaluate_logreg_model(clf, test_loader, class_names):
    """Valuta un classificatore logistico su feature CLIP estratte dal test set."""
    X_test, y_test = extract_clip_features(test_loader)
    y_pred = clf.predict(X_test)
    return compute_metrics(y_test, y_pred, class_names)


def compute_metrics(y_true, y_pred, class_names):
    """Calcola accuracy, confusion matrix e report di classificazione."""
    accuracy = 100 * np.mean(np.array(y_true) == np.array(y_pred))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return {
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "total_samples": len(y_true),
    }


def main():
    global clip_model, preprocess_clip

    parser = argparse.ArgumentParser(description="Testa tutti i modelli salvati su un dataset di test.")
    parser.add_argument("--models_dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--test_dir", type=Path, default=TEST_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    if not args.test_dir.exists():
        raise FileNotFoundError(f"Cartella di test non trovata: {args.test_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_files = list_available_models(args.models_dir)
    if not model_files:
        print(f"Nessun modello trovato in {args.models_dir}")
        return

    torch_models = [f for f in model_files if f.endswith('.pth')]
    logreg_models = [f for f in model_files if f.endswith('.pkl')]

    print(f"Trovati {len(torch_models)} modelli PyTorch e {len(logreg_models)} modelli CLIP+LogReg.")

    test_loader_resnet = None
    if torch_models:
        print("Caricamento dataset di test per ResNet18...")
        test_dataset_resnet = datasets.ImageFolder(
            str(args.test_dir),
            transform=transform,
            loader=safe_pil_loader,
            is_valid_file=is_valid_image,
        )
        test_loader_resnet = DataLoader(
            test_dataset_resnet,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"Test set ResNet: {len(test_dataset_resnet)} immagini, classi: {test_dataset_resnet.classes}")

    test_loader_clip = None
    if logreg_models:
        print("Caricamento dataset di test per CLIP...")
        clip_model, preprocess_clip = load_clip_model()
        test_dataset_clip = datasets.ImageFolder(
            str(args.test_dir),
            transform=preprocess_clip,
            loader=safe_pil_loader,
            is_valid_file=is_valid_image,
        )
        test_loader_clip = DataLoader(
            test_dataset_clip,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"Test set CLIP: {len(test_dataset_clip)} immagini, classi: {test_dataset_clip.classes}")

    for model_file in model_files:
        print(f"\n--- Test del modello: {model_file} ---")
        model_path = args.models_dir / model_file

        try:
            if model_file.endswith('.pth'):
                model, class_names = load_torch_model(model_path)

                if class_names != test_dataset_resnet.classes:
                    print(f"Attenzione: classi del modello ({class_names}) non coincidono con il test set "
                          f"({test_dataset_resnet.classes}). Modello saltato.")
                    continue
                results = evaluate_torch_model(model, test_loader_resnet, class_names)
                results["model_type"] = "resnet18"

            elif model_file.endswith('.pkl'):
                clf, class_names = load_logreg_model(model_path)
                if class_names != test_dataset_clip.classes:
                    print(f"Attenzione: classi del modello ({class_names}) non coincidono con il test set "
                          f"({test_dataset_clip.classes}). Modello saltato.")
                    continue
                results = evaluate_logreg_model(clf, test_loader_clip, class_names)
                results["model_type"] = "clip_logreg"

            results["model_file"] = model_file
            results["model_class_names"] = class_names
            results["test_dir"] = str(args.test_dir)

            output_filename = model_file.replace(".pth", "_results.json").replace(".pkl", "_results.json")
            output_path = args.output_dir / output_filename
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            print(f"Accuracy: {results['accuracy']:.2f}%")
            print(f"Risultati salvati in: {output_path}")

        except Exception as e:
            print(f"Errore durante il test di {model_file}: {e}")
            continue

    print("\nTest completato.")


if __name__ == "__main__":
    main()