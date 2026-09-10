"""
Test di tutti i modelli salvati in MODEL_DIR sul dataset TEST_DIR.

Supporta sia modelli PyTorch (.pth, ResNet18) sia modelli scikit-learn (.pkl, CLIP+LogReg).

Uso:
    python -m src.global_test
    python -m src.global_test --models_dir models --test_dir data/GLOBAL_TEST --output_dir test_results
"""
import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from src.utils import device, load_clip_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
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


def safe_pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
        try:
            img.load()
        except Exception as e:
            print(f"[WARN] immagine non decodificabile: {path} ({e}). Uso placeholder nero.")
            return Image.new("RGB", (224, 224), (0, 0, 0))
        return img.convert("RGB")


def filter_corrupted_samples(dataset):
    prev_flag = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False

    total = len(dataset.samples)
    good_samples = []
    removed = []

    print(f"Scansione di {total} immagini per individuare file corrotti...")
    try:
        for i, (path, target) in enumerate(dataset.samples, 1):
            if i % (total / 10) == 0 or i == total:
                print(f"  [{i}/{total}] verificate finora - corrotte trovate: {len(removed)}")
            try:
                with Image.open(path) as im:
                    im.load()
                good_samples.append((path, target))
            except Exception as e:
                print(f"  [CORROTTA] {path} -> {type(e).__name__}: {e}")
                removed.append(path)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev_flag

    dataset.samples = good_samples
    dataset.imgs = good_samples
    dataset.targets = [t for _, t in good_samples]
    return removed


def list_available_models(models_dir):
    return sorted(p.name for p in models_dir.iterdir() if p.suffix in (".pth", ".pkl"))


def load_torch_model(model_path):
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
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    clf = data["clf"]
    class_names = data["class_names"]
    return clf, class_names


def extract_clip_features(loader):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            batch_features = clip_model.encode_image(imgs.to(device)).cpu().numpy()
            features.append(batch_features)
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)


def evaluate_torch_model(model, test_loader, class_names):
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


def evaluate_logreg_model(clf, X_test, y_test, class_names):
    y_pred = clf.predict(X_test)
    return compute_metrics(y_test, y_pred, class_names)


def compute_metrics(y_true, y_pred, class_names):
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

    print(f"Device: {device}")
    print(f"Trovati {len(torch_models)} modelli PyTorch e {len(logreg_models)} modelli CLIP+LogReg.")

    base_dataset = datasets.ImageFolder(str(args.test_dir), loader=safe_pil_loader)
    removed = filter_corrupted_samples(base_dataset)
    if removed:
        print(f"Rimosse {len(removed)} immagini corrotte.")
        corrupted_log = args.output_dir / "corrupted_images.txt"
        with open(corrupted_log, "w") as f:
            for p in removed:
                f.write(f"{p}\n")
        print(f"Lista salvata in: {corrupted_log}")
    else:
        print("Nessuna immagine corrotta trovata.")

    print(f"Test set valido: {len(base_dataset)} immagini, classi: {base_dataset.classes}")

    test_loader_resnet = None
    test_dataset_resnet = None
    if torch_models:
        print("Preparazione DataLoader per ResNet18...")
        test_dataset_resnet = copy.deepcopy(base_dataset)
        test_dataset_resnet.transform = transform
        test_loader_resnet = DataLoader(
            test_dataset_resnet,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    test_dataset_clip = None
    clip_X, clip_y = None, None
    if logreg_models:
        print("Preparazione DataLoader per CLIP...")
        clip_model, preprocess_clip = load_clip_model()
        test_dataset_clip = copy.deepcopy(base_dataset)
        test_dataset_clip.transform = preprocess_clip
        test_loader_clip = DataLoader(
            test_dataset_clip,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"Estrazione feature CLIP su {len(test_dataset_clip)} immagini...")
        clip_X, clip_y = extract_clip_features(test_loader_clip)
        print(f"Feature CLIP estratte: shape={clip_X.shape}")

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
                results = evaluate_logreg_model(clf, clip_X, clip_y, class_names)
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