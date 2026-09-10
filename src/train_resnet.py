"""
Fine-tuning dell'ultimo layer di ResNet18 per classificare REAL/FAKE sui dataset.

Uso:
    python -m src.train_resnet
    python -m src.train_resnet --datasets CIFAKE AI-vs-Real --n_per_class_train 3000
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils import device, load_combined

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
NUM_EPOCHS = 10

transform = transforms.Compose([
    transforms.Resize(256),                
    transforms.CenterCrop(224),            
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(image_path, model, transform, class_names):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        probability = torch.nn.functional.softmax(output, dim=1)[0]

    return class_names[predicted.item()], probability[predicted.item()].item()


def train_and_evaluate(lr, num_epochs, train_loader, test_loader, class_names):
    """Allena un ResNet18 (solo ultimo layer) con il learning rate indicato e lo valuta sul test set."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    num_classes = len(class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f'  Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    return model, epoch_losses, accuracy, cm, report_dict


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
    parser.add_argument("--lr_candidates", nargs="+", type=float, default=[0.001, 0.0001, 0.01],
                         help="Learning rate da confrontare, es. --lr_candidates 0.001 0.0001 0.01")
    args = parser.parse_args()

    n_per_class_train = None if args.all else args.n_per_class_train
    n_per_class_test = None if args.all else args.n_per_class_test

    size_tag = "all" if args.all else str(args.n_per_class_train)
    output_name = args.output_name or f"resnet18_{'-'.join(args.datasets)}_{size_tag}"

    data_dirs = [Path("data") / name for name in args.datasets]
    for d in data_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Cartella dataset non trovata: {d}")

    print(f"Device: {device}")
    print("Caricamento train:")
    train_data, class_names = load_combined("train", n_per_class_train, data_dirs, transform)
    print("Caricamento test:")
    test_data, _ = load_combined("test", n_per_class_test, data_dirs, transform)

    print(f"Classi: {class_names}")
    print(f"Train totale: {len(train_data)} immagini, Test totale: {len(test_data)} immagini")

    train_loader = DataLoader(
        train_data, batch_size=32, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_data, batch_size=32,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    print(f"Confronto learning rate: {args.lr_candidates}")
    lr_search_results = {}
    best_lr = None
    best_accuracy = -1
    best_run = None

    for lr in args.lr_candidates:
        print(f"\n--- Learning rate {lr} ---")
        model, epoch_losses, accuracy, cm, report_dict = train_and_evaluate(
            lr, NUM_EPOCHS, train_loader, test_loader, class_names
        )
        print(f'Test Accuracy (lr={lr}): {accuracy:.2f}%')
        lr_search_results[str(lr)] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lr = lr
            best_run = (model, epoch_losses, accuracy, cm, report_dict)

    print(f"\nMiglior learning rate: {best_lr} (Test Accuracy: {best_accuracy:.2f}%)")
    model, epoch_losses, accuracy, cm, report_dict = best_run

    print('Training complete')
    print(f'Test Accuracy: {accuracy:.2f}%')
    print("Confusion Matrix:")
    print(cm)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }, MODEL_DIR / f"{output_name}.pth")
    print(f"Modello salvato in {MODEL_DIR / f'{output_name}.pth'}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "output_name": output_name,
        "data_dirs": [str(d) for d in data_dirs],
        "n_per_class_train": n_per_class_train if n_per_class_train is not None else "all",
        "n_per_class_test": n_per_class_test if n_per_class_test is not None else "all",
        "num_epochs": NUM_EPOCHS,
        "learning_rate": best_lr,
        "lr_search": lr_search_results,
        "class_names": class_names,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "epoch_losses": epoch_losses,
        "test_accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }
    with open(REPORT_DIR / f"{output_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Risultati salvati in {REPORT_DIR / f'{output_name}_results.json'}")


if __name__ == "__main__":
    main()