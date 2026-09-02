"""
Fine-tuning dell'ultimo layer di ResNet18 per classificare REAL/FAKE su CIFAKE
e/o AI-vs-Real. Basato sul codice dell'esercitazione, adattato per usare piu'
dataset combinabili e per lavorare correttamente su GPU (incluso Windows).

Uso:
    python -m src.train_resnet
    python -m src.train_resnet --datasets CIFAKE AI-vs-Real --n_per_class_train 3000
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

MODEL_DIR = Path("models")
NUM_EPOCHS = 10
SEED = 42

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Set device (letto una volta, usato anche da predict_image)
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


def load_combined(split, n_per_class, data_dirs):
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


def predict_image(image_path, model, transform, class_names):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        probability = torch.nn.functional.softmax(output, dim=1)[0]

    return class_names[predicted.item()], probability[predicted.item()].item()


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
    output_name = args.output_name or f"resnet18_{'-'.join(args.datasets)}_{size_tag}"

    # Load dataset: una o piu' cartelle root (data/<nome>), ciascuna con
    # sottocartelle train/test contenenti REAL/FAKE. Scelte via --datasets.
    data_dirs = [Path("data") / name for name in args.datasets]
    for d in data_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Cartella dataset non trovata: {d}")

    print(f"Device: {device}")
    print("Caricamento train:")
    train_data, class_names = load_combined("train", n_per_class_train, data_dirs)
    print("Caricamento test:")
    test_data, _ = load_combined("test", n_per_class_test, data_dirs)

    print(f"Classi: {class_names}")
    print(f"Train totale: {len(train_data)} immagini, Test totale: {len(test_data)} immagini")

    # Create data loaders. num_workers>0 carica le immagini in processi paralleli
    # mentre la GPU calcola, cosi' non resta ferma ad aspettare i dati.
    # pin_memory accelera il trasferimento CPU->GPU quando e' disponibile una GPU.
    train_loader = DataLoader(
        train_data, batch_size=32, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_data, batch_size=32,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # Load pre-trained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    # Modify final layer
    num_classes = len(class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Training loop
    epoch_losses = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

    print('Training complete')

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Confusion matrix
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Salvataggio modello + nomi classi (necessario per riusarlo, es. nella app Flask)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }, MODEL_DIR / f"{output_name}.pth")
    print(f"Modello salvato in {MODEL_DIR / f'{output_name}.pth'}")

    # Salvataggio risultati (loss, accuracy, confusion matrix, classification report)
    results = {
        "output_name": output_name,
        "data_dirs": [str(d) for d in data_dirs],
        "n_per_class_train": n_per_class_train if n_per_class_train is not None else "all",
        "n_per_class_test": n_per_class_test if n_per_class_test is not None else "all",
        "num_epochs": NUM_EPOCHS,
        "class_names": class_names,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "epoch_losses": epoch_losses,
        "test_accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }
    with open(MODEL_DIR / f"{output_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Risultati salvati in {MODEL_DIR / f'{output_name}_results.json'}")


if __name__ == "__main__":
    main()