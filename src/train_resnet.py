"""
Fine-tuning dell'ultimo layer di ResNet18 per classificare REAL/FAKE su CIFAKE.
Basato sul codice dell'esercitazione, adattato a data/train e data/test.

Uso:
    python -m src.train_resnet
"""
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
N_PER_CLASS_TRAIN = 5000
N_PER_CLASS_TEST = 1000
NUM_EPOCHS = 10
SEED = 42

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def balanced_subset(dataset, n_per_class, seed=SEED):
    """Restituisce un Subset con al massimo n_per_class immagini per classe."""
    random.seed(seed)
    indices_by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        indices_by_class.setdefault(label, []).append(idx)

    selected = []
    for indices in indices_by_class.values():
        selected += random.sample(indices, min(n_per_class, len(indices)))
    return Subset(dataset, selected)


# Load dataset: una o piu' cartelle root, ciascuna con sottocartelle REAL/FAKE.
# Aggiungi/rimuovi root per scegliere quali dataset usare (devono avere la
# stessa struttura di cartelle, es. CIFAKE e data_aivsreal/ dopo il download).
DATA_DIRS = [
    # Path("data/CIFAKE"),              # CIFAKE
    Path("data/AI-vs-Real"),        # AI-vs-Real (decommenta per includerlo)
]


def load_combined(split: str, n_per_class: int):
    subsets = []
    class_names = None
    for data_dir in DATA_DIRS:
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


print("Caricamento train:")
train_data, class_names = load_combined("train", N_PER_CLASS_TRAIN)
print("Caricamento test:")
test_data, _ = load_combined("test", N_PER_CLASS_TEST)

print(f"Classi: {class_names}")
print(f"Train totale: {len(train_data)} immagini, Test totale: {len(test_data)} immagini")

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

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
print(classification_report(all_labels, all_preds, target_names=class_names))

# Salvataggio modello + nomi classi
MODEL_DIR.mkdir(parents=True, exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
}, MODEL_DIR / "resnet18_clf.pth")
print(f"Modello salvato in {MODEL_DIR / 'resnet18_clf.pth'}")


def predict_image(image_path, model, transform, class_names):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        probability = torch.nn.functional.softmax(output, dim=1)[0]

    return class_names[predicted.item()], probability[predicted.item()].item()


# Example usage
# image_path = 'image.jpg'
# class_name, confidence = predict_image(image_path, model, transform, class_names)
# print(f'Predicted: {class_name} with confidence {confidence:.2f}')