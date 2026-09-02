"""
App Flask per il rilevamento di immagini generate da AI.
Permette di scegliere uno o piu' modelli (.pth in models/) e mostra
un report comparativo dei risultati.

Uso:
    python app/app.py
"""
import base64
import io
from pathlib import Path
import json
import torch
import torch.nn as nn
import torchvision.models as models
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

_model_cache = {}


def list_available_models():
    return sorted(p.name for p in MODEL_DIR.glob("*.pth"))

def list_results():
    results = []
    for json_path in sorted(REPORT_DIR.glob("*_results.json")):
        with open(json_path) as f:
            results.append(json.load(f))
    return results

def load_model(filename):
    if filename in _model_cache:
        return _model_cache[filename]

    checkpoint = torch.load(MODEL_DIR / filename, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)

    _model_cache[filename] = (model, class_names)
    return model, class_names


def predict_with_model(filename, image):
    model, class_names = load_model(filename)
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]

    predicted_idx = int(torch.argmax(probs).item())
    return {
        "model": filename,
        "label": class_names[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", available_models=list_available_models())


@app.route("/predict", methods=["POST"])
def predict():
    available_models = list_available_models()
    file = request.files.get("image")
    selected_models = request.form.getlist("models")

    if file is None or file.filename == "":
        return render_template("index.html", available_models=available_models,
                                error="Nessuna immagine caricata.")
    if not selected_models:
        return render_template("index.html", available_models=available_models,
                                error="Seleziona almeno un modello.")

    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    results = [predict_with_model(m, image) for m in selected_models]

    for r in results:
        r['probabilities_percent'] = {
            label: round(prob * 100, 1)
            for label, prob in r['probabilities'].items()
        }
        r['confidence_percent'] = round(r['confidence'] * 100, 1)

    return render_template("report.html", results=results, image_b64=image_b64)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files.get("image")
    selected_models = request.form.getlist("models") or list_available_models()

    if file is None or file.filename == "":
        return jsonify({"error": "Nessuna immagine fornita"}), 400
    if not selected_models:
        return jsonify({"error": "Nessun modello disponibile in models/"}), 400

    image = Image.open(file.stream).convert("RGB")
    results = [predict_with_model(m, image) for m in selected_models]
    return jsonify({"results": results})

@app.route("/performance", methods=["GET"])
def performance():
    return render_template("performance.html", results=list_results())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)