# AI-Generated-Image-Detection

Progetto universitario per il rilevamento di immagini generate da intelligenza artificiale.

**Corso:** Laboratorio di intelligenza artificiale applicata  
**Autore:** Santini Giovanni
**Anno accademico:** 2025/2026

## Descrizione

Il progetto ha l'obiettivo di distinguere immagini reali da immagini sintetiche (generate da AI).  
Sono stati implementati e confrontati due approcci:

1. **ResNet18** con fine-tuning dell'ultimo layer.
2. **CLIP (ViT-B-32)** per l'estrazione delle feature, seguito da classificatori scikit-learn (Regressione Logistica e Naive Bayes).

La pipeline include preprocessing dei dataset, addestramento, valutazione e una web app Flask per l'inferenza su immagini caricate dall'utente.

## Obiettivi

- Preparare e bilanciare più dataset di immagini reali e generate.
- Addestrare modelli supervised per la classificazione binaria.
- Confrontare le performance dei diversi approcci.
- Valutare la generalizzazione su test set separati.
- Fornire una demo web per il test dei modelli.

## Dataset

I dataset utilizzati sono:

| Dataset | Descrizione |
|---------|-------------|
| CIFAKE | Immagini reali CIFAR-10 vs immagini generate |
| AI-vs-Real | Confronto diretto reali vs sintetiche |
| Defactify | Dataset per il fact-checking visivo |
| AI-Generated-vs-Real-Images | Set ampio per rilevamento generico |

I dati devono essere organizzati nella cartella `data/` con la seguente struttura:

```text
data/
├── nomeDataset1/
│   ├── train/
│   │   ├── REAL/
│   │   └── FAKE/
│   └── test/
│       ├── REAL/
│       └── FAKE/
├── nomeDataset2/
│   └── ...
└── ...
```

Le istruzioni per il download sono in `dataset/download_datasets.txt`.

## Modelli

| Modello | Tipo | Feature |
|---------|------|---------|
| `resnet18_*.pth` | PyTorch | Fine-tuning ultimo layer |
| `clip_logreg_*.pkl` | scikit-learn | CLIP + Logistic Regression |

## Struttura della repository

```text
.
├── app/                    # Applicazione Flask
│   ├── app.py
│   ├── static/css/
│   └── templates/
├── dataset/                # Script e link per il download dei dataset
├── eda/                    # Analisi esplorativa dei dati
├── models/                 # Modelli addestrati (.pth e .pkl)
├── reports/                # Report delle performance in JSON
├── src/                    # Codice sorgente
│   ├── eda.py
│   ├── global_test.py
│   ├── process_datasets.py
│   ├── train_resnet.py
│   ├── train_scikit.py
│   └── utils.py
├── test_results/           # Risultati dei test su dataset esterno
├── Dockerfile
├── .dockerignore
└── .gitignore
```

## Requisiti

- Python 3.9+
- CUDA (opzionale, per accelerazione GPU)
- Librerie principali: `torch`, `torchvision`, `clip`, `scikit-learn`, `flask`, `numpy`, `pandas`, `matplotlib`

Le dipendenze complete sono separate in due file:

- `requirements-app.txt` – per l'esecuzione della web app Flask e Docker.
- `requirements-train.txt` – per l'addestramento e la valutazione dei modelli.

## Installazione

```bash
git clone https://github.com/Giova-Dev/AI-Generated-Image-Detection.git
cd AI-Generated-Image-Detection

python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements-train.txt
```


## Addestramento

### ResNet18

```bash
python -m src.train_resnet --datasets AI-vs-Real

python -m src.train_resnet --datasets CIFAKE --n_per_class_train 3000
```

Il modello viene salvato in `models/` come `resnet18_<dataset>_<size>.pth`.

### CLIP + scikit-learn

```bash
python -m src.train_scikit --datasets AI-vs-Real

python -m src.train_scikit --datasets CIFAKE --all --batch_size 64
```

Il checkpoint viene salvato in `models/` come `clip_logreg_<dataset>_<size>.pkl`.

## Valutazione

Le metriche calcolate includono:

- Accuratezza
- Matrice di confusione
- Precision, recall e F1-score per classe

I report vengono salvati in `reports/` in formato JSON.

Per una valutazione globale su un test set unificato:

```bash
python -m src.global_test
```

I risultati vengono salvati in `test_results/`.

## Analisi esplorativa

```bash
python -m src.eda
```

I grafici comparativi vengono salvati in `eda/comparison/`.

## Web App

Avvio dell'applicazione Flask:

```bash
python app/app.py
```

L'app è disponibile su `http://localhost:5000`.  
Funzionalità:

- Selezione di uno o più modelli dalla cartella `models/`
- Upload di un'immagine
- Visualizzazione della label prevista e della confidenza

### Docker (solo web app)

L'immagine Docker è pensata esclusivamente per eseguire l'applicazione Flask `app/app.py`.
Non include dataset, script di addestramento né le dipendenze per il training.

I modelli addestrati devono essere presenti nella cartella `models/`.

```bash
docker build -t ai-image-detector .
docker run -p 5000:5000
```

Per l'addestramento usare un ambiente Python locale con `requirements-train.txt`.

## Risultati

I risultati finali sono disponibili in:

- `reports/` per i report di addestramento e validazione
- `test_results/` per i test globali