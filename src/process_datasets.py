"""
Elabora in batch tutti i dataset scaricati con 'hf download <repo> --repo-type dataset
--local-dir dataset/<nome>'. Per ogni cartella in datasets/ non ancora presente in
data/, estrae le immagini REAL/FAKE nella struttura standard data/<nome>/train|test/.
Dataset gia' presenti in data/ vengono saltati.

La colonna etichetta e il valore che identifica REAL cambiano da dataset a dataset
e vanno registrati qui sotto in CONFIG (non sono automatizzabili in modo affidabile).

Uso:
    python -m src.process_datasets --all
    python -m src.process_datasets --n_per_class 5000
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from PIL import Image as PILImage

DATASETS_DIR = Path("dataset")
DATA_DIR = Path("data")
SEED = 42

CONFIG = {
    "Defactify_Image_Dataset": {"label_column": "Label_A", "real_value": "0"},
    "AI-vs-Real": {"label_column": "binary_label", "real_value": "1"},
    "AI-Generated-vs-Real-Images-Datasets": {"label_column": "label", "real_value": "1"}
    # aggiungi qui ogni nuovo dataset scaricato in datasets/
}


def detect_image_column(example):
    for key, value in example.items():
        if isinstance(value, PILImage.Image):
            return key
    raise ValueError(
        f"Nessuna colonna immagine rilevata. Colonne disponibili: "
        f"{ {k: type(v).__name__ for k, v in example.items()} }"
    )
 
 
def find_split_files(dataset_dir):
    """Divide i parquet trovati in dataset_dir tra train/test in base al nome file."""
    parquet_files = list(dataset_dir.rglob("*.parquet"))
    train_files = [str(p) for p in parquet_files if "test" not in p.name.lower() and "valid" not in p.name.lower()]
    test_files = [str(p) for p in parquet_files if "test" in p.name.lower()]
    return train_files, test_files
 
 
def make_streaming_split(ds, test_fraction=0.2):
    """Split train/test deterministico che resta in streaming: una riga ogni
    1/test_fraction va al test, le altre al train. Usato quando il dataset non
    ha un file di test separato."""
    modulo = round(1 / test_fraction)
    train_ds = ds.filter(lambda ex, idx: idx % modulo != 0, with_indices=True)
    test_ds = ds.filter(lambda ex, idx: idx % modulo == 0, with_indices=True)
    return train_ds, test_ds
 
 
def count_labels(ds, label_column, real_value):
    """Conta REAL/FAKE leggendo solo label_column."""
    ds = ds.select_columns([label_column]) if hasattr(ds, "select_columns") else ds
    counts = {"REAL": 0, "FAKE": 0}
    for i, example in enumerate(ds, start=1):
        folder = "REAL" if example[label_column] == real_value else "FAKE"
        counts[folder] += 1
        if i % 5000 == 0:
            print(f"    ...{i} righe lette (REAL={counts['REAL']}, FAKE={counts['FAKE']})", flush=True)
    return counts
 
 
def extract_split(ds, label_column, real_value, out_split_dir, n_per_class, all_mode, shuffle_buffer):
    # il conteggio va fatto PRIMA dello shuffle, altrimenti riempire il buffer
    # di shuffle forza gia' la decodifica delle immagini bufferizzate
    if all_mode:
        print("  Conteggio etichette...", flush=True)
        available = count_labels(ds, label_column, real_value)
        quota = min(available.values())
        print(f"  Disponibili: {available['REAL']} REAL, {available['FAKE']} FAKE -> uso {quota} per classe", flush=True)
    else:
        quota = n_per_class
 
    ds = ds.shuffle(seed=SEED, buffer_size=shuffle_buffer)
    ds_iter = iter(ds)
    first_example = next(ds_iter)
    image_column = detect_image_column(first_example)
 
    counts = {"REAL": 0, "FAKE": 0}
    for folder in counts:
        (out_split_dir / folder).mkdir(parents=True, exist_ok=True)
 
    def save(example, idx):
        folder = "REAL" if example[label_column] == real_value else "FAKE"
        if counts[folder] >= quota:
            return
        img = example[image_column].convert("RGB")
        img.save(out_split_dir / folder / f"{idx}.jpg")
        counts[folder] += 1
 
    save(first_example, 0)
    for i, example in enumerate(ds_iter, start=1):
        if all(c >= quota for c in counts.values()):
            break
        save(example, i)
        if sum(counts.values()) % 100 == 0:
            print(f"  Progresso: {counts['REAL']} REAL, {counts['FAKE']} FAKE...", flush=True)
 
    print(f"  Salvate {counts['REAL']} REAL, {counts['FAKE']} FAKE in {out_split_dir}", flush=True)
 
 
def process_dataset(dataset_dir, n_per_class, all_mode, shuffle_buffer):
    name = dataset_dir.name
    if name not in CONFIG:
        print(f"[SKIP] {name}: nessuna configurazione (label_column/real_value) in CONFIG.")
        return
 
    if (DATA_DIR / name).exists():
        print(f"[SKIP] {name}: gia' presente in data/, non riscarico.")
        return
 
    cfg = CONFIG[name]
    train_files, test_files = find_split_files(dataset_dir)
    if not train_files:
        print(f"[SKIP] {name}: nessun file .parquet trovato in {dataset_dir}.")
        return
 
    print(f"[PROCESSO] {name}")
 
    if test_files:
        train_ds = load_dataset("parquet", data_files=train_files, split="train", streaming=True)
        test_ds = load_dataset("parquet", data_files=test_files, split="train", streaming=True)
 
        first = next(iter(train_ds))
        real_value = type(first[cfg["label_column"]])(cfg["real_value"])
 
        print(" Split: train")
        extract_split(train_ds, cfg["label_column"], real_value, DATA_DIR / name / "train",
                       n_per_class, all_mode, shuffle_buffer)
        print(" Split: test")
        extract_split(test_ds, cfg["label_column"], real_value, DATA_DIR / name / "test",
                       n_per_class, all_mode, shuffle_buffer)
        return
 
    raw_ds = load_dataset("parquet", data_files=train_files, split="train", streaming=True)
    first = next(iter(raw_ds))
    real_value = type(first[cfg["label_column"]])(cfg["real_value"])
 
    if all_mode:
        print(" Conteggio etichette sull'intero file...", flush=True)
        raw_ds = load_dataset("parquet", data_files=train_files, split="train", streaming=True)
        available = count_labels(raw_ds, cfg["label_column"], real_value)
        total_quota = min(available.values())
        train_quota = round(total_quota * 0.8)
        test_quota = total_quota - train_quota
        print(f" Disponibili: {available['REAL']} REAL, {available['FAKE']} FAKE -> "
              f"quota totale {total_quota} (train={train_quota}, test={test_quota})", flush=True)
    else:
        train_quota = n_per_class
        test_quota = n_per_class
 
    raw_ds = load_dataset("parquet", data_files=train_files, split="train", streaming=True)
    train_ds, test_ds = make_streaming_split(raw_ds, test_fraction=0.2)
 
    print(" Split: train")
    extract_split(train_ds, cfg["label_column"], real_value, DATA_DIR / name / "train",
                   train_quota, False, shuffle_buffer)
    print(" Split: test")
    extract_split(test_ds, cfg["label_column"], real_value, DATA_DIR / name / "test",
                   test_quota, False, shuffle_buffer)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=5000)
    parser.add_argument("--all", action="store_true",
                         help="Usa il minimo tra le due classi come quota (ignora --n_per_class)")
    parser.add_argument("--shuffle_buffer", type=int, default=500)
    args = parser.parse_args()
 
    if not DATASETS_DIR.exists():
        print(f"Cartella {DATASETS_DIR} non trovata: scarica prima i dataset con 'hf download'.")
        return
 
    dataset_dirs = [d for d in sorted(DATASETS_DIR.iterdir()) if d.is_dir()]
    if not dataset_dirs:
        print(f"Nessuna sottocartella trovata in {DATASETS_DIR}.")
        return
 
    for dataset_dir in dataset_dirs:
        process_dataset(dataset_dir, args.n_per_class, args.all, args.shuffle_buffer)
 
 
if __name__ == "__main__":
    main()
 