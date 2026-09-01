"""
Scarica un subset di Parveshiiii/AI-vs-Real e lo salva su disco in cartelle
REAL/FAKE..

Nota: il dataset espone solo lo split "train" via Hugging Face; lo split
train/test qui e' fatto internamente con train_test_split.

Uso:
    python -m src.download_ai_vs_real --split train --n_per_class 5000
    python -m src.download_ai_vs_real --split test --n_per_class 1000
"""
import argparse
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path("data/AI-vs-Real")
SEED = 42
# label originale del dataset: 0 = AI-generated, 1 = Real
LABEL_TO_FOLDER = {0: "FAKE", 1: "REAL"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--n_per_class", type=int, default=5000)
    parser.add_argument("--test_size", type=float, default=0.2,
                         help="Frazione riservata al test nello split interno")
    args = parser.parse_args()

    full = load_dataset("Parveshiiii/AI-vs-Real", split="train", verification_mode="no_checks")
    split = full.train_test_split(test_size=args.test_size, seed=SEED)
    dataset = split[args.split]

    counts = {"FAKE": 0, "REAL": 0}
    out_split_dir = OUT_DIR / args.split
    for folder in LABEL_TO_FOLDER.values():
        (out_split_dir / folder).mkdir(parents=True, exist_ok=True)

    for i, example in enumerate(dataset):
        folder = LABEL_TO_FOLDER[example["binary_label"]]
        if counts[folder] >= args.n_per_class:
            if all(c >= args.n_per_class for c in counts.values()):
                break
            continue
        img = example["image"].convert("RGB")
        img.save(out_split_dir / folder / f"{i}.jpg")
        counts[folder] += 1

    print(f"{args.split}: salvate {counts['REAL']} REAL, {counts['FAKE']} FAKE in {out_split_dir}")


if __name__ == "__main__":
    main()