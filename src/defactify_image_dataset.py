"""
Scarica un subset di Rajarshi-Roy-research/Defactify_Image_Dataset e lo salva su disco 
in cartelle REAL/FAKE.

Uso:
    python -m src.defactify_image_dataset --split train --n_per_class 5000
    python -m src.defactify_image_dataset --split test --n_per_class 1000
"""
import argparse
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path("data/Defactify_Image_Dataset")

LABEL_TO_FOLDER = {0: "REAL", 1: "FAKE"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test", "validation"], required=True)
    parser.add_argument("--n_per_class", type=int, default=5000)
    args = parser.parse_args()

    print(f"Inizio download dello split '{args.split}' via streaming...")

    dataset = load_dataset(
        "Rajarshi-Roy-research/Defactify_Image_Dataset",
        split=args.split,
        streaming=True
    ).shuffle(seed=42, buffer_size=10_000)

    counts = {"FAKE": 0, "REAL": 0}
    out_split_dir = OUT_DIR / args.split

    for folder in LABEL_TO_FOLDER.values():
        (out_split_dir / folder).mkdir(parents=True, exist_ok=True)

    for i, example in enumerate(dataset):
        folder = LABEL_TO_FOLDER[example["Label_A"]]

        if counts[folder] >= args.n_per_class:
            if all(c >= args.n_per_class for c in counts.values()):
                break
            continue

        img = example["Image"].convert("RGB")
        img.save(out_split_dir / folder / f"{i}.jpg")
        counts[folder] += 1

        if sum(counts.values()) % 100 == 0:
            print(f"Progresso: {counts['REAL']} REAL, {counts['FAKE']} FAKE...")

    print(f"\nCompletato! Split '{args.split}': salvate {counts['REAL']} REAL, {counts['FAKE']} FAKE in {out_split_dir}")


if __name__ == "__main__":
    main()