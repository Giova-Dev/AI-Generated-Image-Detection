"""
EDA su tutti i dataset trovati in data/: bilanciamento classi, esempi visivi,
statistiche pixel, per singolo dataset e in confronto tra loro. Scopre in
automatico ogni sottocartella di data/ con struttura <nome>/train|test/REAL|FAKE.

Uso:
    python -m src.eda
"""
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

DATA_DIR = Path("data")
OUT_DIR = Path("eda")
SEED = 42
COMMON_SIZE = (128, 128)

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
})


def discover_datasets():
    return [d for d in sorted(DATA_DIR.iterdir())
            if d.is_dir() and (d / "train").exists() and (d / "test").exists()]


def count_images(dataset_dir):
    rows = []
    for split in ["train", "test"]:
        for label in ["REAL", "FAKE"]:
            folder = dataset_dir / split / label
            n = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
            rows.append({"dataset": dataset_dir.name, "split": split, "label": label, "count": n})
    return pd.DataFrame(rows)


def plot_sample_grid(dataset_dir, out_dir, n_per_class=5):
    random.seed(SEED)
    fig, axes = plt.subplots(
        2, n_per_class,
        figsize=(2.2 * n_per_class, 5.0),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    for row, label in enumerate(["REAL", "FAKE"]):
        paths = list((dataset_dir / "train" / label).glob("*.jpg"))
        sample = random.sample(paths, min(n_per_class, len(paths)))
        for col in range(n_per_class):
            ax = axes[row, col]
            if col < len(sample):
                ax.imshow(Image.open(sample[col]))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[row, 0].set_ylabel(label, fontsize=11, rotation=90, labelpad=8)
        axes[row, 0].yaxis.set_label_position("left")
    fig.suptitle(f"Esempi REAL vs FAKE - {dataset_dir.name}", fontsize=13)
    fig.savefig(out_dir / "sample_grid.png")
    plt.close(fig)


def brightness_distribution(dataset_dir, n_per_class=1000):
    random.seed(SEED)
    rows = []
    for label in ["REAL", "FAKE"]:
        paths = list((dataset_dir / "train" / label).glob("*.jpg"))
        sample = random.sample(paths, min(n_per_class, len(paths)))
        for p in sample:
            arr = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
            rows.append({"label": label, "mean_brightness": arr.mean()})
    return pd.DataFrame(rows)


def plot_brightness(df, out_dir, dataset_name):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.histplot(
        data=df, x="mean_brightness", hue="label",
        hue_order=["REAL", "FAKE"], palette="deep",
        kde=True, bins=40, alpha=0.5,
        edgecolor="white", linewidth=0.3,
        common_norm=False, stat="density",
        ax=ax,
    )
    ax.set_title(f"Distribuzione luminosita' - {dataset_name}")
    ax.set_xlabel("Luminosita' media")
    ax.set_ylabel("Densita'")
    fig.savefig(out_dir / "brightness_distribution.png")
    plt.close(fig)


def mean_image(dataset_dir, label, n=1000):
    random.seed(SEED)
    paths = list((dataset_dir / "train" / label).glob("*.jpg"))
    sample = random.sample(paths, min(n, len(paths)))
    arrs = np.stack([
        np.array(Image.open(p).convert("RGB").resize(COMMON_SIZE), dtype=np.float32)
        for p in sample
    ])
    return arrs.mean(axis=0).astype(np.uint8)


def plot_mean_images(dataset_dir, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8),
                             gridspec_kw={"wspace": 0.05})
    for ax, label in zip(axes, ["REAL", "FAKE"]):
        ax.imshow(mean_image(dataset_dir, label))
        ax.set_title(f"Media - {label}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(dataset_dir.name, fontsize=13)
    fig.savefig(out_dir / "mean_images.png")
    plt.close(fig)


def analyze_dataset(dataset_dir):
    out_dir = OUT_DIR / dataset_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = count_images(dataset_dir)
    counts.to_csv(out_dir / "class_counts.csv", index=False)
    print(f"\n{dataset_dir.name}:")
    print(counts.to_string(index=False))

    plot_sample_grid(dataset_dir, out_dir)

    brightness = brightness_distribution(dataset_dir)
    plot_brightness(brightness, out_dir, dataset_dir.name)

    plot_mean_images(dataset_dir, out_dir)

    return counts


def plot_dataset_sizes(all_counts, out_dir):
    train_counts = all_counts[all_counts["split"] == "train"].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=train_counts, x="dataset", y="count",
        hue="label", hue_order=["REAL", "FAKE"],
        palette="deep", ax=ax,
        edgecolor="white", linewidth=0.8,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=2, fontsize=9)
    ax.set_title("Immagini per classe, per dataset (train)")
    ax.set_xlabel("")
    ax.set_ylabel("Numero di immagini")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title=None)
    ax.margins(y=0.12)
    fig.savefig(out_dir / "dataset_sizes.png")
    plt.close(fig)


def plot_sample_grid_all_datasets(dataset_dirs, out_dir, label, n_per_dataset=4):
    random.seed(SEED)
    n_rows = len(dataset_dirs)
    fig, axes = plt.subplots(
        n_rows, n_per_dataset,
        figsize=(2.2 * n_per_dataset, 2.4 * n_rows),
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
        squeeze=False,
    )
    for row, dataset_dir in enumerate(dataset_dirs):
        paths = list((dataset_dir / "train" / label).glob("*.jpg"))
        sample = random.sample(paths, min(n_per_dataset, len(paths)))
        for col in range(n_per_dataset):
            ax = axes[row, col]
            if col < len(sample):
                ax.imshow(Image.open(sample[col]))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[row, 0].set_ylabel(dataset_dir.name, fontsize=10,
                                rotation=90, labelpad=8)
        axes[row, 0].yaxis.set_label_position("left")
    fig.suptitle(f"Confronto tra dataset - classe {label}", fontsize=13, y=1.0)
    fig.savefig(out_dir / f"sample_grid_all_datasets_{label}.png")
    plt.close(fig)


def main():
    dataset_dirs = discover_datasets()
    if not dataset_dirs:
        print("Nessun dataset trovato in data/ (serve <nome>/train/ e <nome>/test/).")
        return

    print(f"Dataset trovati: {[d.name for d in dataset_dirs]}")

    all_counts = pd.concat([analyze_dataset(d) for d in dataset_dirs], ignore_index=True)

    comparison_dir = OUT_DIR / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    all_counts.to_csv(comparison_dir / "all_class_counts.csv", index=False)

    plot_dataset_sizes(all_counts, comparison_dir)
    plot_sample_grid_all_datasets(dataset_dirs, comparison_dir, label="REAL")
    plot_sample_grid_all_datasets(dataset_dirs, comparison_dir, label="FAKE")

    print(f"\nReport EDA salvato in {OUT_DIR}/")


if __name__ == "__main__":
    main()