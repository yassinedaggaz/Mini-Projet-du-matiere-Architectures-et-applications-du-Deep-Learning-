"""
Plot image counts per class for train/test splits and save as PNG.
Generates: outputs/dataset_distribution.png

Usage:
    python src/plot_dataset_distribution.py --data_dir data --output outputs/dataset_distribution.png
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}


def count_images(data_dir: Path, splits=("train", "test")):
    counts = {split: defaultdict(int) for split in splits}
    classes = set()

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Warning: split folder not found: {split_dir}")
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            classes.add(class_name)
            n = 0
            for p in class_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    n += 1
            counts[split][class_name] = n

    classes = sorted(classes)
    return classes, counts


def plot_grouped_bar(classes, counts, splits=("train", "test"), out_path="outputs/dataset_distribution.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Prepare data
    values = []
    for split in splits:
        values.append([counts.get(split, {}).get(c, 0) for c in classes])

    x = range(len(classes))
    total_width = 0.8
    n = len(splits)
    width = total_width / n

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, vals in enumerate(values):
        positions = [pos - total_width/2 + i*width + width/2 for pos in x]
        ax.bar(positions, vals, width=width, label=splits[i].capitalize())

    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("Image Distribution by Class and Split")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="outputs/dataset_distribution.png")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Splits to include (default: train test)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    splits = tuple(args.splits)
    classes, counts = count_images(data_dir, splits=splits)

    if not classes:
        raise SystemExit("No classes found in the specified data directory.")

    # Print summary
    print("Classes found:", classes)
    for split in splits:
        print(f"{split}: ", [counts[split].get(c, 0) for c in classes])

    plot_grouped_bar(classes, counts, splits=splits, out_path=args.output)


if __name__ == "__main__":
    main()
