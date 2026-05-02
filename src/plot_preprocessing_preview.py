"""
Generate a preprocessing preview figure for a sample image.
Creates a 2x2 plot showing:
- Original image
- Resized image (224x224)
- Tensor preview after ToTensor()
- Augmented examples

Usage:
    python src/plot_preprocessing_preview.py --image raw_images/keyboard/1.avif --output outputs/preprocessing_preview.png
    python src/plot_preprocessing_preview.py --class_name keyboard --output outputs/preprocessing_preview.png
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}


def find_sample_image(raw_dir: Path, class_name: str | None = None) -> tuple[Path, str]:
    """Return a sample image path and its class name."""
    if class_name:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        candidates = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not candidates:
            raise FileNotFoundError(f"No image found in: {class_dir}")
        return sorted(candidates)[0], class_name

    class_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found in: {raw_dir}")

    for class_dir in sorted(class_dirs):
        candidates = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if candidates:
            return sorted(candidates)[0], class_dir.name

    raise FileNotFoundError(f"No image files found in: {raw_dir}")


def load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def to_display_array(image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().permute(1, 2, 0).numpy()
    else:
        array = np.asarray(image, dtype=np.float32) / 255.0

    array = np.clip(array, 0.0, 1.0)
    return array


def build_transforms(img_size: int = 224):
    resize_transform = transforms.Resize((img_size, img_size))
    to_tensor = transforms.ToTensor()

    augment_transforms = [
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(degrees=12),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        ),
    ]

    return resize_transform, to_tensor, augment_transforms


def plot_preview(image_path: Path, class_name: str, output_path: Path, img_size: int = 224):
    original = load_image(image_path)
    resize_transform, to_tensor, augment_transforms = build_transforms(img_size=img_size)

    resized = resize_transform(original)
    normalized = to_tensor(resized)

    augmented_1 = augment_transforms[0](original)
    augmented_2 = augment_transforms[2](original)

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    fig.suptitle(f"Preprocessing preview - class: {class_name}", fontsize=14)

    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(original)
    ax1.set_title("Original image", fontsize=11)
    ax1.axis("off")

    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(to_display_array(resized))
    ax2.set_title(f"Resized to {img_size}x{img_size}", fontsize=11)
    ax2.axis("off")

    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(to_display_array(normalized))
    ax3.set_title("Normalized (0 to 1)", fontsize=11)
    ax3.axis("off")

    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(np.hstack([to_display_array(augmented_1), to_display_array(augmented_2)]))
    ax4.set_title("Augmented examples", fontsize=11)
    ax4.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a preprocessing preview figure")
    parser.add_argument("--raw_dir", type=str, default="raw_images")
    parser.add_argument("--image", type=str, default=None, help="Optional path to a specific image")
    parser.add_argument("--class_name", type=str, default=None, help="Optional class folder name")
    parser.add_argument("--output", type=str, default="outputs/preprocessing_preview.png")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.image is not None:
        image_path = Path(args.image)
        if not image_path.exists():
            raise SystemExit(f"Image not found: {image_path}")
        class_name = args.class_name or image_path.parent.name
    else:
        image_path, class_name = find_sample_image(Path(args.raw_dir), args.class_name)

    plot_preview(image_path, class_name, Path(args.output), img_size=args.img_size)


if __name__ == "__main__":
    main()
