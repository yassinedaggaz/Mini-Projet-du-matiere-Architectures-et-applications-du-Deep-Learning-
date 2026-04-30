import argparse
import json
import random
from pathlib import Path

import torch

from predict import build_model, preprocess_image


def collect_images(dataset_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
    images = []

    for class_dir in dataset_dir.iterdir():
        if not class_dir.is_dir():
            continue
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() in exts:
                images.append((image_path, class_dir.name))

    return images


def run_quick_test(dataset_dir: Path, model_path: Path, class_map_path: Path, n_samples: int):
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    all_images = collect_images(dataset_dir)
    if not all_images:
        raise ValueError(f"Aucune image trouvee dans {dataset_dir}")

    n_samples = min(n_samples, len(all_images))
    sampled = random.sample(all_images, n_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0

    print("=" * 80)
    print(f"Quick test sur {n_samples} image(s) depuis: {dataset_dir}")
    print("=" * 80)

    for i, (img_path, true_label) in enumerate(sampled, start=1):
        x = preprocess_image(str(img_path)).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        pred_label = idx_to_class[pred_idx.item()]
        confidence = conf.item() * 100
        is_ok = pred_label == true_label
        correct += int(is_ok)

        status = "OK" if is_ok else "ERREUR"
        print(
            f"[{i:02d}] {status:<6} | vrai={true_label:<10} | pred={pred_label:<10} | "
            f"conf={confidence:6.2f}% | fichier={img_path.name}"
        )

    accuracy = (correct / n_samples) * 100
    print("-" * 80)
    print(f"Resultat: {correct}/{n_samples} correct -> accuracy rapide = {accuracy:.2f}%")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Quick batch test for image classifier")
    parser.add_argument("--dataset", type=str, default="data/test")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    parser.add_argument("--class_map", type=str, default="models/class_to_idx.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    run_quick_test(
        dataset_dir=Path(args.dataset),
        model_path=Path(args.model),
        class_map_path=Path(args.class_map),
        n_samples=args.samples,
    )


if __name__ == "__main__":
    main()
