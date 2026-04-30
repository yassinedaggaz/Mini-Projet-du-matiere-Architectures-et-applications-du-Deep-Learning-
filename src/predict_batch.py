import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}


def preprocess_image(image_path: str, img_size: int = 224):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor


def main():
    parser = argparse.ArgumentParser(description="Predict class for multiple images in a folder")
    parser.add_argument("--folder", type=str, required=True, help="Path to folder with images")
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    parser.add_argument("--class_map", type=str, default="models/class_to_idx.json")
    parser.add_argument("--output", type=str, default=None, help="Optional: save results to CSV")
    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Erreur: Le dossier {args.folder} n'existe pas!")
        return

    # Charge le mapping des classes
    with open(args.class_map, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Charge le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Trouve toutes les images
    image_files = [p for p in folder_path.rglob("*") if is_image_file(p)]
    if not image_files:
        print(f"Aucune image trouvée dans {args.folder}")
        return

    print(f"🖼️  Trouvé {len(image_files)} images. Prédiction en cours...\n")

    results = []
    correct = 0

    for idx, image_path in enumerate(image_files, 1):
        try:
            # Prédiction
            x = preprocess_image(str(image_path)).to(device)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, 1)

            pred_class = idx_to_class[pred_idx.item()]
            confidence = conf.item() * 100

            # Extrait le label réel du nom du fichier si possible
            true_class = None
            for class_name in idx_to_class.values():
                if class_name in image_path.parent.name:
                    true_class = class_name
                    break

            is_correct = true_class == pred_class if true_class else None
            if is_correct:
                correct += 1
                status = "✅"
            elif is_correct is False:
                status = "❌"
            else:
                status = "❓"

            result = {
                "image": image_path.name,
                "prediction": pred_class,
                "confidence": f"{confidence:.2f}%",
                "true_class": true_class,
                "correct": is_correct,
            }
            results.append(result)

            # Affiche le résultat
            print(f"{status} [{idx:3d}/{len(image_files)}] {image_path.name:30s} → {pred_class:10s} ({confidence:6.2f}%)")

        except Exception as e:
            print(f"❌ Erreur avec {image_path.name}: {str(e)}")

    # Résumé
    print(f"\n{'='*70}")
    print(f"Prédictions sur {len(image_files)} images terminées!")

    if any(r["true_class"] is not None for r in results):
        accuracy = correct / sum(1 for r in results if r["true_class"] is not None) * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{sum(1 for r in results if r['true_class'] is not None)})")

    # Sauvegarde optionnelle en CSV
    if args.output:
        import csv

        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Résultats sauvegardés dans: {args.output}")


if __name__ == "__main__":
    main()
