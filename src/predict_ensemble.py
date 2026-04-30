"""
Ensemble Learning avec Test-Time Augmentation (TTA)
Améliore la précision en prédisant plusieurs versions augmentées de l'image
"""

import argparse
import json
import os
from collections import Counter

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def build_model(num_classes: int = 4):
    """Crée le modèle ResNet18"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(model_path: str, class_map_path: str, device):
    """Charge le modèle et la correspondance des classes"""
    model = build_model(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(class_map_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, idx_to_class


def get_augmented_transforms(img_size: int = 224):
    """Retourne une liste de transformations pour l'augmentation au test-time"""
    transforms_list = [
        # Original
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        # Horizontal flip
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        # Slight rotation
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        # Color jitter
        transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    ]
    return transforms_list


def predict_with_tta(model, image_path: str, device, idx_to_class: dict, num_augmentations: int = 4):
    """Prédit avec Test-Time Augmentation (votes multiples)"""
    img = Image.open(image_path).convert("RGB")
    transforms_list = get_augmented_transforms()[:num_augmentations]

    predictions = []
    confidences = []

    print(f"\n🔄 Prédictions avec TTA ({num_augmentations} versions):")
    print("-" * 50)

    for idx, transform in enumerate(transforms_list, 1):
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)

        pred_class = idx_to_class[pred_idx.item()]
        confidence_pct = confidence.item() * 100

        predictions.append(pred_class)
        confidences.append(confidence_pct)

        print(f"  Version {idx}: {pred_class.upper():12s} (confiance: {confidence_pct:6.2f}%)")

    # Vote majoritaire
    vote_result = Counter(predictions).most_common(1)[0]
    final_class = vote_result[0]
    votes_count = vote_result[1]
    avg_confidence = sum(confidences) / len(confidences)

    print("-" * 50)
    print(f"\n✅ RÉSULTAT FINAL (ENSEMBLE):")
    print(f"   Classe: {final_class.upper()}")
    print(f"   Votes: {votes_count}/{num_augmentations}")
    print(f"   Confiance moyenne: {avg_confidence:.2f}%")

    return final_class, avg_confidence, predictions


def main():
    parser = argparse.ArgumentParser(description="Ensemble prediction avec Test-Time Augmentation")
    parser.add_argument("--image", type=str, required=True, help="Chemin vers l'image")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Chemin du modèle")
    parser.add_argument("--class_map", type=str, default="models/class_to_idx.json", help="Chemin de la correspondance des classes")
    parser.add_argument("--num_augmentations", type=int, default=4, help="Nombre de versions augmentées à tester")
    args = parser.parse_args()

    # Vérifications
    if not os.path.exists(args.image):
        print(f"❌ Erreur: Image non trouvée: {args.image}")
        return

    if not os.path.exists(args.model) or not os.path.exists(args.class_map):
        print(f"❌ Erreur: Modèle ou class map non trouvé!")
        print(f"   Assurez-vous que {args.model} et {args.class_map} existent")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    print(f"✓ Image: {os.path.basename(args.image)}")

    # Chargement
    model, idx_to_class = load_model(args.model, args.class_map, device)

    # Prédiction avec TTA
    final_class, avg_confidence, all_predictions = predict_with_tta(
        model, args.image, device, idx_to_class, num_augmentations=args.num_augmentations
    )


if __name__ == "__main__":
    main()
