import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_dataloader(data_dir: str, split: str = "test"):
    eval_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(os.path.join(data_dir, split), eval_transforms)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    return dataloader, dataset.classes, dataset.class_to_idx


def plot_confusion_matrix(cm, class_names, output_dir: str):
    """Crée un joli graphe de matrice de confusion"""
    plt.figure(figsize=(10, 8))

    # Affiche la matrice
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=16)
    plt.colorbar(im)

    # Axes
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Ajoute les nombres
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=12,
        )

    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Matrice de confusion sauvegardée: {out_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model and show confusion matrix")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    parser.add_argument("--class_map", type=str, default="models/class_to_idx.json")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Charge les données
    dataloader, class_names, class_to_idx = build_dataloader(args.data_dir, split=args.split)
    print(f"Classes: {class_names}")

    # Charge le modèle
    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Prédictions
    all_preds = []
    all_labels = []

    print(f"Évaluation sur {args.split}...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calcule la matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)

    # Affiche les résultats
    print(f"\nMatrice de Confusion ({args.split}):")
    print("Prédictions →")
    print("Réel ↓")
    print(f"{'':12} {' '.join(f'{c:>10}' for c in class_names)}")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:12} {' '.join(f'{cm[i, j]:>10}' for j in range(len(class_names)))}")

    # Calcule accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # Affiche per-class accuracy
    print("\nAccuracy par classe:")
    for i, class_name in enumerate(class_names):
        if np.sum(cm[i]) > 0:
            class_acc = cm[i, i] / np.sum(cm[i])
            print(f"  {class_name}: {class_acc * 100:.2f}%")

    # Sauvegarde le graphe
    plot_confusion_matrix(cm, class_names, args.output_dir)


if __name__ == "__main__":
    main()
