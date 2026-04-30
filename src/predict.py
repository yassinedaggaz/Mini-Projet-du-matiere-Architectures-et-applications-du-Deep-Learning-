import argparse
import json

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


def get_image_path_interactive():
    """Affiche une fenetre pour choisir l'image graphiquement"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        image_path = filedialog.askopenfilename(
            title="Choisis une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.avif"), ("Tous les fichiers", "*.*")],
        )
        root.destroy()
        return image_path
    except ImportError:
        print("Erreur: tkinter n'est pas disponible")
        return None


def main():
    parser = argparse.ArgumentParser(description="Predict one image class")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    parser.add_argument("--class_map", type=str, default="models/class_to_idx.json")
    args = parser.parse_args()

    # Si pas d'image fournie, demande à l'utilisateur de choisir graphiquement
    if args.image is None:
        print("Pas d'image spécifiée. Ouvre la fenetre de sélection...")
        args.image = get_image_path_interactive()
        if args.image is None:
            print("Aucune image sélectionnée. Abandon.")
            return

    with open(args.class_map, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    x = preprocess_image(args.image).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        top_probs, top_indices = torch.topk(probs, k=min(3, probs.shape[1]), dim=1)

    pred_class = idx_to_class[pred_idx.item()]
    print(f"Prediction: {pred_class}")
    print(f"Confidence: {conf.item() * 100:.2f}%")
    print("Top predictions:")
    for rank, (prob, idx) in enumerate(zip(top_probs[0].tolist(), top_indices[0].tolist()), start=1):
        print(f"  {rank}. {idx_to_class[idx]} - {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
