import argparse
import copy
import json
import os
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(data_dir: str, batch_size: int = 16, img_size: int = 224):
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    eval_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(data_dir, "train"), train_transforms),
        "val": datasets.ImageFolder(os.path.join(data_dir, "val"), eval_transforms),
        "test": datasets.ImageFolder(os.path.join(data_dir, "test"), eval_transforms),
    }

    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        for split in image_datasets
    }

    dataset_sizes = {split: len(image_datasets[split]) for split in image_datasets}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names, image_datasets["train"].class_to_idx


def build_model(num_classes: int, fine_tune: bool = False):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if fine_tune:
        # Débloque layer4 pour fine-tuning
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        print("✓ Fine-tuning: layer4 débloquée")
    else:
        # Mode transfer learning: gèle toutes les couches
        for param in model.parameters():
            param.requires_grad = False
        print("✓ Transfer Learning: toutes les couches gelées")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def train_model(model, dataloaders, dataset_sizes, device, epochs: int, lr: float, fine_tune: bool = False):
    criterion = nn.CrossEntropyLoss()
    
    # Sélectionne les paramètres à optimiser
    if fine_tune:
        params_to_train = list(model.fc.parameters()) + list(model.layer4.parameters())
        print(f"✓ Optimisant: {len(params_to_train)} paramètres (FC + layer4)")
    else:
        params_to_train = model.fc.parameters()
        print(f"✓ Optimisant: seulement la couche FC")
    
    optimizer = optim.Adam(params_to_train, lr=lr)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase:5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    model.load_state_dict(best_model_wts)
    return model, history, best_val_acc


def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("Test classification report:")
    print(report)
    return report


def plot_history(history, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved curves to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning image classifier")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fine_tune", action="store_true", help="Active le fine-tuning de layer4")
    parser.add_argument("--model_out", type=str, default="models/best_model.pth")
    parser.add_argument("--class_map_out", type=str, default="models/class_to_idx.json")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    start = time.time()
    dataloaders, dataset_sizes, class_names, class_to_idx = build_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )

    print(f"Classes: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    model = build_model(num_classes=len(class_names), fine_tune=args.fine_tune).to(device)
    model, history, best_val_acc = train_model(
        model,
        dataloaders,
        dataset_sizes,
        device,
        epochs=args.epochs,
        lr=args.lr,
        fine_tune=args.fine_tune,
    )

    report = evaluate_model(model, dataloaders["test"], device, class_names)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    with open(args.class_map_out, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    plot_history(history, args.output_dir)

    elapsed = time.time() - start
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Model saved to: {args.model_out}")
    print(f"Class mapping saved to: {args.class_map_out}")
    print(f"Done in {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()
