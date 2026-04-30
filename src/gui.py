"""
Interface graphique pour prédire des images
Utilise PySimpleGUI pour une interface simple et intuitive
"""

import json
import os
from io import BytesIO
from pathlib import Path
from datetime import datetime

import PySimpleGUI as sg
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def build_model(num_classes: int = 4):
    """Crée le modèle ResNet18 avec la couche FC personnalisée"""
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


def preprocess_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    """Prétraite l'image pour la prédiction"""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(img).unsqueeze(0)


def predict_image(model, image_tensor: torch.Tensor, device, idx_to_class: dict, top_k: int = 3):
    """Prédit la classe d'une image et retourne les meilleures probabilités"""
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)

    pred_class = idx_to_class[pred_idx.item()]
    confidence_pct = confidence.item() * 100

    top_predictions = []
    for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
        top_predictions.append((idx_to_class[idx], prob * 100))

    return pred_class, confidence_pct, top_predictions


def image_to_png_bytes(image_path: str, max_size=(320, 320)) -> bytes:
    """Convertit une image en bytes PNG pour affichage fiable dans PySimpleGUI."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail(max_size)
    bio = BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def format_top_predictions(top_predictions):
    lines = ["Top predictions:"]
    for rank, (name, score) in enumerate(top_predictions, start=1):
        bar_len = int(round(score / 5))
        bar = "#" * bar_len
        lines.append(f"  {rank}. {name:<10} {score:6.2f}%  {bar}")
    return "\n".join(lines)


def get_confidence_style(confidence_pct: float):
    if confidence_pct >= 80:
        return "Confiance elevee", "lime"
    if confidence_pct >= 50:
        return "Confiance moyenne", "yellow"
    return "Confiance faible", "red"


def main():
    # Configuration
    model_path = "models/best_model.pth"
    class_map_path = "models/class_to_idx.json"

    # Vérification des fichiers
    if not os.path.exists(model_path) or not os.path.exists(class_map_path):
        sg.popup_error(
            "❌ Erreur!",
            f"Modèle non trouvé!\n\n"
            f"Assurez-vous que {model_path} et {class_map_path} existent.\n"
            f"Exécutez d'abord: python src/train.py",
        )
        return

    # Initialisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_class = load_model(model_path, class_map_path, device)

    # Configuration du thème
    sg.theme("DarkBlue2")
    sg.set_options(font=("Arial", 11))

    # Définition du layout
    layout = [
        [sg.Text("📷 Predicteur d'Images - Materiel Informatique", font=("Arial", 16, "bold"))],
        [sg.Text("Sélectionnez une image:", font=("Arial", 12))],
        [
            sg.Input(key="-IMAGE_PATH-", size=(40, 1), disabled=True),
            sg.FileBrowse("Parcourir", file_types=(("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.avif"),)),
        ],
        [sg.Text("", size=(60, 1))],  # Séparateur
        [
            sg.Column(
                [
                    [sg.Image(key="-IMAGE-", size=(320, 320), background_color="gray")],
                ],
                justification="center",
            )
        ],
        [sg.Text("", size=(60, 1))],  # Séparateur
        [
            sg.Button("🔍 Prédire", button_color=("white", "green"), size=(15, 2)),
            sg.Button("💾 Sauvegarder", button_color=("white", "blue"), size=(15, 2)),
            sg.Button("🗑️ Réinitialiser", button_color=("white", "orange"), size=(15, 2)),
            sg.Button("❌ Quitter", button_color=("white", "red"), size=(15, 2)),
        ],
        [sg.Text("Etat confiance: -", key="-CONF_STATUS-", font=("Arial", 11, "bold"), text_color="white")],
        [sg.Text("", size=(60, 1))],  # Séparateur
        [
            sg.Multiline(
                size=(60, 8),
                key="-OUTPUT-",
                disabled=True,
                background_color="black",
                text_color="lime",
                font=("Courier", 10),
            )
        ],
        [sg.Text("Historique (derniers resultats):", font=("Arial", 11, "bold"))],
        [
            sg.Multiline(
                size=(60, 6),
                key="-HISTORY-",
                disabled=True,
                autoscroll=True,
                background_color="#1f1f1f",
                text_color="white",
                font=("Courier", 10),
            )
        ],
    ]

    window = sg.Window("Prédicteur d'Images", layout, finalize=True)
    prediction_history = []
    latest_result = ""

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "❌ Quitter":
            break

        elif event == "🔍 Prédire":
            image_path = values["-IMAGE_PATH-"]

            if not image_path:
                sg.popup_error("❌ Erreur!", "Veuillez sélectionner une image!")
                continue

            if not os.path.exists(image_path):
                sg.popup_error("❌ Erreur!", f"Fichier non trouvé: {image_path}")
                continue

            try:
                # Affiche l'image dans un format compatible PySimpleGUI
                image_bytes = image_to_png_bytes(image_path, max_size=(320, 320))
                window["-IMAGE-"].update(data=image_bytes)

                # Prédiction
                image_tensor = preprocess_image(image_path)
                pred_class, confidence_pct, top_predictions = predict_image(
                    model, image_tensor, device, idx_to_class, top_k=3
                )

                # Affiche les résultats
                output_text = (
                    f"✅ PRÉDICTION RÉUSSIE!\n"
                    f"\n"
                    f"🕒 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"📁 Fichier: {os.path.basename(image_path)}\n"
                    f"🎯 Classe: {pred_class.upper()}\n"
                    f"📊 Confiance: {confidence_pct:.2f}%\n"
                    f"\n"
                    f"{format_top_predictions(top_predictions)}\n"
                    f"\n"
                    f"{'=' * 55}\n"
                    f"Appareils détectés:\n"
                    f"  • Clavier\n"
                    f"  • Souris\n"
                    f"  • Ordinateur Portable\n"
                    f"  • Moniteur\n"
                )
                window["-OUTPUT-"].update(output_text)

                # Mise a jour du badge de confiance
                conf_label, conf_color = get_confidence_style(confidence_pct)
                window["-CONF_STATUS-"].update(
                    value=f"Etat confiance: {conf_label} ({confidence_pct:.2f}%)",
                    text_color=conf_color,
                )

                # Mise a jour de l'historique en memoire
                history_line = (
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"{os.path.basename(image_path)} -> {pred_class} ({confidence_pct:.2f}%)"
                )
                prediction_history.append(history_line)
                prediction_history = prediction_history[-20:]
                window["-HISTORY-"].update("\n".join(prediction_history))

                latest_result = output_text

            except Exception as e:
                window["-OUTPUT-"].update(f"❌ Erreur: {str(e)}")

        elif event == "💾 Sauvegarder":
            if not latest_result:
                sg.popup_error("❌ Erreur!", "Aucun resultat a sauvegarder. Lancez d'abord une prediction.")
                continue

            try:
                outputs_dir = Path("outputs")
                outputs_dir.mkdir(parents=True, exist_ok=True)
                save_path = outputs_dir / "predictions_log.txt"

                with open(save_path, "a", encoding="utf-8") as f:
                    f.write(latest_result)
                    f.write("\n" + "-" * 60 + "\n")

                sg.popup_ok("✅ Sauvegarde reussie", f"Resultat enregistre dans:\n{save_path}")
            except Exception as e:
                sg.popup_error("❌ Erreur sauvegarde", str(e))

        elif event == "🗑️ Réinitialiser":
            window["-IMAGE_PATH-"].update("")
            window["-IMAGE-"].update(data=None)
            window["-OUTPUT-"].update("")
            window["-CONF_STATUS-"].update(value="Etat confiance: -", text_color="white")
            window["-HISTORY-"].update("")
            prediction_history = []
            latest_result = ""

    window.close()


if __name__ == "__main__":
    main()
