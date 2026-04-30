# Projet debutant Deep Learning - Classification d'images (4 classes)

Ce projet classe des images de materiel informatique dans 4 categories :
- keyboard
- mouse
- laptop
- monitor

Le modele utilise le **Transfer Learning** avec **ResNet18 pre-entraine** (torchvision).

## 1) Pourquoi ce projet est bien pour debuter

- Tu reutilises un modele deja entraine sur ImageNet.
- Tu ne modifies que la derniere couche pour tes 4 classes.
- Tu as un pipeline complet : preparation des donnees, entrainement, validation, test, prediction.

## 2) Structure du projet

```
.
|-- data/
|   |-- train/
|   |   |-- keyboard/
|   |   |-- mouse/
|   |   |-- laptop/
|   |   `-- monitor/
|   |-- val/
|   |   |-- keyboard/
|   |   |-- mouse/
|   |   |-- laptop/
|   |   `-- monitor/
|   `-- test/
|       |-- keyboard/
|       |-- mouse/
|       |-- laptop/
|       `-- monitor/
|-- models/
|-- outputs/
|-- src/
|   |-- train.py
|   `-- predict.py
|-- requirements.txt
`-- README.md
```

## 3) Preparer les donnees

Dans chaque dossier de classe, ajoute des images (.jpg/.png).

Exemple :
- `data/train/keyboard/img1.jpg`
- `data/val/keyboard/img2.jpg`
- `data/test/keyboard/img3.jpg`

Regle simple pour debuter :
- train: 70%
- val: 15%
- test: 15%

Minimum conseille pour commencer proprement :
- au moins 50 images par classe (mieux: 100+)

## 4) Installation

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 5) Entrainer le modele

```powershell
python src/train.py --data_dir data --epochs 8 --batch_size 16 --lr 0.001
```

Le script va :
- charger les datasets
- entrainer sur train
- evaluer sur val a chaque epoch
- garder le meilleur modele
- evaluer le modele final sur test
- sauvegarder:
  - `models/best_model.pth`
  - `models/class_to_idx.json`
  - `outputs/training_curves.png`
  - `outputs/classification_report.txt`

## 6) Predire une nouvelle image

```powershell
python src/predict.py --image "chemin/vers/ton_image.jpg"
```

Exemple de sortie :
- Prediction: mouse
- Confidence: 93.45%

## 7) Comment comprendre ce qu'il se passe (version simple)

- Le modele ResNet18 sait deja extraire des "features" visuelles (formes, textures).
- On bloque ses couches internes (`requires_grad=False`).
- On remplace seulement la couche finale par une couche pour 4 classes.
- Pendant l'entrainement, seule cette couche apprend tes categories.

En pratique, c'est plus rapide, plus stable, et ca marche mieux qu'un modele entraine de zero quand on a peu de donnees.

## 8) Problemes frequents

- Erreur "Found 0 files": il manque des images dans un dossier de classe.
- Mauvaise precision: trop peu d'images ou images floues/melangees.
- Overfitting: train tres haut, val faible -> ajouter plus de donnees, ou augmenter les augmentations.

## 9) Prochaine amelioration

- Debloquer progressivement quelques couches de ResNet (fine-tuning).
- Ajouter un script de confusion matrix.
- Tester des modeles comme EfficientNet.

---

Si tu veux, je peux maintenant te generer automatiquement :
1. un mini jeu de donnees de test factice (pour verifier le pipeline),
2. une version notebook Jupyter pedagogique ligne par ligne.
