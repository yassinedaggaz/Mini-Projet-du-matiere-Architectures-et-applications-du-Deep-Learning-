# Rapport de projet

## Titre
Classification d'images de materiel informatique par Transfer Learning (ResNet18)

## 1. Contexte et objectif
Ce mini-projet s'inscrit dans le cadre du module Architectures et applications du Deep Learning.
L'objectif est de construire un classifieur d'images capable d'identifier automatiquement 4 classes d'objets informatiques:
- keyboard
- laptop
- monitor
- mouse

Le projet suit une chaine complete:
- preparation des donnees
- entrainement du modele
- evaluation
- inference (prediction simple, batch et interface graphique)

## 2. Jeux de donnees
Le dataset est organise suivant la structure standard ImageFolder de PyTorch:
- data/train
- data/val
- data/test

Repartition observee dans ce projet:
- train: 43 images
- val: 8 images
- test: 12 images

Details par classe:
- train: keyboard 10, laptop 11, monitor 11, mouse 11
- val: keyboard 2, laptop 2, monitor 2, mouse 2
- test: keyboard 3, laptop 4, monitor 4, mouse 3

Une etape de preparation automatique est fournie via src/prepare_dataset.py, avec split configurable (70/15/15 par defaut).

## 3. Methodologie
### 3.1 Approche choisie
Le projet utilise le Transfer Learning avec ResNet18 pre-entraine sur ImageNet (torchvision).

Deux modes sont prevus dans src/train.py:
- mode transfer learning: toutes les couches convolutionnelles sont gelees, seule la couche finale est apprise
- mode fine-tuning optionnel (--fine_tune): deblocage de layer4 + apprentissage de la couche finale

### 3.2 Pretraitement et augmentation
Transformations appliquees:
- train:
  - RandomResizedCrop(224)
  - RandomHorizontalFlip
  - ColorJitter (brightness/contrast/saturation)
  - Normalisation ImageNet
- val/test:
  - Resize(224,224)
  - Normalisation ImageNet

### 3.3 Entrainement
Parametres par defaut:
- epochs: 8
- batch_size: 16
- learning rate: 1e-3
- optimiseur: Adam
- loss: CrossEntropyLoss

Le meilleur modele est selectionne selon la meilleure accuracy de validation.

## 4. Architecture logicielle
Scripts principaux:
- src/train.py: entrainement, sauvegarde du meilleur modele, rapport de classification, courbes
- src/evaluate.py: evaluation complete + matrice de confusion (texte + image)
- src/predict.py: prediction sur une image unique
- src/predict_batch.py: prediction sur un dossier et export CSV
- src/predict_ensemble.py: inference avec Test-Time Augmentation (vote majoritaire)
- src/gui.py: interface PySimpleGUI pour prediction interactive
- src/test_quick.py: test rapide sur un echantillon aleatoire
- src/prepare_dataset.py: creation automatique des splits train/val/test

Dependances (requirements.txt):
- torch, torchvision
- pillow
- matplotlib
- scikit-learn
- PySimpleGUI

## 5. Resultats experimentaux
D'apres outputs/classification_report.txt (evaluation test, 12 images):
- accuracy globale: 0.5833 (58.33%)
- macro avg F1-score: 0.6458
- weighted avg F1-score: 0.5694

Scores par classe:
- keyboard: precision 1.0000, recall 1.0000, F1 1.0000 (support 1)
- laptop: precision 0.5000, recall 0.2500, F1 0.3333 (support 4)
- monitor: precision 0.7500, recall 0.7500, F1 0.7500 (support 4)
- mouse: precision 0.4000, recall 0.6667, F1 0.5000 (support 3)

Un export batch est aussi disponible (results.csv) et montre un niveau coherent avec l'evaluation test, avec plusieurs confusions entre laptop, mouse et monitor.

## 6. Analyse des performances
Points positifs:
- pipeline complet de bout en bout
- architecture robuste pour petit dataset (transfer learning)
- presence d'outils pratiques: GUI, batch prediction, TTA, quick test

Limites observees:
- taille de dataset reduite (63 images au total)
- fort risque de variance statistique sur test (12 images seulement)
- confiance parfois moderee sur certaines classes proches visuellement
- desequilibre leger selon les splits reels et nombre tres faible en validation

Interpretation:
Le modele apprend deja des patterns utiles, mais la generalisation reste limitee par le volume de donnees et la variete visuelle des exemples.

## 7. Ameliorations proposees
1. Augmenter significativement le nombre d'images par classe (objectif minimal: 100+ / classe).
2. Nettoyer les donnees (qualite, cadrage, elimination des images ambigues).
3. Activer et comparer systematiquement le mode fine-tuning (--fine_tune).
4. Realiser une recherche d'hyperparametres (lr, batch_size, epochs, scheduler).
5. Ajouter des metriques visuelles systematiques (confusion_matrix.png, courbes train/val).
6. Mettre en place la validation croisee ou un split test plus large.
7. Tester une architecture plus recente (EfficientNet, ConvNeXt) pour comparaison.

## 8. Reproductibilite
Commandes typiques:

```bash
pip install -r requirements.txt
python src/prepare_dataset.py --raw_dir raw_images --output_dir data --clear_output
python src/train.py --data_dir data --epochs 8 --batch_size 16 --lr 0.001
python src/evaluate.py --data_dir data --split test
python src/predict.py --image "chemin/vers/image.avif"
python src/predict_batch.py --folder data/test --output results.csv
```

## 9. Conclusion
Ce projet valide la faisabilite d'un classifieur d'images multi-classes avec une approche de Transfer Learning simple et efficace pour un contexte debutant.

Le resultat actuel (58.33% d'accuracy test) constitue une base fonctionnelle. Avec davantage de donnees, un fine-tuning mieux controle et une evaluation plus solide, le systeme peut etre nettement ameliore.

## 10. Annexes
Fichiers de sortie importants:
- outputs/classification_report.txt
- results.csv
- models/class_to_idx.json

Artefacts potentiels generes par les scripts (selon execution):
- models/best_model.pth
- outputs/training_curves.png
- outputs/confusion_matrix.png
- outputs/predictions_log.txt
