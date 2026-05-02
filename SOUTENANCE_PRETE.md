# Soutenance prete - Mini projet Deep Learning

## Objectif
Te donner une trame directement presentable a l oral:
- plan de slides
- texte a dire
- points techniques a justifier
- reponses aux questions probables du jury

Duree cible: 8 a 10 minutes

---

## Slide 1 - Titre et contexte (30-45 sec)
Titre suggere:
Classification d images de materiel informatique par Transfer Learning

A dire:
"Dans ce projet, j ai developpe un systeme de classification d images pour reconnaitre 4 classes: keyboard, laptop, monitor et mouse. L objectif etait de mettre en place un pipeline Deep Learning complet, de la preparation des donnees jusqu a la prediction." 

Contenu slide:
- Nom du module
- Nom de l etudiant
- 4 classes ciblees
- Outils: PyTorch, torchvision, scikit-learn

---

## Slide 2 - Problematique et objectifs (45 sec)
A dire:
"Le probleme est une classification multi-classes sur un petit dataset. Le vrai enjeu est d obtenir un modele exploitable malgre peu de donnees, d ou le choix du Transfer Learning."

Contenu slide:
- Probleme: classification image 4 classes
- Contraintes: dataset limite
- Objectifs:
  - modele fonctionnel
  - evaluation objective
  - outils de prediction utilisables

---

## Slide 3 - Donnees et split (60 sec)
A dire:
"Le dataset est organise au format ImageFolder de PyTorch, avec une separation train/val/test."

Contenu slide:
- Repartition globale:
  - train: 43
  - val: 8
  - test: 12
- Detail par classe:
  - train: keyboard 10, laptop 11, monitor 11, mouse 11
  - val: keyboard 2, laptop 2, monitor 2, mouse 2
  - test: keyboard 3, laptop 4, monitor 4, mouse 3
- Script de split: src/prepare_dataset.py

Message important a dire:
"La taille du test est petite, donc les performances sont sensibles a chaque erreur."

---

## Slide 4 - Choix du modele (60 sec)
A dire:
"J ai utilise ResNet18 pre-entraine sur ImageNet, puis j ai remplace la couche finale pour l adapter a mes 4 classes."

Contenu slide:
- Backbone: ResNet18 pre-entraine
- Tete de classification: Linear vers 4 classes
- Deux modes:
  - transfer learning (couches gelees)
  - fine-tuning partiel de layer4 (optionnel)

Justification:
- rapide a entrainer
- robuste pour petits datasets
- meilleure base qu un entrainement from scratch

---

## Slide 5 - Pipeline d entrainement (60 sec)
A dire:
"Le pipeline applique des augmentations sur train et une normalisation standard ImageNet sur tous les splits."

Contenu slide:
- Preprocessing train:
  - RandomResizedCrop 224
  - RandomHorizontalFlip
  - ColorJitter
- Validation/test:
  - Resize 224x224
  - Normalisation ImageNet
- Hyperparametres:
  - epochs 8
  - batch size 16
  - lr 1e-3
  - Adam + CrossEntropyLoss
- Selection du meilleur modele par accuracy validation

---

## Slide 6 - Resultats quantitatifs (75 sec)
A dire:
"Le modele atteint 58.33% d accuracy sur test. Les performances varient selon les classes."

Contenu slide:
- Accuracy test: 58.33%
- Macro F1: 0.6458
- Weighted F1: 0.5694
- Par classe:
  - keyboard: F1 1.0000
  - laptop: F1 0.3333
  - monitor: F1 0.7500
  - mouse: F1 0.5000

Message critique a dire:
"Le score keyboard est tres eleve mais sur un support faible, donc il faut l interpreter avec prudence."

---

## Slide 7 - Analyse des erreurs (60 sec)
A dire:
"Les confusions principales concernent laptop, mouse et monitor, probablement a cause de similarites visuelles et du faible volume de donnees."

Contenu slide:
- Exemples de confusion depuis results.csv
- Causes possibles:
  - fond non controle
  - angles de prise de vue
  - peu d exemples par classe
- Impact:
  - confiance souvent moyenne (environ 30-50% pour plusieurs images)

---

## Slide 8 - Outils developpes (60 sec)
A dire:
"J ai depasse l entrainement de base en ajoutant plusieurs outils de prediction et d evaluation."

Contenu slide:
- prediction unitaire: src/predict.py
- prediction batch + CSV: src/predict_batch.py
- prediction ensemble TTA: src/predict_ensemble.py
- interface graphique: src/gui.py
- evaluation avec matrice de confusion: src/evaluate.py

Message valeur projet:
"Le projet est directement utilisable, pas seulement experimental."

---

## Slide 9 - Limites et ameliorations (60 sec)
A dire:
"La limite principale est la taille des donnees. Les prochaines etapes visent la generalisation."

Contenu slide:
- Limites:
  - dataset reduit
  - test tres petit
  - variabilite elevee des metriques
- Plan d amelioration:
  - augmenter les donnees (100+ images/classe)
  - nettoyage et uniformisation visuelle
  - fine-tuning systematique
  - comparaison avec EfficientNet/ConvNeXt
  - evaluation plus robuste

---

## Slide 10 - Conclusion (30-45 sec)
A dire:
"Ce projet valide un pipeline Deep Learning complet pour la classification d images. Les resultats actuels sont encourageants pour une base de donnees reduite, et le systeme peut etre nettement ameliore en augmentant la qualite et la quantite des donnees."

Contenu slide:
- Pipeline complet valide
- Resultat actuel exploitable comme baseline
- Feuille de route claire pour progresser

---

## Demo courte conseillee (optionnel, 1 minute)
Si tu fais une demo live:
1. lancer prediction sur 1 image claire
2. montrer top-3 predictions
3. afficher fichier results.csv ou rapport

Commande type:
python src/predict.py --image "data/test/mouse/mouse_0000.jpg"

---

## Questions probables du jury et reponses courtes
1. Pourquoi ResNet18 et pas un modele plus grand?
Reponse: dataset petit, ResNet18 est un bon compromis precision/temps/cout de calcul.

2. Pourquoi Transfer Learning?
Reponse: il reutilise des representations visuelles deja apprises sur ImageNet et marche mieux avec peu de donnees.

3. Pourquoi accuracy seulement 58%?
Reponse: volume de donnees faible et test de 12 images, donc forte sensibilite aux erreurs. Les metriques vont monter avec plus de donnees et un fine-tuning mieux regle.

4. Comment valider une vraie amelioration?
Reponse: comparer sur le meme split test, reporter accuracy + macro F1 + matrice de confusion, et refaire plusieurs essais avec seed fixe.

5. Quelle est ta contribution principale?
Reponse: implementation de la chaine complete (preparation, training, evaluation, inference batch/single, GUI, TTA) et analyse des limites avec plan d amelioration concret.

---

## Checklist juste avant passage
- verifier que le projet s execute sans erreur
- preparer 2 images de demo (1 facile, 1 difficile)
- garder ce message de conclusion memorise (20 sec)
- anticiper 2 questions techniques (transfer learning, metriques)
