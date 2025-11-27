# CNN - Gender Classification Project

## Description du Projet

Ce projet implémente un réseau de neurones convolutif (CNN) pour la classification de genre à partir d'images de visages. L'objectif est de prédire le genre d'une personne en analysant des caractéristiques faciales telles que la barbe, la moustache, la forme du menton, la couleur des yeux, le chapeau, etc.

## Objectif

Développer un classificateur capable de :
- Équilibrer les performances de classification entre hommes et femmes
- Obtenir une bonne performance globale (moyenne hommes + femmes)
- Maintenir des performances similaires pour les deux genres

## Métrique d'Évaluation

La performance est évaluée selon la formule suivante :
```
Performance finale = Moyenne(Acc_hommes, Acc_femmes) - 2 × |Acc_hommes - Acc_femmes|
```

Cette métrique pénalise fortement les déséquilibres de performance entre genres, encourageant ainsi un modèle équitable.

## Structure du Repository
```
CNN/
├── projet/
│   ├── data.py              # DataLoader et prétraitement
│   ├── model.py             # Architecture CNN
│   ├── train.py             # Script d'entraînement
│   ├── utils.py             # Fonctions utilitaires
│   └── runs/                # Logs TensorBoard
├── train/                   # Images d'entraînement
├── test/                    # Images de test
├── train.txt                # Paths + labels d'entraînement
├── test.txt                 # Paths + labels de test
├── .gitignore
└── README.md
```
