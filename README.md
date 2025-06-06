# 🧬 Modélisation stochastique pour l'évolution

## 📋 Introduction

> Nota Bene : Ce modèle a été réalisé dans le cadre d'un projet étudiant visant à introduire des notions de modélisation stochastique en python. 

Implémentation d'un modèle d’évolution des fréquences génotypiques d’un marqueur génétique à 2 allèles évoluant selon trois scénarios possibles :

1. Reproduction sexuée panmictique
2. Reproduction sexuée autogame
3. Reproduction à 95% clonale.

Ces populations sont supposées diploïdes, à générations non-chevauchantes, de tailles finies (N=1000) et soumises à un taux de mutation de 1/N.

Le modèle est stochastique. 100 répétitions sont effectuées dans son implémentation actuelle.

## ⚙️ Dépendances

Pour fonctionner, le script a besoin que vous ayez certaines librairies installées sur votre environnement python. Ces librairies sont les suivantes :

* [numpy](https://numpy.org/doc/stable/user/index.html)
* [pandas](https://pandas.pydata.org/)
* [seaborn](https://seaborn.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [random](https://docs.python.org/3/library/random.html)
* [math](https://docs.python.org/3/library/math.html)
* [tqdm](https://github.com/tqdm/tqdm)

## ✅ Lancer le script

Le script se lance simplement avec la commande suivante :

`python3 frequencies.py`

## 📈 Sorties

Le script produit un ensemble de sorties brutes et graphiques qui seront enregistrées dans le dossier de lancement du script.

Les sorties brutes contiennent l'ensemble des générations produites pour chacun des scénarios, ainsi qu'un tableau des divergences de Kullback-Leibler à respectivement 5 et 20 générations.

Les sorties graphiques représentent les évolutions de fréquences génotypiques au fil des générations pour chaque scénario.
Deux types de sorties graphiques sont produites : l'une d'elle présente l'évolution des fréquences moyennes ainsi que trois trajectoires possibles, l'autre présente l'évolution des fréquences moyennes et son interval de confiance à 95%.
