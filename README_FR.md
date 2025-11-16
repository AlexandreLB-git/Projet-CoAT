TITRE

Optimisation algorithmique et implémentation d’un classifieur fondé sur l’incompatibilité (CoAT)

RESUME

Ce projet étudie et optimise le classifieur CoAT (Complexity-based Analogical Transfer). Après avoir implémenté une version naïve en Python (complexité O(n^3)), nous proposons une analyse visant à ne considérer que les triplets impliquant la nouvelle situation, conduisant à une version optimisée en O(n^2). Nous présentons également une implémentation vectorisée en PyTorch exploitant le calcul GPU.

Description du travail

Analyse théorique de la fonction d’incompatibilité utilisée par CoAT et justification mathématique de la réduction de complexité.

Implémentation comparée de trois versions :
-Version naïve en Python (boucles sur triplets).
-Version optimisée en Python (itère uniquement sur les paires pertinentes).
-Version vectorisée en PyTorch (calculs parallélisés sur GPU).

Application expérimentale choisie : classification binaire de points dans le plan (prédiction de la couleur — rouge/bleu).

Protocole expérimental : génération synthétique de bases de cas, prédiction de points et mesure des temps d’exécution moyens selon la taille de la base.


Résultats principaux

Réduction théorique de la complexité de O(n^3) à O(n^2) pour l’algorithme optimisé.

Gains empiriques observés :

Version optimisée (Python) : amélioration d’un facteur ~30 par rapport à la version naïve pour une base de 100 points (temps pour prédire 1 point ≈ 0.5 s vs 15 s).

Version PyTorch (vectorisée, GPU) : amélioration d’un facteur ~3750 pour la prédiction de 100 points sur une base de 100 éléments (temps comparatif rapporté dans le rapport).


Limites et perspectives

Limites principales : saturation mémoire GPU pour de très grandes bases de cas.

Pistes d’amélioration proposées : découpage dynamique en sous-blocs pour contourner la limite mémoire, apprentissage de métriques de similarité, comparaison avec d’autres classifieurs (K-NN, SVM) et tests sur jeux de données réels et bruités.

Saturation mémoire du GPU pour des bases > 500 points

Améliorations possibles :
-découpage dynamique (batching)
-apprentissage de métriques (learning-to-rank / metric learning)
-comparaison avec d’autres classifieurs (kNN, SVM)
