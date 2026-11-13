# Optimisation algorithmique d'un classifieur fondé sur la complexité (CoAT)

Alexandre Lebleu, Léonard Jung · Encadrante : Chunyang Fan · Sorbonne Université, mai 2025

📄 [Rapport complet (anglais)](CoAT_Project_Report.pdf) · [Version française](CoAT_Project_Report_FR.pdf) · [Notebook](coat_final.ipynb) · [English README](README.md)

## Présentation

CoAT (Complexity-based Analogical Transfer, Badra, IJCAI 2020) est un classifieur à base de cas. Pour étiqueter un nouveau point, il essaie chaque classe candidate et garde celle qui minimise le nombre d'*inversions de similarité* dans les données : les triplets où x_j est plus proche de x_i que x_k dans l'espace d'entrée, alors que y_j est plus éloigné de y_i que y_k dans l'espace de sortie.

L'algorithme publié compte tous les triplets de la base de cas, ce qui coûte O(n³) par prédiction et devient inutilisable au-delà de quelques centaines de points. Ce projet le rend rapide :

- **Théorie.** Nous prouvons que les inversions internes à la base de cas ne dépendent pas de la classe candidate : la prédiction ne dépend donc que des triplets contenant le nouveau point. Certains de ces sous-ensembles ne contiennent aucune inversion, ce qui se démontre. La prédiction passe ainsi à O(n²), avec des résultats identiques.
- **Version Python optimisée**, qui ne parcourt que les paires utiles.
- **Version PyTorch vectorisée**, qui remplace les boucles par des opérations tensorielles et s'exécute sur GPU.

Le notebook vérifie numériquement que les trois versions donnent les mêmes prédictions : le score O(n²) est égal au score naïf moins une constante indépendante de la classe, et la version PyTorch donne la même prédiction que la version Python sur chaque point de test.

## Résultats

Classification binaire de points du plan (données synthétiques), base de 100 cas, exécution sur un GPU Google Colab :

| Version | Complexité | Temps de prédiction | Accélération |
|---|---|---|---|
| Python naïf | O(n³) | ≈ 8 s par point | |
| Python optimisé | O(n²) | ≈ 0,25 s par point | ≈ 30× par rapport à la naïve |
| PyTorch (GPU) | O(n²), vectorisée | ≈ 3,5 ms par point | ≈ 70× par rapport à l'optimisée, plus de 2 000× par rapport à la naïve |

Le gain de la version O(n²) vient de l'algorithme et reste stable d'une exécution à l'autre. Le gain GPU dépend du matériel et de sa charge : pour un seul point, le temps mesuré correspond surtout au coût fixe de lancement des opérations sur le GPU. Le GPU devient vraiment rentable sur des lots : 100 points sont prédits en environ 10 ms avec une base de 100 cas, et en moins de 20 ms pour des bases allant jusqu'à 300 cas.

## Contenu du dépôt

| Fichier | Contenu |
|---|---|
| `CoAT_Project_Report.pdf` | Rapport complet (anglais) : preuves, algorithmes, expériences |
| `CoAT_Project_Report_FR.pdf` | Rapport original (français) |
| `coat_final.ipynb` | Implémentations naïve, optimisée et PyTorch, vérifications, mesures de temps et graphiques |

Pour l'exécuter, ouvrir `coat_final.ipynb` (par exemple dans Google Colab avec un environnement GPU). La partie PyTorch utilise un GPU CUDA s'il est disponible, sinon le CPU.

**Mémoire GPU.** Les tenseurs de comparaison de la version vectorisée croissent en p × N² (p points à prédire, N cas). Sur de grandes entrées, la mémoire du GPU peut être saturée. Le découpage des tenseurs en blocs, décrit dans le rapport, permet de l'éviter.

## Limites et pistes

- Croissance rapide de la mémoire GPU sur de très grandes bases de cas (découpage en blocs proposé).
- L'évaluation porte sur le temps de calcul ; la qualité des prédictions n'est pas encore comparée à d'autres classifieurs (k-NN, SVM).
- Les expériences utilisent des données 2D synthétiques ; des jeux de données réels ou bruités sont une suite naturelle.
- Apprendre les mesures de similarité à partir des données.

## Mises à jour

- **Novembre 2026 :** correction de bugs dans le notebook, ajout de vérifications de concordance entre les trois versions, traduction du notebook en anglais et nouvelle exécution de toutes les mesures. Les temps ci-dessus proviennent de cette version ; le rapport présente les expériences d'origine de mai 2025.
- **Octobre 2026 :** traduction du rapport en anglais.
