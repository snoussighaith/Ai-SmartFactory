# Audit du projet AI-SmartFactory

## 1. Périmètre et méthode

L'audit a porté sur les six notebooks Jupyter, les scripts Python, le fichier de dépendances, les données exportées, les modèles sérialisés, les figures disponibles et le dashboard Streamlit. Les résultats décrits dans le rapport proviennent des fichiers présents dans le dépôt au 15 novembre 2026.

Les contrôles ont combiné la lecture des cellules et sorties des notebooks, la
lecture du code source, l'inspection des dimensions et colonnes des CSV, ainsi
que la lecture des métadonnées des modèles sauvegardés. Aucun notebook, dataset
original ou fichier `app.py` n'a été modifié.

## 2. Architecture observée

| Composant | Rôle |
|---|---|
| `01_pretraitement_eda.ipynb` | Nettoyage, contrôle qualité, feature engineering et analyse exploratoire |
| `02_pca.ipynb` | Standardisation, PCA, variance expliquée, projections et sauvegarde de la PCA |
| `03_kmeans.ipynb` | Choix de `k`, clustering, profils de groupes et détection par distance |
| `04_classification.ipynb` | Entraînement des six modèles, comparaison originale/PCA et optimisation du Random Forest |
| `05_evaluation.ipynb` | Évaluation finale, matrices de confusion, seuils, intervalles et modes de panne |
| `06_interpretation.ipynb` | Importance MDI, permutation, groupes de variables et arbre de règles |
| `app.py` | Dashboard Streamlit connecté aux CSV, modèles et figures sauvegardés |
| `export_figures.py` | Export des figures produites par les notebooks |
| `data/` | Données nettoyées, jeux train/test et tableaux d'évaluation |
| `models/` | Scaler, PCA, K-Means, modèles de classification et métadonnées |
| `figures/` | 11 figures exportées pour l'analyse et le rapport |

## 3. Méthodes réellement implémentées

- Encodage ordinal de `Type` : L=0, M=1, H=2.
- Création de `Temp_diff`, `Power` et `Torque_x_Wear`.
- Exclusion des identifiants et des cinq colonnes de modes de panne afin d'éviter la fuite de cible.
- Split stratifié 80/20 et `StandardScaler` ajusté sur l'apprentissage.
- PCA sur les variables standardisées, avec cinq composantes conservant 97,17 % de la variance.
- K-Means testé pour `k` de 2 à 10 ; choix de `k=4` selon la silhouette.
- Détection d'anomalies par distance au centroïde avec seuils aux percentiles 90, 95, 97 et 99.
- Six classifieurs : régression logistique, KNN, SVM RBF, arbre de décision, Random Forest et HistGradientBoosting.
- Validation croisée stratifiée à cinq plis et recherche aléatoire de 20 configurations pour le Random Forest.
- Évaluation par accuracy, precision, recall, F1, ROC-AUC et PR-AUC.
- Interprétation par MDI, permutation, groupes de variables et arbre de règles.

Les artefacts confirment cinq composantes PCA, avec les variances expliquées
33,44 %, 25,15 %, 19,29 %, 11,10 % et 8,19 %, soit 97,17 % cumulés. Le
Random Forest optimisé est sauvegardé avec `n_estimators=200`,
`min_samples_leaf=2`, `max_features=0,5`, `max_depth=10` et un seuil de
0,50257 (affiché 0,503).

## 4. Résultats vérifiés

- Dataset : 10 000 observations, 339 pannes et 9 661 observations normales.
- Train/test : 8 000/2 000 observations ; 271/68 pannes.
- PCA : 5 composantes, 97,17 % de variance conservée.
- K-Means : 4 clusters, silhouette train = 0,2138.
- Cluster le plus exposé : cluster 0, taux test = 6,24 %.
- Détection par distance au percentile 99 : precision = 71,43 %, recall = 14,71 %.
- Meilleur compromis supervisé : Random Forest optimisé, seuil = 0,503, precision = 0,903, recall = 0,824, F1 = 0,862, PR-AUC = 0,882.
- Les performances de tous les modèles diminuent avec la PCA.

## 5. Fonctionnalités réelles du dashboard

1. **Accueil** : KPI, aperçu du jeu actif, répartition Normal/Panne et rappel du pipeline.
2. **Capteurs** : filtre par type, histogrammes, boxplots, nuages de points, corrélations et statistiques.
3. **PCA** : scree plot, variance cumulée, projections 2D/3D et heatmap des loadings.
4. **K-Means** : résumé des clusters, profils normalisés, projection PCA, seuil d'anomalie interactif et métriques.
5. **Modèles** : tableau comparatif, métrique sélectionnable, matrice de confusion, courbes ROC et Precision-Recall.
6. **Prédiction** : saisie ou scénarios prédéfinis, probabilités par modèle, seuil d'alerte, cluster et règles physiques.
7. **Interprétation** : importance MDI, permutation à la demande, importance par groupes et figures disponibles.

Le chargement d'un CSV est prévu. Les colonnes de capteurs sont vérifiées et la cible est facultative.

## 6. Incohérences et limites détectées

- Le dataset est synthétique et ne représente pas une installation réelle.
- Les colonnes de modes de panne sont disponibles dans les données mais volontairement exclues des variables explicatives.
- Le fichier de figures ne contient pas `shap_summary.png` ; le dashboard affiche donc un message si cette figure est absente.
- Aucune capture d'écran du dashboard n'est fournie dans le dépôt ; le rapport décrit le code sans présenter de capture inventée.
- Le dashboard charge des artefacts sérialisés et dépend de la compatibilité des versions Python/scikit-learn.
- La cible est binaire et les modes TWF/HDF/PWF/OSF/RNF ne sont pas prédits comme classes indépendantes.
- Les intervalles d'évaluation sont exportés, mais aucune calibration probabiliste ni analyse temporelle n'est implémentée.
- Le fichier de dépendances ne fige pas les versions ; la relecture des modèles
  sérialisés peut donc produire des avertissements de compatibilité.

## 7. Informations manquantes

Le nom de l'encadrant et les éventuelles captures institutionnelles ne sont pas présents dans le dépôt. La page de garde utilise donc un champ « Non renseigné » à compléter avant dépôt.
