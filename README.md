# SmartFactory AI

Application de maintenance prédictive pour détecter les pannes de machines
industrielles à partir des données de capteurs **AI4I 2020**.

Le projet combine une analyse exploratoire, une réduction de dimension (PCA),
un clustering K-Means, plusieurs modèles de classification et un dashboard
interactif Streamlit.

## Fonctionnalités

- exploration des mesures de température, vitesse, couple et usure ;
- création de variables dérivées : `Temp_diff`, `Power` et `Torque_x_Wear` ;
- réduction de dimension par PCA ;
- segmentation des machines avec K-Means et détection d’anomalies ;
- comparaison de six modèles de classification ;
- interprétation des prédictions par importance des variables ;
- simulation d’une prédiction à partir de mesures saisies ou d’un fichier CSV.

Le dashboard propose les pages **Accueil**, **Capteurs**, **PCA**, **K-Means**,
**Modèles**, **Prédiction** et **Interprétation**.

## Installation

Depuis la racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Lancer le dashboard

Les notebooks doivent être exécutés dans l’ordre afin de produire les données
préparées et les modèles utilisés par l’application :

1. `01_pretraitement_eda.ipynb`
2. `02_pca.ipynb`
3. `03_kmeans.ipynb`
4. `04_classification.ipynb`
5. `05_evaluation.ipynb`
6. `06_interpretation.ipynb`

Puis lancer Streamlit :

```powershell
streamlit run app.py
```

L’application vérifie la présence des fichiers nécessaires dans `data/` et
`models/` au démarrage.

## Importer un fichier CSV

Le fichier importé doit contenir les colonnes suivantes :

- `Type` : `L`, `M` ou `H` ;
- `Air temperature [K]` ;
- `Process temperature [K]` ;
- `Rotational speed [rpm]` ;
- `Torque [Nm]` ;
- `Tool wear [min]`.

La colonne `Machine failure` est facultative. Lorsqu’elle est fournie, elle
permet d’afficher les métriques et la répartition des pannes.

## Structure du projet

```text
app.py                    Dashboard Streamlit
01_*.ipynb ... 06_*.ipynb Notebooks d’analyse et de modélisation
data/                     Données préparées et résultats exportés
models/                   Modèles et métadonnées sérialisés
figures/                  Figures utilisées par le dashboard et le rapport
export_figures.py         Régénération des figures
report/                   Rapport LaTeX et audit du projet
requirements.txt          Dépendances Python
```

Pour régénérer les figures produites par les premières analyses :

```powershell
python export_figures.py
```

## Résultats principaux

- 10 000 observations, dont 339 pannes ;
- séparation entraînement/test : 8 000 / 2 000 observations ;
- PCA conservant 97,17 % de la variance avec cinq composantes ;
- K-Means configuré avec quatre clusters ;
- Random Forest optimisé : précision 0,903, rappel 0,824 et F1-score 0,862
  sur le jeu de test selon l’audit du projet.

## Limites

Les données AI4I 2020 sont synthétiques et ne contiennent pas de dimension
temporelle. Le modèle prédit une panne binaire ; les modes de panne détaillés
ne sont pas prédits comme des classes indépendantes. Les fichiers `.pkl`
dépendent également de la compatibilité entre les versions de Python et de
scikit-learn utilisées lors de leur création.

Une description plus détaillée des méthodes et des résultats est disponible
dans [`report/PROJECT_AUDIT.md`](report/PROJECT_AUDIT.md).
