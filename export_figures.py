"""
Exporte dans figures/ les graphiques du rapport (EDA, PCA, K-Means).
Les figures de l'évaluation et de l'interprétation sont déjà créées par les notebooks 05 et 06.

Lancer depuis le dossier du projet :  python export_figures.py
Prérequis : notebooks 01 à 03 exécutés.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid")
BASE = Path(__file__).resolve().parent
DATA, MODELS, FIG = BASE / "data", BASE / "models", BASE / "figures"
FIG.mkdir(exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close("all")
    print("  ->", name)


df = pd.read_csv(DATA / "ai4i_clean.csv")
train = pd.read_csv(DATA / "train.csv")
feature_cols = joblib.load(MODELS / "feature_cols.pkl")
pca = joblib.load(MODELS / "pca.pkl")
kmeans = joblib.load(MODELS / "kmeans.pkl")
km_cols = joblib.load(MODELS / "kmeans_meta.pkl")["km_cols"]

num = ["Air_temperature", "Process_temperature", "Rotational_speed", "Torque", "Tool_wear"]
eda = num + ["Temp_diff", "Power", "Torque_x_Wear"]

print("EDA")
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Machine_failure")
plt.title("Répartition de Machine_failure")
save("eda_cible.png")

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, c in zip(axes, num):
    sns.boxplot(y=df[c], ax=ax); ax.set_title(c)
save("eda_boxplots.png")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, c in zip(axes.ravel(), eda):
    sns.histplot(data=df, x=c, hue="Machine_failure", stat="density", common_norm=False, bins=40, ax=ax)
    ax.set_title(c)
save("eda_histogrammes_etat.png")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, c in zip(axes.ravel(), eda):
    sns.boxplot(data=df, x="Machine_failure", y=c, ax=ax); ax.set_title(c)
save("eda_boxplots_etat.png")

plt.figure(figsize=(9, 7))
sns.heatmap(df[eda + ["Machine_failure"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matrice de corrélation")
save("eda_correlation.png")

modes = [m for m in ["TWF", "HDF", "PWF", "OSF"] if m in df.columns]
if modes:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, c in zip(axes, ["Torque", "Tool_wear", "Temp_diff"]):
        for m in modes:
            if df[m].sum() > 1:
                sns.kdeplot(df.loc[df[m] == 1, c], label=m, ax=ax, warn_singular=False)
        ax.set_title(c + " selon le mode de panne"); ax.legend()
    save("eda_kde_modes.png")

print("PCA")
var = pca.explained_variance_ratio_ * 100
cum = np.cumsum(var)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].bar(range(1, len(var) + 1), var); axes[0].set(title="Scree plot", xlabel="Composante", ylabel="Variance expliquée (%)")
axes[1].plot(range(1, len(cum) + 1), cum, marker="o")
axes[1].axhline(95, color="red", linestyle="--", label="95 %"); axes[1].legend()
axes[1].set(title="Variance cumulée", xlabel="Nombre de composantes", ylabel="Variance cumulée (%)")
save("pca_variance.png")

loadings = pd.DataFrame(pca.components_.T, index=feature_cols, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
plt.figure(figsize=(10, 6))
sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Loadings des composantes")
save("pca_loadings.png")

X, y = train[feature_cols], train["target"].values
Z = pca.transform(X)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].scatter(Z[y == 0, 0], Z[y == 0, 1], s=6, alpha=0.3, label="Normal")
axes[0].scatter(Z[y == 1, 0], Z[y == 1, 1], s=18, color="red", alpha=0.9, label="Panne")
axes[0].legend(); axes[0].set_title("Projection PCA 2D (train)")
axes[1].scatter(Z[y == 1, 0], Z[y == 1, 1], s=18, color="red", alpha=0.8); axes[1].set_title("Pannes uniquement")
for ax in axes:
    ax.set_xlabel(f"PC1 ({var[0]:.1f} %)"); ax.set_ylabel(f"PC2 ({var[1]:.1f} %)")
save("pca_projection_2d.png")

fig, ax = plt.subplots(figsize=(7, 7))
for i, name in enumerate(feature_cols):
    ax.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1], head_width=0.02, color="tab:blue")
    ax.text(loadings.iloc[i, 0] * 1.08, loadings.iloc[i, 1] * 1.08, name, fontsize=9)
ax.add_patch(plt.Circle((0, 0), 1, fill=False, linestyle="--", color="gray"))
ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xlabel="PC1", ylabel="PC2", title="Contribution des variables (PC1 / PC2)")
ax.set_aspect("equal")
save("pca_cercle.png")

print("K-Means")
Xk = train[km_cols]
labels = kmeans.predict(Xk)
K = kmeans.n_clusters
profile = Xk.assign(cluster=labels).groupby("cluster").mean()
plt.figure(figsize=(10, 0.9 * K + 2))
sns.heatmap(profile, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Profil moyen des clusters (variables normalisées)")
save("kmeans_profil.png")

rate = pd.DataFrame({"c": labels, "p": y}).groupby("c")["p"].mean() * 100
plt.figure(figsize=(7, 4))
sns.barplot(x=rate.index, y=rate.values)
plt.axhline(y.mean() * 100, color="red", linestyle="--", label="Moyenne globale")
plt.xlabel("Cluster"); plt.ylabel("Taux de pannes (%)"); plt.legend(); plt.title("Taux de pannes par cluster (train)")
save("kmeans_taux_pannes.png")

pca2 = PCA(n_components=2, random_state=42).fit(Xk)
Z2 = pca2.transform(Xk)
plt.figure(figsize=(7, 5.5))
plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, cmap="tab10", s=6, alpha=0.5)
c2 = pca2.transform(pd.DataFrame(kmeans.cluster_centers_, columns=km_cols))
plt.scatter(c2[:, 0], c2[:, 1], c="black", marker="X", s=200, label="Centroïdes")
plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Clusters K-Means")
save("kmeans_clusters.png")

print("Terminé : figures dans", FIG)
