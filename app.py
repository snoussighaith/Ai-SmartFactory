"""
SmartFactory AI : dashboard de maintenance prédictive.

Lancer depuis le dossier du projet :
    streamlit run app.py

Prérequis : avoir exécuté les notebooks 01 à 04 (fichiers dans data/ et models/).
"""
import warnings
import sys
from pathlib import Path

# Correctif : relecture des modèles HistGradientBoosting (module Cython "_loss")
try:
    import sklearn._loss._loss as _cy_loss
    sys.modules.setdefault("_loss", _cy_loss)
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SmartFactory AI", page_icon="🏭", layout="wide")

BASE = Path(__file__).resolve().parent
DATA, MODELS, FIG = BASE / "data", BASE / "models", BASE / "figures"
REPORT = BASE / "report"

COLORS = {"Normal": "#4C78A8", "Panne": "#E45756"}
RAW_COLS = ["Type", "Air_temperature", "Process_temperature", "Rotational_speed", "Torque", "Tool_wear"]
NUM_COLS = ["Air_temperature", "Process_temperature", "Rotational_speed", "Torque", "Tool_wear",
            "Temp_diff", "Power", "Torque_x_Wear"]

REQUIRED_FILES = {
    DATA: ["ai4i_clean.csv", "train.csv", "test.csv", "train_pca.csv", "test_pca.csv",
           "resultats_classification.csv"],
    MODELS: ["scaler.pkl", "feature_cols.pkl", "pca.pkl", "kmeans.pkl", "kmeans_meta.pkl",
             "models_original.pkl", "models_pca.pkl", "rf_tuned.pkl", "rf_tuned_meta.pkl"],
}


# --------------------------------------------------------------------------------------
# Utilitaires d'affichage (compatibles avec plusieurs versions de Streamlit)
# --------------------------------------------------------------------------------------
def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


# --------------------------------------------------------------------------------------
# Vérification des fichiers et chargement
# --------------------------------------------------------------------------------------
missing = [f"{folder.name}/{f}" for folder, files in REQUIRED_FILES.items()
           for f in files if not (folder / f).exists()]
if missing:
    st.error("Fichiers manquants : exécute d'abord les notebooks 01 à 04 (dans cet ordre).")
    st.code("\n".join(missing))
    st.caption(f"Dossier du projet détecté : {BASE}")
    st.stop()


@st.cache_resource
def load_artifacts():
    return {
        "scaler": joblib.load(MODELS / "scaler.pkl"),
        "feature_cols": joblib.load(MODELS / "feature_cols.pkl"),
        "pca": joblib.load(MODELS / "pca.pkl"),
        "kmeans": joblib.load(MODELS / "kmeans.pkl"),
        "km_meta": joblib.load(MODELS / "kmeans_meta.pkl"),
        "models_orig": joblib.load(MODELS / "models_original.pkl"),
        "models_pca": joblib.load(MODELS / "models_pca.pkl"),
        "rf": joblib.load(MODELS / "rf_tuned.pkl"),
        "rf_meta": joblib.load(MODELS / "rf_tuned_meta.pkl"),
    }


@st.cache_data
def load_tables():
    return {
        "clean": pd.read_csv(DATA / "ai4i_clean.csv"),
        "train": pd.read_csv(DATA / "train.csv"),
        "test": pd.read_csv(DATA / "test.csv"),
        "test_pca": pd.read_csv(DATA / "test_pca.csv"),
        "results": pd.read_csv(DATA / "resultats_classification.csv"),
    }


art = load_artifacts()
tab = load_tables()

scaler, feature_cols, pca = art["scaler"], art["feature_cols"], art["pca"]
kmeans, km_cols = art["kmeans"], art["km_meta"]["km_cols"]
rf, rf_thr = art["rf"], float(art["rf_meta"]["threshold"])
models_orig = {**art["models_orig"], "Random Forest optimisé": rf}
models_pca = art["models_pca"]
pc_cols = [f"PC{i + 1}" for i in range(pca.n_components_)]


def clean_columns(df):
    df = df.copy()
    df.columns = (df.columns.str.replace(r"\[.*?\]", "", regex=True)
                  .str.strip().str.replace(" ", "_"))
    return df


def add_features(df):
    df = df.copy()
    df["Temp_diff"] = df["Process_temperature"] - df["Air_temperature"]
    df["Power"] = df["Torque"] * df["Rotational_speed"] * 2 * np.pi / 60
    df["Torque_x_Wear"] = df["Torque"] * df["Tool_wear"]
    df["Type_enc"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
    return df


def scale(df):
    return pd.DataFrame(scaler.transform(df[feature_cols]), columns=feature_cols, index=df.index)


def pc_frame(Xs):
    return pd.DataFrame(pca.transform(Xs), columns=pc_cols, index=Xs.index)


@st.cache_data(show_spinner=False)
def compute(df):
    """Projections, clusters et distances pour le jeu de données actif."""
    Xs = scale(df)
    Z = pc_frame(Xs)
    return Z, kmeans.predict(Xs[km_cols]), kmeans.transform(Xs[km_cols]).min(axis=1)


# --------------------------------------------------------------------------------------
# Barre latérale : navigation et chargement des données
# --------------------------------------------------------------------------------------
st.sidebar.title("🏭 SmartFactory AI")
page = st.sidebar.radio("Navigation", ["Accueil", "Capteurs", "PCA", "K-Means", "Modèles",
                                       "Prédiction", "Interprétation", "Rapport LaTeX"])

st.sidebar.divider()
uploaded = st.sidebar.file_uploader("Charger les données d'une machine (CSV)", type="csv",
                                    help="Format AI4I 2020 : colonnes Type, Air temperature [K], "
                                         "Process temperature [K], Rotational speed [rpm], "
                                         "Torque [Nm], Tool wear [min]. Machine failure est facultative.")

df, source = tab["clean"], "jeu de données AI4I 2020 (10 000 machines)"
if uploaded is not None:
    try:
        raw = clean_columns(pd.read_csv(uploaded))
        absent = [c for c in RAW_COLS if c not in raw.columns]
        if absent:
            st.sidebar.error("Colonnes manquantes : " + ", ".join(absent))
        elif raw["Type"].isin(["L", "M", "H"]).mean() < 1:
            st.sidebar.error("La colonne Type doit contenir uniquement L, M ou H.")
        else:
            df, source = add_features(raw).reset_index(drop=True), f"fichier chargé ({len(raw)} lignes)"
    except Exception as exc:  # fichier illisible
        st.sidebar.error(f"Lecture impossible : {exc}")

has_target = "Machine_failure" in df.columns
if has_target:
    df = df.assign(État=np.where(df["Machine_failure"] == 1, "Panne", "Normal"))
Z, labels, dist = compute(df)
st.sidebar.caption(f"Données actives : {source}")


# --------------------------------------------------------------------------------------
# Page Accueil
# --------------------------------------------------------------------------------------
if page == "Accueil":
    st.title("SmartFactory AI")
    st.subheader("Détection d'anomalies et prédiction des pannes industrielles")
    st.write("Prototype de maintenance prédictive : exploration des capteurs, réduction de "
             "dimension (PCA), groupes de comportements (K-Means), classification de l'état de "
             "la machine et interprétation des facteurs de panne.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Machines", f"{len(df):,}".replace(",", " "))
    if has_target:
        c2.metric("Pannes", int(df["Machine_failure"].sum()))
        c3.metric("Taux de pannes", f"{df['Machine_failure'].mean() * 100:.2f} %")
    else:
        c2.metric("Pannes", "n/a")
        c3.metric("Taux de pannes", "n/a")
    c4.metric("Variables d'entrée", len(feature_cols))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Aperçu des données**")
        show_df(df.drop(columns=["État"], errors="ignore").head(20))
    with right:
        if has_target:
            st.markdown("**Répartition de l'état des machines**")
            counts = df["État"].value_counts().reset_index()
            counts.columns = ["État", "Nombre"]
            show(px.pie(counts, names="État", values="Nombre", color="État",
                        color_discrete_map=COLORS, hole=0.5))
            st.caption("Les pannes sont rares : l'accuracy serait trompeuse. Les modèles sont "
                       "jugés sur le Recall, le F1-score et la PR-AUC.")
        else:
            st.info("Le fichier chargé ne contient pas la colonne Machine failure.")

    with st.expander("Pipeline et limites"):
        st.markdown(
            "1. **Prétraitement** : nettoyage, outliers conservés, 3 variables construites "
            "(`Temp_diff`, `Power`, `Torque_x_Wear`), split stratifié, normalisation.\n"
            "2. **PCA** : réduction à quelques composantes.\n"
            "3. **K-Means** : groupes de comportements et détection d'anomalies.\n"
            "4. **Classification** : 6 modèles, avec et sans PCA.\n"
            "5. **Interprétation** : importance des variables.\n\n"
            "**Limites :** données synthétiques (AI4I 2020), pas de dimension temporelle, "
            "capteurs limités à température, vitesse, couple et usure.")


# --------------------------------------------------------------------------------------
# Page Capteurs
# --------------------------------------------------------------------------------------
elif page == "Capteurs":
    st.title("Paramètres des capteurs")
    types = st.multiselect("Type de produit", ["L", "M", "H"], default=["L", "M", "H"])
    view = df[df["Type"].isin(types)] if types else df.iloc[0:0]
    if view.empty:
        st.warning("Aucune machine pour ce filtre.")
        st.stop()
    st.caption(f"{len(view)} machines sélectionnées")

    t1, t2, t3, t4, t5 = st.tabs(["Distribution", "Boxplot", "Nuage de points", "Corrélations",
                                  "Statistiques"])
    with t1:
        var = st.selectbox("Variable", NUM_COLS, key="hist_var")
        fig = px.histogram(view, x=var, color="État" if has_target else None, nbins=50,
                           barmode="overlay", opacity=0.65, histnorm="probability density",
                           color_discrete_map=COLORS)
        show(fig)
        st.caption("Chaque classe est normalisée séparément : on compare les formes, pas les effectifs.")
    with t2:
        var = st.selectbox("Variable", NUM_COLS, key="box_var")
        show(px.box(view, y=var, x="État" if has_target else None,
                    color="État" if has_target else None, color_discrete_map=COLORS))
    with t3:
        a, b = st.columns(2)
        x = a.selectbox("Axe X", NUM_COLS, index=3)
        y = b.selectbox("Axe Y", NUM_COLS, index=4)
        show(px.scatter(view, x=x, y=y, color="État" if has_target else None,
                        color_discrete_map=COLORS, opacity=0.6))
    with t4:
        cols = NUM_COLS + (["Machine_failure"] if has_target else [])
        show(px.imshow(view[cols].corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                       zmin=-1, zmax=1, aspect="auto"))
        st.caption("La corrélation ne mesure que les liens linéaires : une variable peut être "
                   "utile sans être corrélée à la panne (relation en U, seuils).")
    with t5:
        show_df(view[NUM_COLS].describe().T.round(2))


# --------------------------------------------------------------------------------------
# Page PCA
# --------------------------------------------------------------------------------------
elif page == "PCA":
    st.title("Analyse en composantes principales")
    var = pca.explained_variance_ratio_ * 100
    cum = np.cumsum(var)

    st.markdown("**Variance expliquée**")
    v1, v2 = st.columns(2)
    with v1:
        fig = px.bar(x=pc_cols, y=var, title="Par composante (scree plot)",
                     labels={"x": "Composante", "y": "Variance expliquée (%)"})
        show(fig)
    with v2:
        fig = px.line(x=pc_cols, y=cum, markers=True, title="Variance cumulée",
                      labels={"x": "Nombre de composantes", "y": "Variance cumulée (%)"})
        fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="95 %")
        fig.update_yaxes(range=[0, 105])
        show(fig)
    st.caption(f"{len(pc_cols)} composantes conservent {cum[-1]:.1f} % de l'information.")

    st.markdown("**Projection des machines**")
    a, b, c = st.columns(3)
    px_axis = a.selectbox("Axe X", pc_cols, index=0)
    py_axis = b.selectbox("Axe Y", pc_cols, index=1)
    color_by = c.selectbox("Couleur", (["État"] if has_target else []) + ["Cluster K-Means", "Type"])
    plot = Z.copy()
    plot["État"] = df["État"].values if has_target else "n/a"
    plot["Cluster K-Means"] = labels.astype(str)
    plot["Type"] = df["Type"].values
    show(px.scatter(plot, x=px_axis, y=py_axis, color=color_by, opacity=0.55,
                    color_discrete_map=COLORS if color_by == "État" else None))

    if st.checkbox("Afficher la projection 3D") and len(pc_cols) >= 3:
        sample = plot.sample(min(5000, len(plot)), random_state=0)
        show(px.scatter_3d(sample, x="PC1", y="PC2", z="PC3", color=color_by, opacity=0.6,
                           color_discrete_map=COLORS if color_by == "État" else None))

    st.markdown("**Loadings : de quoi chaque composante est faite**")
    loadings = pd.DataFrame(pca.components_.T, index=feature_cols, columns=pc_cols)
    show(px.imshow(loadings, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                   aspect="auto"))
    st.caption("Une valeur élevée (en valeur absolue) signifie que la variable pèse beaucoup dans la composante.")


# --------------------------------------------------------------------------------------
# Page K-Means
# --------------------------------------------------------------------------------------
elif page == "K-Means":
    st.title("Clusters K-Means et anomalies")
    k = kmeans.n_clusters
    st.caption(f"K-Means entraîné avec k = {k} sur : {', '.join(km_cols)}")

    summary = pd.DataFrame({"Cluster": labels}).groupby("Cluster").size().rename("Machines").to_frame()
    summary["Part (%)"] = (summary["Machines"] / len(df) * 100).round(1)
    if has_target:
        g = pd.DataFrame({"Cluster": labels, "p": df["Machine_failure"].values}).groupby("Cluster")["p"]
        summary["Pannes"] = g.sum().astype(int)
        summary["Taux de pannes (%)"] = (g.mean() * 100).round(2)

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Résumé par cluster**")
        show_df(summary)
        if has_target:
            fig = px.bar(summary.reset_index(), x="Cluster", y="Taux de pannes (%)")
            fig.add_hline(y=df["Machine_failure"].mean() * 100, line_dash="dash", line_color="red",
                          annotation_text="Moyenne globale")
            show(fig)
    with c2:
        st.markdown("**Profil des clusters** (valeurs normalisées : +1 = un écart-type au-dessus de la moyenne)")
        Xs = scale(df)
        profile = Xs[km_cols].assign(Cluster=labels).groupby("Cluster").mean()
        show(px.imshow(profile, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-2, zmax=2,
                       aspect="auto"))

    st.markdown("**Clusters dans le plan PCA**")
    plot = Z[["PC1", "PC2"]].assign(Cluster=labels.astype(str))
    show(px.scatter(plot, x="PC1", y="PC2", color="Cluster", opacity=0.55))

    st.markdown("**Détection d'anomalies par distance au centroïde**")
    q = st.slider("Seuil (percentile des distances du jeu d'entraînement)", 90.0, 99.9, 95.0, 0.1)
    d_train = kmeans.transform(tab["train"][km_cols]).min(axis=1)
    thr = float(np.percentile(d_train, q))
    flagged = dist > thr
    m1, m2 = st.columns(2)
    m1.metric("Machines signalées", int(flagged.sum()))
    m2.metric("Seuil de distance", f"{thr:.2f}")
    if has_target:
        y = df["Machine_failure"].values
        tp = int(((flagged) & (y == 1)).sum())
        prec = tp / flagged.sum() if flagged.sum() else 0
        rec = tp / y.sum() if y.sum() else 0
        m3, m4, m5 = st.columns(3)
        m3.metric("Precision", f"{prec * 100:.1f} %")
        m4.metric("Recall", f"{rec * 100:.1f} %")
        m5.metric("Taux de base", f"{y.mean() * 100:.1f} %")
        st.caption("Si la precision dépasse nettement le taux de base, la distance au centroïde apporte de l'information.")
    fig = px.histogram(x=dist, nbins=60, labels={"x": "Distance au centroïde"})
    fig.add_vline(x=thr, line_dash="dash", line_color="red")
    show(fig)


# --------------------------------------------------------------------------------------
# Page Modèles
# --------------------------------------------------------------------------------------
elif page == "Modèles":
    st.title("Comparaison des modèles")
    res = tab["results"]
    st.markdown("**Résultats sur le jeu de test**")
    cols = ["Modèle", "Variante", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC",
            "CV F1", "CV PR-AUC"]
    show_df(res[cols].round(3).sort_values(["Variante", "F1"], ascending=[True, False]))

    metric = st.selectbox("Métrique à comparer", ["F1", "Recall", "Precision", "PR-AUC", "ROC-AUC", "Accuracy"])
    show(px.bar(res, x="Modèle", y=metric, color="Variante", barmode="group"))
    st.caption("Variables originales contre composantes PCA. Pour des pannes rares, privilégie Recall, F1 et PR-AUC.")

    st.divider()
    st.markdown("**Analyse détaillée d'un modèle (jeu de test)**")
    a, b = st.columns(2)
    variant = a.radio("Variables", ["Originales", "PCA"], horizontal=True)
    names = list(models_orig) if variant == "Originales" else list(models_pca)
    name = b.selectbox("Modèle", names, index=len(names) - 1 if variant == "Originales" else 0)

    y_test = tab["test"]["target"]
    if variant == "Originales":
        Xt = tab["test"].drop(columns="target")
        model = models_orig[name]
    else:
        Xt = tab["test_pca"].drop(columns="target")
        model = models_pca[name]
    score = model.predict_proba(Xt)[:, 1]

    default_thr = rf_thr if name == "Random Forest optimisé" else 0.5
    thr = st.slider("Seuil de décision", 0.05, 0.95, float(round(default_thr, 2)), 0.01)
    pred = (score >= thr).astype(int)

    m = st.columns(6)
    m[0].metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
    m[1].metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
    m[2].metric("Recall", f"{recall_score(y_test, pred):.3f}")
    m[3].metric("F1", f"{f1_score(y_test, pred):.3f}")
    m[4].metric("ROC-AUC", f"{roc_auc_score(y_test, score):.3f}")
    m[5].metric("PR-AUC", f"{average_precision_score(y_test, score):.3f}")

    c1, c2, c3 = st.columns(3)
    with c1:
        cm = confusion_matrix(y_test, pred)
        show(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="equal",
                       x=["Prédit : normal", "Prédit : panne"], y=["Réel : normal", "Réel : panne"],
                       title="Matrice de confusion"))
    with c2:
        fpr, tpr, _ = roc_curve(y_test, score)
        fig = px.area(x=fpr, y=tpr, labels={"x": "Taux de faux positifs", "y": "Recall"}, title="Courbe ROC")
        fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
        show(fig)
    with c3:
        p, r, _ = precision_recall_curve(y_test, score)
        fig = px.area(x=r, y=p, labels={"x": "Recall", "y": "Precision"}, title="Courbe Precision-Recall")
        fig.add_hline(y=float(y_test.mean()), line_dash="dash")
        show(fig)


# --------------------------------------------------------------------------------------
# Page Prédiction
# --------------------------------------------------------------------------------------
elif page == "Prédiction":
    st.title("Prédire l'état d'une machine")

    PRESETS = {
        "Saisie manuelle": None,
        "Machine normale": dict(t="M", air=299.0, proc=309.2, speed=1500, torque=40.0, wear=90),
        "Outil très usé (type TWF)": dict(t="L", air=299.0, proc=309.2, speed=1500, torque=38.0, wear=215),
        "Mauvais refroidissement (type HDF)": dict(t="M", air=302.0, proc=310.3, speed=1350, torque=45.0, wear=60),
        "Puissance anormale (type PWF)": dict(t="M", air=299.0, proc=309.2, speed=1400, torque=70.0, wear=30),
        "Surcharge (type OSF)": dict(t="L", air=299.0, proc=309.2, speed=1300, torque=62.0, wear=190),
    }
    defaults = dict(t="M", air=300.0, proc=310.0, speed=1500, torque=40.0, wear=100)
    for key, val in defaults.items():
        st.session_state.setdefault(f"in_{key}", val)

    def apply_preset():
        p = PRESETS[st.session_state["preset"]]
        if p:
            for key, val in p.items():
                st.session_state[f"in_{key}"] = val

    left, right = st.columns([1, 1])
    with left:
        st.selectbox("Scénario", list(PRESETS), key="preset", on_change=apply_preset)
        if st.session_state.get("preset") == "Outil très usé (type TWF)":
            st.caption("Dans le dataset, la panne TWF survient à un instant aléatoire entre 200 et "
                       "240 min d'usure : même avec un outil très usé, la probabilité prédite peut "
                       "rester modérée. C'est une limite de prédictibilité, pas un bug.")
        typ = st.selectbox("Type de produit", ["L", "M", "H"], key="in_t")
        air = st.slider("Température de l'air [K]", 295.0, 305.0, key="in_air", step=0.1)
        proc = st.slider("Température du processus [K]", 305.0, 314.5, key="in_proc", step=0.1)
        speed = st.slider("Vitesse de rotation [rpm]", 1100, 3000, key="in_speed", step=10)
        torque = st.slider("Couple [Nm]", 3.0, 80.0, key="in_torque", step=0.1)
        wear = st.slider("Usure de l'outil [min]", 0, 260, key="in_wear", step=1)

        st.divider()
        variant = st.radio("Variables du modèle", ["Originales", "PCA"], horizontal=True)
        names = list(models_orig) if variant == "Originales" else list(models_pca)
        name = st.selectbox("Modèle", names, index=len(names) - 1 if variant == "Originales" else 0)
        default_thr = rf_thr if name == "Random Forest optimisé" else 0.5
        thr = st.slider("Seuil d'alerte", 0.05, 0.95, float(round(default_thr, 2)), 0.01)

    row = add_features(pd.DataFrame([dict(Type=typ, Air_temperature=air, Process_temperature=proc,
                                          Rotational_speed=speed, Torque=torque, Tool_wear=wear)]))
    Xs_row = scale(row)

    def proba_of(model_name, var):
        if var == "PCA":
            return float(models_pca[model_name].predict_proba(pc_frame(Xs_row))[:, 1][0])
        return float(models_orig[model_name].predict_proba(Xs_row)[:, 1][0])

    p = proba_of(name, variant)

    with right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": " %"},
            title={"text": "Probabilité de panne"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#E45756" if p >= thr else "#4C78A8"},
                   "threshold": {"line": {"color": "black", "width": 3}, "value": thr * 100}}))
        fig.update_layout(height=280, margin=dict(t=60, b=10))
        show(fig)
        if p >= thr:
            st.error(f"⚠️ Risque de panne détecté (probabilité {p * 100:.1f} % ≥ seuil {thr * 100:.0f} %). "
                     "Inspection recommandée.")
        else:
            st.success(f"✅ Fonctionnement normal prévu (probabilité {p * 100:.1f} % < seuil {thr * 100:.0f} %).")

        st.markdown("**Position de la machine**")
        lab = int(kmeans.predict(Xs_row[km_cols])[0])
        d = float(kmeans.transform(Xs_row[km_cols]).min(axis=1)[0])
        thr95 = float(art["km_meta"].get("threshold_95", np.inf))
        c1, c2, c3 = st.columns(3)
        c1.metric("Cluster K-Means", lab)
        c2.metric("Distance au centroïde", f"{d:.2f}")
        c3.metric("Atypique (> 95e percentile)", "Oui" if d > thr95 else "Non")

    st.divider()
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Variables calculées**")
        show_df(row[["Temp_diff", "Power", "Torque_x_Wear"]].round(2).rename(
            columns={"Temp_diff": "Écart de température [K]", "Power": "Puissance [W]",
                     "Torque_x_Wear": "Couple × usure"}))
        st.markdown("**Tous les modèles (variables originales)**")
        allp = pd.DataFrame({"Modèle": list(models_orig), "Probabilité de panne": [
            proba_of(n, "Originales") for n in models_orig]})
        fig = px.bar(allp, x="Probabilité de panne", y="Modèle", orientation="h", range_x=[0, 1])
        fig.add_vline(x=thr, line_dash="dash")
        show(fig)
    with cB:
        st.markdown("**Repères physiques du dataset AI4I 2020**")
        td, pw, tw = float(row["Temp_diff"][0]), float(row["Power"][0]), float(row["Torque_x_Wear"][0])
        osf_thr = {"L": 11000, "M": 12000, "H": 13000}[typ]
        rules = pd.DataFrame([
            ["Usure d'outil (TWF)", "Usure entre 200 et 240 min", 200 <= wear <= 240],
            ["Dissipation thermique (HDF)", "Écart < 8,6 K et vitesse < 1380 rpm", td < 8.6 and speed < 1380],
            ["Puissance (PWF)", "Puissance < 3500 W ou > 9000 W", pw < 3500 or pw > 9000],
            [f"Surcharge (OSF, type {typ})", f"Couple × usure > {osf_thr}", tw > osf_thr],
        ], columns=["Mode de panne", "Règle documentée", "Zone à risque"])
        rules["Zone à risque"] = rules["Zone à risque"].map({True: "🔴 Oui", False: "🟢 Non"})
        show_df(rules)
        st.caption("Règles du dataset simulé, données à titre pédagogique. Les modèles ne les "
                   "utilisent pas directement : ils les apprennent à partir des mesures.")


# --------------------------------------------------------------------------------------
# Page Interprétation
# --------------------------------------------------------------------------------------
elif page == "Interprétation":
    st.title("Variables les plus influentes")
    st.caption("Modèle : Random Forest optimisé (variables originales)")

    mdi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    st.markdown("**Importance par impureté**")
    show(px.bar(mdi, orientation="h", labels={"value": "Importance", "index": "Variable"}))
    st.caption("Quand deux variables sont redondantes (Torque et Power), l'importance se partage "
               "entre elles : chacune paraît moins importante qu'elle ne l'est.")

    @st.cache_data(show_spinner="Calcul de l'importance par permutation...")
    def perm_table():
        X, y = tab["test"].drop(columns="target"), tab["test"]["target"]
        r = permutation_importance(rf, X, y, scoring="average_precision", n_repeats=5,
                                   random_state=42, n_jobs=-1)
        return pd.DataFrame({"Baisse de PR-AUC": r.importances_mean, "Écart-type": r.importances_std},
                            index=feature_cols).sort_values("Baisse de PR-AUC")

    st.markdown("**Importance par permutation (jeu de test)**")
    if st.button("Calculer l'importance par permutation"):
        perm = perm_table()
        show(px.bar(perm, y=perm.index, x="Baisse de PR-AUC", orientation="h", error_x="Écart-type"))
        st.caption("On mélange une variable et on mesure la perte de performance : plus elle est grande, "
                   "plus le modèle s'appuie sur cette variable.")

    group_file = DATA / "importance_groupes.csv"
    if group_file.exists():
        st.markdown("**Importance par groupes de variables** (notebook 06)")
        gi = pd.read_csv(group_file, index_col=0)
        show(px.bar(gi.sort_values("Baisse PR-AUC"), x="Baisse PR-AUC", orientation="h"))

    images = [("dependance_partielle.png", "Dépendance partielle"),
              ("shap_summary.png", "SHAP : contribution des variables"),
              ("arbre_regles.png", "Arbre de décision simple (règles lisibles)")]
    present = [(f, t) for f, t in images if (FIG / f).exists()]
    if present:
        st.markdown("**Analyses du notebook 06**")
        for f, t in present:
            st.image(str(FIG / f), caption=t)
    else:
        st.info("Exécute le notebook 06_interpretation.ipynb pour ajouter les analyses SHAP, "
                "la dépendance partielle et l'arbre de règles.")