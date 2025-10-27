import os
import math
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import requests
import streamlit as st
import joblib

st.set_page_config(page_title="4 - Modélisation & Machine Learning", layout="wide")

# Global CSS helpers (text justification, etc.)
st.markdown(
    """
    <style>
    .justify { text-align: justify; white-space: pre-line;}
    </style>
    """,
    unsafe_allow_html=True,
)

def md_justify(text: str):
    """Render a markdown paragraph with justified alignment."""
    st.markdown(f'<div class="justify">{text}</div>', unsafe_allow_html=True)

# Config & aide (local, standalone depuis main app)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://crypto:crypto@localhost:5432/crypto_trading",
)
ALGO_DIR = Path(
    os.getenv("MODELS_DIR")
    or ("/app/algo_crypto" if Path("/app").exists() else (Path(__file__).resolve().parents[2] / "algo_crypto"))
)


def _api_base_url() -> str:
    """Base URL de l'API FastAPI.
    - En Docker, le service est accessible via http://api:8000
    - En local, on utilise http://localhost:8000
    """
    return os.getenv("API_URL", "http://api:8000" if Path("/app").exists() else "http://localhost:8000")


def api_predict_symbol(symbol: str) -> dict | None:
    """Appelle l'endpoint /predict/{symbol}. Retourne le JSON ou None si indisponible.
    On garde un time-out court et on ne fait pas échouer la page si l'API est down.
    """
    try:
        base = _api_base_url().rstrip("/")
        resp = requests.get(f"{base}/predict/{symbol}", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

@st.cache_data(ttl=300)
def list_symbols():
    try:
        engine = create_engine(DATABASE_URL, future=True)
        q = text("SELECT DISTINCT symbol FROM pair WHERE is_active = TRUE ORDER BY symbol")
        with engine.connect() as conn:
            df = pd.read_sql(q, conn)
        return df["symbol"].tolist()
    except Exception:
        return []

@st.cache_data(ttl=300)
def load_candles(symbol: str, years: int = 2):
    engine = create_engine(DATABASE_URL, future=True)
    # sécurisation: bornage années entre 1 et 10
    years = max(1, min(int(years), 10))
    q = text(
        f"""
        SELECT p.symbol,
               c.open_datetime AS timestamp,
               c.open_price, c.high_price, c.low_price, c.close_price,
               c.volume_base, c.volume_quote
        FROM candlestick c
        JOIN pair p ON p.id = c.pair_id
        WHERE p.symbol = :sym AND c.open_datetime >= NOW() - INTERVAL '{years} years'
        ORDER BY c.open_datetime ASC
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"sym": symbol})
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    price = df["close_price"]
    for k in range(1, 6):
        df[f"price_lag_{k}h"] = price.shift(k)
        df[f"volume_lag_{k}h"] = df["volume_base"].shift(k)
    df["rolling_mean_24h"] = price.rolling(24, min_periods=12).mean()
    df["rolling_mean_72h"] = price.rolling(72, min_periods=36).mean()
    delta = price.diff()
    gain = (delta.clip(lower=0)).rolling(14, min_periods=7).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_diff"] = macd - signal
    prev_close = price.shift(1)
    tr = pd.concat([
        (df["high_price"] - df["low_price"]),
        (df["high_price"] - prev_close).abs(),
        (df["low_price"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=7).mean()
    ts = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["target_price"] = df["close_price"].shift(-1)
    return df

def list_latest_artifacts():
    reg_m = sorted(ALGO_DIR.glob("crypto_regressor_lgbm_*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    clf_m = sorted(ALGO_DIR.glob("crypto_classifier_lgbm_*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    reg_f = sorted(ALGO_DIR.glob("regressor_features_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    clf_f = sorted(ALGO_DIR.glob("classifier_features_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    reg_model = reg_m[0] if reg_m else (ALGO_DIR / "crypto_regressor_lgbm.joblib" if (ALGO_DIR / "crypto_regressor_lgbm.joblib").exists() else None)
    clf_model = clf_m[0] if clf_m else (ALGO_DIR / "crypto_classifier_lgbm.joblib" if (ALGO_DIR / "crypto_classifier_lgbm.joblib").exists() else None)
    reg_feats = reg_f[0] if reg_f else (ALGO_DIR / "regressor_features.json" if (ALGO_DIR / "regressor_features.json").exists() else None)
    clf_feats = clf_f[0] if clf_f else (ALGO_DIR / "classifier_features.json" if (ALGO_DIR / "classifier_features.json").exists() else None)
    return reg_model, clf_model, reg_feats, clf_feats

def scan_artifacts_metrics():
    res = {"regressor": [], "classifier": []}
    try:
        for meta in ALGO_DIR.glob("crypto_regressor_lgbm_*_metadata.json"):
            ts = datetime.fromtimestamp(meta.stat().st_mtime)
            try:
                data = json.load(open(meta, "r"))
            except Exception:
                data = {}
            res["regressor"].append({"path": meta, "ts": ts, "metrics": data.get("metrics") or data})
        for rep in ALGO_DIR.glob("crypto_regressor_lgbm_*_report.json"):
            ts = datetime.fromtimestamp(rep.stat().st_mtime)
            try:
                data = json.load(open(rep, "r"))
            except Exception:
                data = {}
            res["regressor"].append({"path": rep, "ts": ts, "metrics": data.get("metrics") or data})
        for meta in ALGO_DIR.glob("crypto_classifier_lgbm_*_metadata.json"):
            ts = datetime.fromtimestamp(meta.stat().st_mtime)
            try:
                data = json.load(open(meta, "r"))
            except Exception:
                data = {}
            res["classifier"].append({"path": meta, "ts": ts, "metrics": data.get("metrics") or data})
        for rep in ALGO_DIR.glob("crypto_classifier_lgbm_*_report.json"):
            ts = datetime.fromtimestamp(rep.stat().st_mtime)
            try:
                data = json.load(open(rep, "r"))
            except Exception:
                data = {}
            res["classifier"].append({"path": rep, "ts": ts, "metrics": data.get("metrics") or data})
    except Exception:
        pass
    return res

def load_features_list(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def plot_feature_importances(names, importances, title="Importances des features"):
    order = np.argsort(importances)[::-1]
    names = [names[i] for i in order][:30]
    vals = [float(importances[i]) for i in order][:30]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h"))
    fig.update_layout(height=600, title=title, yaxis=dict(autorange="reversed"))
    return fig

@st.cache_resource
def get_models_and_features():
    """Charge les modèles et listes de features les plus récents (avec cache)."""
    reg_model_path, clf_model_path, reg_feat_path, clf_feat_path = list_latest_artifacts()
    if not all([reg_model_path, clf_model_path, reg_feat_path, clf_feat_path]):
        return None
    reg_model = joblib.load(reg_model_path)
    clf_model = joblib.load(clf_model_path)
    reg_feats = load_features_list(reg_feat_path)
    clf_feats = load_features_list(clf_feat_path)
    return {
        "reg_model": reg_model,
        "clf_model": clf_model,
        "reg_feats": reg_feats,
        "clf_feats": clf_feats,
        "paths": {
            "reg_model": reg_model_path,
            "clf_model": clf_model_path,
            "reg_feats": reg_feat_path,
            "clf_feats": clf_feat_path,
        },
    }

st.title("4️⃣ Modélisation et Machine Learning")

tabs = st.tabs([
    "4.1 Problème & choix d’algorithmes",
    "4.2 Pré-traitements & justification",
    "4.3 DataViz & statistiques",
    "4.4 Évaluation & baseline"
])


with tabs[0]:
    st.header("4.1 Problème & choix d’algorithmes")
    st.info("🎯 **Mission :** Assister la décision de trading à court terme")

    # Sélecteur de crypto en haut
    symbols = list_symbols()
    if not symbols:
        st.warning("Aucun symbole actif en base.")
        st.stop()
    sym_top = st.selectbox("Symbole", symbols, index=0, key="ml4_tab1_sym")

    # Essayer d'abord de consommer l'API pour les prédictions (recommandé)
    api_result = api_predict_symbol(sym_top)
    bundle = None  # On ne chargera les modèles locaux qu'en fallback si nécessaire

    try:
        # On charge ~1 an pour assurer assez d'historique pour les fenêtres longues (72h, RSI, etc.)
        df_all = load_candles(sym_top, years=1)
        dff = compute_features(df_all).dropna().reset_index(drop=True)
        if dff.empty:
            st.info("Pas assez d'historique pour calculer les features.")
            st.stop()
        last_row = dff.iloc[[-1]]  # DataFrame d'une ligne pour prédiction
        last_close = float(dff.iloc[-1]["close_price"])  # close de la dernière bougie observée (t)

        col_reg, col_class = st.columns(2)

        with col_reg:
            st.subheader("Régression (Estimation du Prix)")
            st.markdown("📈 Estimer le prix de clôture à $t+1h$ (prochaine bougie).")
            if api_result is not None:
                try:
                    y_next_pred = float(api_result["prediction"]["next_close_price"])  # via API
                    abs_delta = y_next_pred - last_close
                    pct_delta = (abs_delta / last_close * 100.0) if last_close else 0.0
                    delta_str = f"{abs_delta:+.2f} ({pct_delta:+.2f}%) vs close(t)"
                    st.metric("Prix prédit (t+1h)", f"{y_next_pred:,.2f}", delta=delta_str, delta_color="normal")
                except Exception as e:
                    st.warning(f"Réponse API inattendue, bascule en local: {e}")
                    api_result = None  # force fallback
            if api_result is None:
                if bundle is None:
                    bundle = get_models_and_features()
                if not bundle:
                    st.warning("Artefacts (modèles ou listes de features) indisponibles pour la prédiction locale.")
                else:
                    Xr = last_row[bundle["reg_feats"]]
                    y_next_pred = float(bundle["reg_model"].predict(Xr)[0])
                    abs_delta = y_next_pred - last_close
                    pct_delta = (abs_delta / last_close * 100.0) if last_close else 0.0
                    delta_str = f"{abs_delta:+.2f} ({pct_delta:+.2f}%) vs close(t)"
                    st.metric("Prix prédit (t+1h)", f"{y_next_pred:,.2f}", delta=delta_str, delta_color="normal")

        with col_class:
            st.subheader("Classification (Direction)")
            st.markdown("⬆️↔️⬇️ Catégoriser la direction (Baisse / Stable / Hausse) pour $t+1h$.")
            # Mapping souhaité 0=Baisse, 1=Stable, 2=Hausse
            code_to_label = {0: "Baisse", 1: "Stable", 2: "Hausse"}

            if api_result is not None:
                try:
                    pred_label = str(api_result["prediction"]["direction"])  # via API
                    st.metric("Classe prédite (t+1h)", pred_label)
                    # Probabilités si disponibles
                    probs = api_result.get("details", {}).get("probabilities")
                    if isinstance(probs, dict):
                        st.caption("Probabilités de classe (%):")
                        for lbl in ["Baisse", "Stable", "Hausse"]:
                            v = probs.get(lbl)
                            if v is not None:
                                st.progress(min(max(int(round(v)), 0), 100), text=f"{lbl} – {v:.1f}%")
                except Exception as e:
                    st.warning(f"Réponse API inattendue, bascule en local: {e}")
                    api_result = None  # force fallback

            if api_result is None:
                if bundle is None:
                    bundle = get_models_and_features()
                if not bundle:
                    st.warning("Artefacts manquants pour la prédiction locale.")
                else:
                    Xc = last_row[bundle["clf_feats"]]
                    clf_model = bundle["clf_model"]
                    raw_pred = clf_model.predict(Xc)[0]
                    try:
                        pred_idx = int(getattr(raw_pred, 'item', lambda: raw_pred)())
                    except Exception:
                        pred_idx = int(raw_pred)
                    pred_label = code_to_label.get(pred_idx, str(pred_idx))
                    st.metric("Classe prédite (t+1h)", pred_label)

                    if hasattr(clf_model, "predict_proba"):
                        proba = clf_model.predict_proba(Xc)[0]
                        labels = []
                        probs = []
                        if hasattr(clf_model, "classes_"):
                            for i, cls in enumerate(list(clf_model.classes_)):
                                lbl = code_to_label.get(int(cls), str(cls))
                                labels.append(lbl)
                                probs.append(float(proba[i]))
                        else:
                            labels = ["Baisse", "Stable", "Hausse"]
                            probs = [float(proba[i]) if i < len(proba) else 0.0 for i in range(3)]
                        st.caption("Probabilités de classe (%):")
                        for lbl, p in zip(labels, probs):
                            st.progress(min(max(int(round(p * 100)), 0), 100), text=f"{lbl} – {p*100:.1f}%")

            st.caption("Classes réelles: 0=Baisse, 1=Stable, 2=Hausse")
    except Exception as e:
        st.error(f"Prédiction indisponible: {e}")

    st.subheader("Recherche d’Hyperparamètres & Déploiement")
    
    col_search, col_metrics, col_save = st.columns(3)
    
    with col_search:
        st.markdown("### 1. Stratégie")
        st.markdown("💡 **RandomizedSearchCV** (Rapide)")
        st.caption("Balance entre performance et temps de calcul.")

    with col_metrics:
        st.markdown("### 2. Métriques de Sélection")
        
        metrics_df = pd.DataFrame({
            "Tâche": ["Régression", "Classification"],
            "Critères": ["MAE / NMAE", "F1 / Accuracy"]
        })
        st.dataframe(metrics_df, hide_index=True)

    with col_save:
        st.markdown("### 3. Persistance")
        st.markdown("💾 **Volume Docker** 🐳")
        st.caption("Enregistrement automatique pour disponibilité des modèles dans tous les conteneurs.")
        # Afficher les derniers modèles enregistrés dans le volume models_data
        try:
            joblibs = sorted(ALGO_DIR.glob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)[:4]
            if joblibs:
                for p in joblibs:
                    ts = datetime.fromtimestamp(p.stat().st_mtime)
                    st.write(f"• {p.name} (modifié: {ts:%Y-%m-%d %H:%M:%S})")
            else:
                st.info("Aucun modèle .joblib trouvé dans le volume.")
        except Exception as e:
            st.warning(f"Impossible de lister les modèles: {e}")


with tabs[1]:
    st.header("4.2 Pré-traitements & Justification 🧪")
    st.info("💡 **Objectif :** Transformer les **bougies brutes** en **variables explicatives** pour les modèles. ➡️")

    st.write("---")

    st.markdown("### Aperçu et diagnostics par symbole")
    symbols = list_symbols()
    if not symbols:
        st.info("Aucun symbole actif en base.")
    else:
        colp1, colp2 = st.columns([1,1])
        with colp1:
            sym = st.selectbox("Symbole", symbols, key="prep_sym_ml4")
        with colp2:
            months_prep = st.slider("Fenêtre (mois)", 3, 48, 6, step=3, key="ml4_prep_months")

        try:
            years_need = max(1, math.ceil(months_prep / 12))
            df_full = load_candles(sym, years=years_need)
            # Restreindre à la fenêtre choisie
            dfw = df_full[df_full["timestamp"] >= (pd.Timestamp.utcnow() - pd.DateOffset(months=months_prep))].copy()
            dff = compute_features(dfw).reset_index(drop=True)

            # Aperçu tabulaire (dernières lignes)
            if not dff.empty:
                st.dataframe(dff.dropna().tail(100)[[
                    "timestamp", "close_price", "price_lag_1h", "rolling_mean_24h", "rsi", "macd_diff", "atr", "hour_of_day", "day_of_week", "target_price"
                ]])
            else:
                st.info("Pas assez d'historique pour calculer les features sur la fenêtre.")

            st.write("---")
            st.subheader("Autocorrélation des lags (corrélation roulante)")
            lag_win = st.slider("Fenêtre de corrélation (heures)", 24, 240, 72, step=12, help="Taille de la fenêtre pour la corrélation roulante", key="ml4_lag_win")
            try:
                if not dff.empty:
                    fig_ac = go.Figure()
                    for k in range(1, 6):
                        col = f"price_lag_{k}h"
                        if col in dff.columns:
                            corr_series = dff["close_price"].rolling(lag_win, min_periods=max(6, lag_win//6)).corr(dff[col])
                            fig_ac.add_trace(go.Scatter(x=dff["timestamp"], y=corr_series, name=f"lag {k}h", mode="lines"))
                    fig_ac.update_layout(height=320, title=f"{sym} – Corrélation roulante close vs lags (fenêtre={lag_win}h)", yaxis=dict(range=[-1,1]))
                    st.plotly_chart(fig_ac, use_container_width=True)
                else:
                    st.info("Données insuffisantes pour l'autocorrélation.")
            except Exception as e:
                st.warning(f"Autocorrélation indisponible: {e}")

            st.subheader("Répartition des volumes par heure/jour")
            dim = st.radio("Agréger par", ["Heure", "Jour"], horizontal=True, key="ml4_vol_dim")
            try:
                if not dff.empty:
                    dfv = dff.copy()
                    # Assurer la présence des features de temps
                    if "hour_of_day" not in dfv.columns or "day_of_week" not in dfv.columns:
                        ts = pd.to_datetime(dfv["timestamp"])  # sauvegarde
                        dfv["hour_of_day"] = ts.dt.hour
                        dfv["day_of_week"] = ts.dt.dayofweek
                    if dim == "Heure":
                        agg = dfv.groupby("hour_of_day", as_index=False)["volume_base"].sum()
                        x = agg["hour_of_day"].astype(int)
                        x_title = "Heure du jour"
                    else:
                        agg = dfv.groupby("day_of_week", as_index=False)["volume_base"].sum()
                        dow_map = {0:"Lun",1:"Mar",2:"Mer",3:"Jeu",4:"Ven",5:"Sam",6:"Dim"}
                        x = agg["day_of_week"].map(dow_map)
                        x_title = "Jour de la semaine"
                    fig_vol = go.Figure(go.Bar(x=x, y=agg["volume_base"], marker_color="#1f77b4"))
                    fig_vol.update_layout(height=320, title=f"{sym} – Somme des volumes ({months_prep} mois)", xaxis_title=x_title, yaxis_title="Volume (base)")
                    st.plotly_chart(fig_vol, use_container_width=True)
                else:
                    st.info("Données insuffisantes pour la distribution des volumes.")
            except Exception as e:
                st.warning(f"Distribution des volumes indisponible: {e}")

            st.subheader("Lissage par moyennes mobiles")
            try:
                if not dff.empty:
                    fig_ma = go.Figure()
                    fig_ma.add_trace(go.Scatter(x=dff["timestamp"], y=dff["close_price"], name="Close", mode="lines", line=dict(color="#999")))
                    if "rolling_mean_24h" in dff.columns:
                        fig_ma.add_trace(go.Scatter(x=dff["timestamp"], y=dff["rolling_mean_24h"], name="MM 24h", mode="lines", line=dict(color="#2ca02c")))
                    if "rolling_mean_72h" in dff.columns:
                        fig_ma.add_trace(go.Scatter(x=dff["timestamp"], y=dff["rolling_mean_72h"], name="MM 72h", mode="lines", line=dict(color="#d62728")))
                    fig_ma.update_layout(height=360, title=f"{sym} – Close vs Moyennes Mobiles ({months_prep} mois)")
                    st.plotly_chart(fig_ma, use_container_width=True)
                else:
                    st.info("Données insuffisantes pour le plot des moyennes mobiles.")
            except Exception as e:
                st.warning(f"Plot des moyennes mobiles indisponible: {e}")

        except Exception as e:
            st.error(f"Erreur préparation: {e}")


with tabs[2]:
    st.header("4.3 DataViz & statistiques")
    symbols = list_symbols()
    if not symbols:
        st.warning("Aucun symbole actif en base.")
    else:
        col1, col2, col3 = st.columns([2,1,1])
        with col2:
            sym = st.selectbox("Symbole", symbols, index=0, key="viz_sym_ml4")
        with col3:
            months = st.slider("Fenêtre (mois)", 3, 48, 12, step=3)
        with st.spinner("Chargement…"):
            years_need = max(1, math.ceil(months / 12))
            df = load_candles(sym, years=years_need)
            dfl = df[df["timestamp"] >= (pd.Timestamp.utcnow() - pd.DateOffset(months=months))].copy()
            dfl["returns"] = dfl["close_price"].pct_change()
        # Courbe des prix
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dfl["timestamp"], y=dfl["close_price"], name="Close", mode="lines"))
        fig.update_layout(height=380, title=f"{sym} – Prix (derniers {months} mois)")
        st.plotly_chart(fig, use_container_width=True)
        # Histogramme des rendements
        st.subheader("Distribution des rendements horaires")
        hist = go.Figure(data=[go.Histogram(x=dfl["returns"].dropna(), nbinsx=50)])
        hist.update_layout(height=280)
        st.plotly_chart(hist, use_container_width=True)


        st.subheader("Artefacts les plus récents")
        reg_model_path, clf_model_path, reg_feat_path, clf_feat_path = list_latest_artifacts()
        if not all([reg_model_path, clf_model_path, reg_feat_path, clf_feat_path]):
            st.warning("Certains artefacts (modèles ou features) sont manquants dans le dossier algo_crypto.")
        else:
            def fmt(p):
                try:
                    ts = datetime.fromtimestamp(p.stat().st_mtime)
                    return f"{p.name} (modifié: {ts:%Y-%m-%d %H:%M:%S})"
                except Exception:
                    return str(p)
            colA, colB = st.columns(2)
            with colA:
                st.write("Régression:")
                st.write("• Modèle:", fmt(reg_model_path))
                st.write("• Features:", fmt(reg_feat_path))
            with colB:
                st.write("Classification:")
                st.write("• Modèle:", fmt(clf_model_path))
                st.write("• Features:", fmt(clf_feat_path))

            with st.expander("Chargement rapide des modèles (sanity check)"):
                try:
                    reg_model = joblib.load(reg_model_path)
                    clf_model = joblib.load(clf_model_path)
                    with open(reg_feat_path, 'r') as f:
                        reg_feats = json.load(f)
                    with open(clf_feat_path, 'r') as f:
                        clf_feats = json.load(f)
                    c1, c2 = st.columns(2)
                    c1.write(f"Régression: {type(reg_model).__name__} – {len(reg_feats)} features")
                    c2.write(f"Classification: {type(clf_model).__name__} – {len(clf_feats)} features")
                except Exception as e:
                    st.error(f"Erreur chargement modèles: {e}")


with tabs[3]:
    st.header("4.4 Évaluation & comparaison à une baseline")
    md_justify(
        """
        Méthode : comparaison au naïf « persistance » (prix(t + 1h) ≈ close(t)).
        """
    )
    symbols = list_symbols()
    if not symbols:
        st.info("Base indisponible ou aucun symbole actif.")
    else:
        col1, col2 = st.columns([2,1])
        with col2:
            sym = st.selectbox("Symbole", symbols, index=0, key="ml4_eval_sym")
            months_eval = st.slider("Fenêtre (mois)", 3, 48, 12, step=3, key="ml4_eval_months")
            roll_w = st.slider("Fenêtre MAE roulante (pas)", 12, 200, 48, help="Nombre de points (heures)")
        try:
            bundle = get_models_and_features()
            if not bundle:
                st.warning("Artefacts manquants pour l'évaluation.")
                st.stop()
            years_need = max(1, math.ceil(months_eval / 12))
            df = load_candles(sym, years=years_need)
            dfl = df[df["timestamp"] >= (pd.Timestamp.utcnow() - pd.DateOffset(months=months_eval))]
            dff = compute_features(dfl).dropna().reset_index(drop=True)
            # Prédictions régression
            Xr = dff[bundle["reg_feats"]]
            yr_true = dff["target_price"].to_numpy()
            yr_pred = bundle["reg_model"].predict(Xr)
            y_base = dff["close_price"].to_numpy()
            # Overlay séries
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=dff["timestamp"], y=yr_true, name="Prix (t+1)", mode="lines"))
            fig1.add_trace(go.Scatter(x=dff["timestamp"], y=yr_pred, name="Prédit", mode="lines"))
            fig1.add_trace(go.Scatter(x=dff["timestamp"], y=y_base, name="Baseline (persistance)", mode="lines", line=dict(dash="dot")))
            fig1.update_layout(height=380, title=f"{sym} – Réel (t+1) vs Prédit vs Baseline")
            st.plotly_chart(fig1, use_container_width=True)
            # MAE globaux
            mae = float(np.mean(np.abs(yr_true - yr_pred)))
            rmse = float(np.sqrt(np.mean((yr_true - yr_pred) ** 2)))
            mae_base = float(np.mean(np.abs(yr_true - y_base)))
            rmse_base = float(np.sqrt(np.mean((yr_true - y_base) ** 2)))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE (modèle)", f"{mae:.2f}")
            m2.metric("RMSE (modèle)", f"{rmse:.2f}")
            m3.metric("MAE (baseline)", f"{mae_base:.2f}", f"Δ {(mae_base-mae)/mae_base*100:.1f}%")
            m4.metric("RMSE (baseline)", f"{rmse_base:.2f}", f"Δ {(rmse_base-rmse)/rmse_base*100:.1f}%")
            st.caption("Baseline = dernier close. Δ = amélioration relative vs baseline.")
            # MAE roulante
            err_model = np.abs(yr_true - yr_pred)
            err_base = np.abs(yr_true - y_base)
            s_model = pd.Series(err_model).rolling(roll_w, min_periods=max(4, roll_w//4)).mean()
            s_base = pd.Series(err_base).rolling(roll_w, min_periods=max(4, roll_w//4)).mean()
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(x=dff["timestamp"], y=s_model, name="MAE roulante – modèle"))
            fig_roll.add_trace(go.Scatter(x=dff["timestamp"], y=s_base, name="MAE roulante – baseline", line=dict(dash="dot")))
            fig_roll.update_layout(height=320, title="MAE roulante (stabilité des erreurs)")
            st.plotly_chart(fig_roll, use_container_width=True)
        except Exception as e:
            st.error(f"Analyse indisponible: {e}")

    st.markdown("---")
    st.subheader("Backtest classification – stratégie Buy/Hold/Sell vs Buy&Hold")
    md_justify(
        """
        Méthode de démo : on applique le classificateur sur la fenêtre choisie,
        puis on simule deux courbes d’équité avec frais:
        - Équité stratégie (positions −1/0/+1, frais à chaque changement: 1 trade pour entrée/sortie, 2 pour inversion directe)
        - Équité Buy&Hold (référence)
        """
    )
    symbols = list_symbols()
    if symbols:
        colb1, colb2, colb3 = st.columns([2,1,1])
        with colb2:
            sym_b = st.selectbox("Symbole (clf)", symbols, index=0, key="ml4_bt_sym")
        with colb3:
            fee = st.slider("Frais (bps)", 0, 50, 10, help="1 bps = 0.01% – appliqués aux changements de position") / 10000.0
        init_cap = st.number_input("Capital initial (USDT)", min_value=100.0, max_value=1000000.0, value=1000.0, step=100.0)
        try:
            bundle = get_models_and_features()
            if not bundle:
                st.info("Artefacts manquants pour le backtest classification.")
            else:
                months_eval_bt = st.session_state.get("ml4_eval_months", 12)
                years_need = max(1, math.ceil(months_eval_bt / 12))
                df = load_candles(sym_b, years=years_need)
                dfl = df[df["timestamp"] >= (pd.Timestamp.utcnow() - pd.DateOffset(months=months_eval_bt))]
                dff = compute_features(dfl).dropna().reset_index(drop=True)
                # Prédire classes
                Xc = dff[bundle["clf_feats"]]
                raw_pred = bundle["clf_model"].predict(Xc)
                # Convertir en natifs
                preds_idx = [int(getattr(v, 'item', lambda: v)()) if hasattr(v, 'item') else int(v) for v in raw_pred]
                # Map classes -> signaux
                label_to_signal = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
                signals = [label_to_signal.get(v, 'Hold') for v in preds_idx]
                # Construire positions -1/0/+1 et calculer equity
                price = dff["close_price"].reset_index(drop=True)
                ts = dff["timestamp"].reset_index(drop=True)
                pos = pd.Series(signals).map({'Sell': -1, 'Hold': 0, 'Buy': 1}).astype(int)
                pos_expo = pos.shift(1).fillna(0)
                ret = price.pct_change().fillna(0)
                # Frais: calculés aux changements de position (1 trade pour entrer/sortir, 2 pour flip)
                prev = pos.shift(1).fillna(0)
                curr = pos
                trade_count = np.where((prev==0) & (curr!=0), 1,
                                  np.where((prev!=0) & (curr==0), 1,
                                      np.where((prev!=0) & (curr!=0) & (np.sign(prev)!=np.sign(curr)), 2, 0)))
                fee_factor = (1 - fee) ** trade_count
                growth = (1 + ret * pos_expo) * fee_factor
                equity_model = pd.Series(growth).cumprod()
                equity_bh = price / price.iloc[0]
                # Conversion en USDT
                equity_model_usd = equity_model * init_cap
                equity_bh_usd = equity_bh * init_cap
                # Plot
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=ts, y=equity_bh_usd, name="Buy&Hold"))
                fig_bt.add_trace(go.Scatter(x=ts, y=equity_model_usd, name="Stratégie (clf)", line=dict(dash="solid")))
                fig_bt.update_layout(height=380, title=f"{sym_b} – Equity (USDT): stratégie (clf) vs Buy&Hold")
                st.plotly_chart(fig_bt, use_container_width=True)
                # Métriques
                def _max_drawdown(equity_series: pd.Series) -> float:
                    if len(equity_series) == 0:
                        return 0.0
                    rolling_max = equity_series.cummax()
                    dd = equity_series / rolling_max - 1.0
                    return float(dd.min())
                model_final = float(equity_model_usd.iloc[-1])
                bh_final = float(equity_bh_usd.iloc[-1])
                model_ret = (model_final / float(init_cap) - 1.0) * 100.0
                bh_ret = (bh_final / float(init_cap) - 1.0) * 100.0
                dd_model = _max_drawdown(equity_model_usd) * 100.0
                dd_bh = _max_drawdown(equity_bh_usd) * 100.0
                total_trades = int(np.sum(trade_count))
                flips = int(np.sum((prev!=0) & (curr!=0) & (np.sign(prev)!=np.sign(curr))))
                entries_exits = total_trades - 2*flips
                cma, cmb, cmc, cmd = st.columns(4)
                cma.metric("Final stratégie (USDT)", f"{model_final:,.2f}")
                cmb.metric("Rendement stratégie", f"{model_ret:.2f}%")
                cmc.metric("Max drawdown (strat)", f"{dd_model:.2f}%")
                cmd.metric("Trades (± flips)", f"{total_trades} (±{flips})")
                c2a, c2b = st.columns(2)
                c2a.metric("Final Buy&Hold (USDT)", f"{bh_final:,.2f}")
                c2b.metric("Max drawdown (BH)", f"{dd_bh:.2f}%")
                st.caption("Remarque: l’équité est simulée en USDT avec capital initial configurable. Les frais sont appliqués aux changements de position (1 trade entrée/sortie, 2 trades pour inversion).")
        except Exception as e:
            st.error(f"Backtest classification indisponible: {e}")

