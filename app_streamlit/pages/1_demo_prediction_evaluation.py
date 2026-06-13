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

st.set_page_config(page_title="Demo - Prediction & Evaluation", layout="wide")

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


def api_status(symbol: str) -> dict | None:
    try:
        base = _api_base_url().rstrip("/")
        resp = requests.get(f"{base}/status", params={"symbol": symbol}, timeout=5)
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
    # Feature parfois attendue par les artefacts existants.
    df["atr_pct"] = (df["atr"] / df["close_price"].replace(0, np.nan)) * 100.0
    ts = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    # Valeur par defaut si la feature categorielle n'est pas recalculable ici.
    df["symbol_cat"] = 0.0
    df["target_price"] = df["close_price"].shift(-1)
    return df


def align_model_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Construit une matrice X compatible avec les features attendues par le modele.

    Si certaines colonnes n'existent pas dans le DataFrame courant, on les cree
    avec des valeurs neutres pour eviter les erreurs bloquantes en demo.
    """
    x = df.copy()

    if "symbol_cat" in feature_names and "symbol_cat" not in x.columns:
        x["symbol_cat"] = 0.0

    if "atr_pct" in feature_names and "atr_pct" not in x.columns:
        if "atr" in x.columns and "close_price" in x.columns:
            x["atr_pct"] = (x["atr"] / x["close_price"].replace(0, np.nan)) * 100.0
        else:
            x["atr_pct"] = 0.0

    for col in feature_names:
        if col not in x.columns:
            x[col] = 0.0

    return x[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)

def list_latest_artifacts():
    reg_m = sorted(ALGO_DIR.glob("crypto_regressor_lgbm_*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    reg_f = sorted(ALGO_DIR.glob("regressor_features_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    reg_model = reg_m[0] if reg_m else (ALGO_DIR / "crypto_regressor_lgbm.joblib" if (ALGO_DIR / "crypto_regressor_lgbm.joblib").exists() else None)
    reg_feats = reg_f[0] if reg_f else (ALGO_DIR / "regressor_features.json" if (ALGO_DIR / "regressor_features.json").exists() else None)
    return reg_model, reg_feats

def scan_artifacts_metrics():
    res = {"regressor": []}
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
    reg_model_path, reg_feat_path = list_latest_artifacts()
    if not all([reg_model_path, reg_feat_path]):
        return None
    reg_model = joblib.load(reg_model_path)
    reg_feats = load_features_list(reg_feat_path)
    return {
        "reg_model": reg_model,
        "reg_feats": reg_feats,
        "paths": {
            "reg_model": reg_model_path,
            "reg_feats": reg_feat_path,
        },
    }

st.title("Demo ML: Prediction & Evaluation")

tabs = st.tabs([
    "Demo Prediction",
    "Evaluation Modele"
])


with tabs[0]:
    st.header("Prediction en direct")
    st.info("🎯 **Mission :** Assister la décision de trading à court terme")

    # Sélecteur de crypto en haut
    symbols = list_symbols()
    if not symbols:
        st.warning("Aucun symbole actif en base.")
        st.stop()
    sym_top = st.selectbox("Symbole", symbols, index=0, key="ml4_tab1_sym")

    # Essayer d'abord de consommer l'API pour les prédictions (recommandé)
    api_result = api_predict_symbol(sym_top)
    api_status_payload = api_status(sym_top)
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

        col_reg, col_price = st.columns(2)

        with col_reg:
            st.subheader("Régression (Variation Relative)")
            st.markdown("📈 Estimer la variation en % du prix à $t+1h$ (prochaine bougie).")
            open_current = None
            close_current = last_close
            predicted_next_close = None
            if api_result is not None:
                try:
                    pct_change_pred = float(api_result["prediction"]["next_close_pct_change"])  # via API
                    current_candle = api_result.get("current_candle") or {}
                    open_current = current_candle.get("open")
                    close_current = float(current_candle.get("close", last_close))
                    predicted_next_close = float(api_result["prediction"]["next_close_predicted"])
                    st.metric("Variation prédite (t+1h)", f"{pct_change_pred:+.3f}%")
                except Exception as e:
                    st.warning(f"Réponse API inattendue, bascule en local: {e}")
                    api_result = None  # force fallback
            if api_result is None:
                if bundle is None:
                    bundle = get_models_and_features()
                if not bundle:
                    st.warning("Artefacts (modèles ou listes de features) indisponibles pour la prédiction locale.")
                else:
                    Xr = align_model_features(last_row, bundle["reg_feats"])
                    pct_change_pred = float(bundle["reg_model"].predict(Xr)[0])
                    predicted_next_close = close_current * (1 + pct_change_pred / 100.0)
                    st.metric("Variation prédite (t+1h)", f"{pct_change_pred:+.3f}%")

        with col_price:
            st.subheader("Prix (Bougie actuelle vs prévision)")
            if api_result is None:
                open_current = float(dff.iloc[-1]["open_price"])
                close_current = float(dff.iloc[-1]["close_price"])
                if "pct_change_pred" in locals():
                    predicted_next_close = close_current * (1 + pct_change_pred / 100.0)

            st.metric("Open actuel (t)", f"{float(open_current):,.4f}" if open_current is not None else "n/a")
            st.metric("Close actuel (t)", f"{float(close_current):,.4f}")
            st.metric("Close prédit (t+1h)", f"{float(predicted_next_close):,.4f}" if predicted_next_close is not None else "n/a")

            if predicted_next_close is not None:
                delta_abs = float(predicted_next_close) - float(close_current)
                direction_txt = "Hausse" if delta_abs > 0 else ("Baisse" if delta_abs < 0 else "Stable")
                st.metric("Sens implicite", direction_txt)

        st.write("---")
        st.subheader("Modèle déployé (Champion) et Challenger")
        deployment = (api_status_payload or {}).get("model_deployment") if api_status_payload else None
        if deployment and isinstance(deployment, dict):
            champion = deployment.get("deployed") or {}
            challenger = deployment.get("last_candidate") or {}
            cdep1, cdep2 = st.columns(2)
            with cdep1:
                st.markdown("**Champion déployé**")
                st.write(f"- Regressor: {champion.get('regressor', 'n/a')}")
                st.write(f"- Trained at: {champion.get('trained_at', 'n/a')}")
                st.write(f"- Score: {champion.get('score', 'n/a')}")
            with cdep2:
                st.markdown("**Dernier challenger**")
                st.write(f"- Regressor: {challenger.get('regressor', 'n/a')}")
                st.write(f"- Trained at: {challenger.get('trained_at', 'n/a')}")
                st.write(f"- Score: {challenger.get('score', 'n/a')}")
        else:
            st.info("Aucun registre de déploiement trouvé. Le dernier modèle est utilisé par défaut.")
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
            "Tâche": ["Régression"],
            "Critères": ["MAE / RMSE / R²"]
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
    st.header("Evaluation vs baseline")
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
            # Le modele de regression predit une variation en pourcentage.
            # Conversion en prix predit pour comparer des grandeurs homogenes.
            Xr = align_model_features(dff, bundle["reg_feats"])
            yr_true = dff["target_price"].to_numpy()
            y_base = dff["close_price"].to_numpy()
            yr_pred_pct = bundle["reg_model"].predict(Xr)
            yr_pred = y_base * (1.0 + (yr_pred_pct / 100.0))
            # Overlay séries
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=dff["timestamp"], y=yr_true, name="Prix (t+1)", mode="lines"))
            fig1.add_trace(go.Scatter(x=dff["timestamp"], y=yr_pred, name="Prix predit (depuis %)", mode="lines"))
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
            m3.metric("MAE (baseline)", f"{mae_base:.2f}")
            m4.metric("RMSE (baseline)", f"{rmse_base:.2f}")

            # Delta explicite "modèle - baseline" pour éviter toute ambiguïté visuelle.
            # Sur des erreurs, plus bas = mieux: une delta négative est favorable.
            d1, d2 = st.columns(2)
            d1.metric(
                "Δ MAE (modèle - baseline)",
                f"{(mae - mae_base):+.2f}",
                f"{((mae - mae_base) / mae_base) * 100:+.1f}%",
                delta_color="inverse",
            )
            d2.metric(
                "Δ RMSE (modèle - baseline)",
                f"{(rmse - rmse_base):+.2f}",
                f"{((rmse - rmse_base) / rmse_base) * 100:+.1f}%",
                delta_color="inverse",
            )
            st.caption("Le modele predit d'abord une variation %, convertie ici en prix predit pour une comparaison prix vs prix.")
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

            st.markdown("---")
            st.subheader("Backtest directionnel (régression) vs Buy&Hold")
            st.caption(
                "Règle: on prend une position LONG si la variation prédite t+1 dépasse un seuil, sinon position neutre."
            )

            cbt1, cbt2, cbt3 = st.columns(3)
            with cbt1:
                threshold_bps = st.slider(
                    "Seuil signal (bps/pourcentage)", 0, 30, 5,
                    help="Le signal LONG est activé si prediction_pct > seuil. (1 bps = 0.01%)"
                )
            with cbt2:
                fee_bps = st.slider(
                    "Frais (bps)", 0, 30, 5,
                    help="Frais appliqués lors d'un changement de position. (1 bps = 0.01%)"
                )
            with cbt3:
                init_cap = st.number_input(
                    "Capital initial (USDT)",
                    min_value=100.0,
                    max_value=1_000_000.0,
                    value=1000.0,
                    step=100.0,
                )

            threshold_pct = threshold_bps / 100.0
            fee_rate = fee_bps / 10000.0

            # Strategie long/flat: 1 si signal positif au-dessus du seuil, sinon 0
            signal = (yr_pred_pct > threshold_pct).astype(int)
            position = pd.Series(signal, index=dff.index)
            position_expo = position.shift(1).fillna(0)

            # Rendements reels de marche sur chaque pas
            mkt_ret = pd.Series(y_base, index=dff.index).pct_change().fillna(0.0)

            # Frais uniquement lors des changements de position
            trade_count = (position != position.shift(1).fillna(0)).astype(int)
            fee_factor = (1.0 - fee_rate) ** trade_count

            strat_growth = (1.0 + (mkt_ret * position_expo)) * fee_factor
            strat_equity = strat_growth.cumprod() * init_cap

            bh_equity = (1.0 + mkt_ret).cumprod() * init_cap

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=dff["timestamp"], y=bh_equity, name="Buy&Hold", mode="lines"))
            fig_bt.add_trace(go.Scatter(x=dff["timestamp"], y=strat_equity, name="Strategie long/flat", mode="lines"))
            fig_bt.update_layout(height=360, title=f"{sym} - Equity (USDT) : Strategie vs Buy&Hold")
            st.plotly_chart(fig_bt, use_container_width=True)

            strat_final = float(strat_equity.iloc[-1])
            bh_final = float(bh_equity.iloc[-1])
            strat_ret = (strat_final / init_cap - 1.0) * 100.0
            bh_ret = (bh_final / init_cap - 1.0) * 100.0
            nb_trades = int(trade_count.sum())

            b1, b2, b3 = st.columns(3)
            b1.metric("Rendement stratégie", f"{strat_ret:.2f}%")
            b2.metric("Rendement Buy&Hold", f"{bh_ret:.2f}%")
            b3.metric("Nombre de changements", f"{nb_trades}")

            st.caption(
                "Suggestion: cette stratégie est volontairement simple. Une variante robuste est de remplacer long/flat par "
                "long/flat/short avec seuil symétrique et filtre ATR pour éviter de trader les faibles signaux."
            )
        except Exception as e:
            st.error(f"Analyse indisponible: {e}")

