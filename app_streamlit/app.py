import os
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(
    page_title="CryptoBot - Demo",
    page_icon="📈",
    layout="wide",
)


def _api_base_url() -> str:
    return os.getenv("API_URL", "http://api:8000" if Path("/app").exists() else "http://localhost:8000")


def _api_status(symbol: str = "BTCUSDT") -> dict | None:
    try:
        base = _api_base_url().rstrip("/")
        resp = requests.get(f"{base}/status", params={"symbol": symbol}, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


st.markdown(
    """
    <style>
    .hero {
        padding: 1.2rem 1.3rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #f3f8ff 0%, #ecfdf5 100%);
        border: 1px solid #dbeafe;
        margin-bottom: 1rem;
    }
    .muted {
        color: #475569;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("CryptoBot")
st.markdown(
    """
    <div class="hero">
      <h4>Assistant de decision crypto en continu</h4>
      <p class="muted">
      Le projet automatise la collecte des donnees, le retrain des modeles et l'exposition des predictions via API.
      Cette page donne une vue executif: statut plateforme, fraicheur donnees et version des modeles deployes.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Résoudre le chemin de l'image relativement à ce fichier, pour fonctionner en local et en Docker
img_path = Path(__file__).parent / "images"

# Centrer l'image avec des colonnes Streamlit
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    st.image(str(img_path / "19910.jpg"), width=900)

status = _api_status("BTCUSDT")

st.markdown("### Indicateurs cles")
c1, c2, c3, c4 = st.columns(4)

if status:
    # Le payload /status expose surtout: models, data_freshness, data_quality, model_deployment.
    overall = status.get("overall_status") or ("ok" if status.get("models") else "unknown")
    freshness_obj = status.get("data_freshness") or {}
    freshness = freshness_obj.get("last_candle") or freshness_obj.get("latest_timestamp") or "n/a"
    deployed = ((status.get("model_deployment") or {}).get("deployed") or {})
    model_pair = f"{deployed.get('regressor', 'n/a')} | {deployed.get('classifier', 'n/a')}"
    dq = status.get("data_quality") or {}
    if isinstance(dq, dict):
        if dq.get("available") is True:
            quality_state = "available"
        elif dq.get("available") is False:
            quality_state = dq.get("reason", "unavailable")
        else:
            quality_state = dq.get("status", "n/a")
    else:
        quality_state = "n/a"

    c1.metric("Statut plateforme", overall)
    c2.metric("Fraicheur donnees", freshness)
    c3.metric("Modeles deployes", model_pair)
    c4.metric("Qualite donnees", quality_state)
else:
    c1.metric("Statut plateforme", "API indisponible")
    c2.metric("Fraicheur donnees", "n/a")
    c3.metric("Modeles deployes", "n/a")
    c4.metric("Qualite donnees", "n/a")


st.markdown(
    """
### Comment utiliser la demo

1. Ouvrir la page 4 pour lancer une prediction en direct.
2. Verifier la comparaison Modele vs Baseline et le backtest.
3. Ouvrir la page 5 pour voir les preuves d'industrialisation.
4. Ouvrir la page 6 pour les limites actuelles et les ameliorations proposees.

    """
)

