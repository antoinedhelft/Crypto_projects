import streamlit as st
from pathlib import Path

st.set_page_config(page_title="1 - Introduction", layout="wide")

st.title("1️⃣ Introduction")

st.markdown(
    """
Cette demo presente un pipeline data + ML applique au marche crypto:
- ingestion automatisee des donnees,
- stockage SQL,
- entrainement et deploiement de modeles,
- exposition des predictions via API,
- visualisation dans Streamlit.
    """
)

c1, c2, c3 = st.columns(3)
c1.metric("Source", "Binance")
c2.metric("Frequence data", "1h")
c3.metric("Cible", "Direction t+1h")

st.markdown("---")

st.subheader("Ce que la demo veut prouver")
st.markdown(
    """
1. Le pipeline est industrialisable (Airflow + Docker + CI).
2. Les predictions sont exploitables via API.
3. Les resultats sont comparables a une baseline simple.
    """
)

root_dir = Path(__file__).parents[1]
img_path = root_dir / "images"
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    st.image(str(img_path / "CryptoBot Architecture.png"), width=900)

st.caption("Page 1 - Version courte pour presentation")
