import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="CryptoBot - Demo",
    page_icon="📈",
    layout="wide",
)


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
            Le projet automatise la collecte des donnees, l'entrainement des modeles et l'exposition des predictions via API.
            Pour la soutenance, cette home sert d'entree de demo: aller directement a la prediction puis a l'evaluation.
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

st.markdown(
    """
### Comment utiliser la demo

1. Ouvrir la page 4 pour lancer une prediction en direct.
2. Montrer le graphe d'evaluation (prix reel vs prix predit vs baseline).
3. Montrer le backtest classification.
4. Utiliser les pages 5 et 6 uniquement en support si besoin.

    """
)

