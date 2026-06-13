import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="CryptoBot - Demo",
    page_icon="📈",
    layout="wide",
)



st.title("Pipeline de prédiction")
st.markdown(
    """
    <div class="hero">
      <h4>Aide à l'achat de cryptomonnaies</h4>
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

1. Ouvrir la page "Demo Prediction" pour lancer une prediction en direct.
2. Ouvrir la section "Evaluation Modele" pour comparer modele vs baseline.
3. Montrer l'analyse "volume vs variation de prix" par heure/jour.

    """
)

