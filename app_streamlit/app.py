import os
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Crypto Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("CryptoBot")

st.markdown("---")

# Résoudre le chemin de l'image relativement à ce fichier, pour fonctionner en local et en Docker
img_path = Path(__file__).parent / "images"

# Centrer l'image avec des colonnes Streamlit
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    st.image(str(img_path / "19910.jpg"), width=1000)


st.markdown(
    """
    Bienvenue 👋

    Utilisez le menu de la barre latérale pour naviguer entre les pages.
    
    """
)
st.markdown("---")
st.markdown("Un projet développé par BLONDEL Jean-Christophe, DHELFT Antoine, HACHICHE Heykel, NEGGAL Amine.")

