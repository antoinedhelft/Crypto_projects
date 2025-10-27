import streamlit as st
from pathlib import Path

st.set_page_config(page_title="1 - Introduction", layout="wide")

st.title("1️⃣ Introduction")

tabs = st.tabs([
    "1.1 Contexte général du projet",
    "1.2 Positionnement Data Engineer",
    "1.3 Dimensions techniques / éco / scientifiques",
    "1.4 Objectifs du projet",
])

with tabs[0]:
    st.header("1.1 Contexte général du projet")

    st.markdown(
    """
    - 📈 Explosion des données sur les marchés financiers  
    - 💻 Le Data Engineer joue un rôle central dans leur structuration  
    - 🪙 Projet centré sur les données de **Binance**  
    - 🧠 Reflechir afin de concevoir une pipeline de données complète appliquée au domaine des crypto-actifs, depuis la collecte jusqu’à la mise en production.
    """)
    
with tabs[1]:
    st.header("1.2 Positionnement du projet dans le métier de Data Engineer")

    st.markdown(
    """
    Le projet permet de mettre en pratique l’ensemble des compétences clés attendues d’un Data Engineer :
    - Ingestion de données à partir de sources externes (APIs Binance)
    Nettoyage et prétraitement pour rendre la donnée exploitable
    - Modélisation relationnelle et stockage dans une base SQL
    - Industrialisation via Docker, FastAPI, et automatisation (Airflow / CI/CD)
    - Supervision et monitoring du pipeline en production

    Ainsi, ce travail reflète fidèlement la réalité d’un poste de Data Engineer, où l’objectif n’est pas seulement d’analyser des données, mais de concevoir une architecture durable et automatisée pour les exploiter dans un cadre métier.
    """)



with tabs[2]:
    st.header("1.3 Dimensions techniques, économiques et scientifiques")

    root_dir = Path(__file__).parents[1]
    img_path = root_dir / "images"

    # Centrer l'image avec des colonnes Streamlit
    col_l,col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.image(str(img_path / "CryptoBot Architecture.png"), width=1000)

    st.markdown(
    """
    **Technique**
    - 🐍 Python, API Binance, PostgreSQL, Docker, Airflow  
    - 📈 Données OHLCV, structurées et fiables  

    **Économique**
    - 💸 Marché volatile et concurrentiel  
    - 📊 Données = actif stratégique  

    **Scientifique**
    - 🔮 Objectif : prédire les tendances à court terme  
    - 🧠 Gestion de la volatilité et des séries temporelles
    """)




with tabs[3]:
    st.header("1.4 Objectifs du projet")

    st.markdown(
    """
    - 🧩 Créer un **pipeline complet** de données financières  
    - 🔍 Collecter, stocker et modéliser les données Binance  
    - 📊 Prédire la **tendance des marchés** (hausse / baisse)  
    - 🚀 Mettre en production le modèle via une **API + automatisation**

    """)

st.caption("Page 1 – Introduction")
