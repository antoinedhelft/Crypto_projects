import streamlit as st
from pathlib import Path

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(page_title="6 - Synthèse", layout="wide")

st.title("6️⃣ Synthèse")

# ============================================================
# ONGLET PRINCIPAUX (ordre modifié)
# ============================================================
tabs = st.tabs([
    "6.1 Difficultés rencontrées",
    "6.2 Perspectives",
    "6.3 Conclusions"
])

# ============================================================
# 6.1 DIFFICULTÉS RENCONTRÉES
# ============================================================
with tabs[0]:
    st.header("6.1 Difficultés rencontrées (données, compétences, infrastructure, timing)")

    st.markdown("""
    ### ⚙️ 

    - **Partitionner les scripts initiaux** : réorganisation et découpage des scripts pour une meilleure modularité.
    - **Gestion du stockage des modèles** : mise en place d’un volume partagé (`models_data`) pour rendre les modèles accessibles à Streamlit et à l’API.
    - **Automatisation dans Airflow** : difficulté à exécuter automatiquement les DAGS sous certaines conditions.
    - **Performance des modèles ML** : recherche de modèles suffisamment performants et stables.
    - **Surveillance Prometheus & Grafana** : problèmes d’incompatibilité, le système fonctionnant sur une machine mais pas sur une autre.
    """)


# ============================================================
# 6.2 PERSPECTIVES
# ============================================================
with tabs[1]:
    st.header("6.2 Perspectives & améliorations futures")
    st.markdown("""
    ### 🚀

    - **Amélioration du Machine Learning** : affiner les modèles avec un plus grand volume de données.
    - **Migration vers le Cloud** : déploiement complet pour améliorer la scalabilité et la disponibilité.
    - **Optimisation du Lag ML** : réduction du temps d’exécution des algorithmes.
    - **Recherche de meilleurs paramètres** : approfondir les essais avec `GridSearchCV` pour optimiser les hyperparamètres.
    - **Mise en production complète de Prometheus & Grafana** : assurer la supervision stable sur toutes les machines.
    """)

# ============================================================
# 6.3 CONCLUSIONS
# ============================================================

with tabs[2]:
    st.header("6.3 Conclusions")

    root_dir = Path(__file__).parents[1]
    img_path = root_dir / "images"

    # Centrer l'image avec des colonnes Streamlit
    col_l,col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.image(str(img_path / "CryptoBot Architecture.png"), width=1000)

    st.markdown("""
    ### 📈

    🔄 **Automatisation via Airflow** : collecte, entraînement, mise à jour des modèles.  
    🐳 **Conteneurisation Docker** : chaque service est isolé, portable et réutilisable.  
    🚀 **API FastAPI** : expose les modèles ML pour des prédictions en temps réel.  
    🧠 **Machine Learning régulier** : modèles mis à jour automatiquement chaque mois.  
    ✅ **Environnement reproductible, scalable et prêt pour la production.**  
    """)

# ============================================================
# BAS DE PAGE
# ============================================================
st.caption("Page 6 – Synthèse & perspectives")
