import streamlit as st
from pathlib import Path

st.set_page_config(page_title="6 - Synthese", layout="wide")

st.title("6️⃣ Synthese")

st.markdown("---")

tabs = st.tabs([
    "6.1 Ce que le projet prouve",
    "6.2 Points a ameliorer et comment",
    "6.3 Conclusion"
])

with tabs[0]:
    st.header("6.1 Ce que le projet prouve")

    c1, c2, c3 = st.columns(3)
    c1.metric("Pipeline data", "Automatise")
    c2.metric("Prediction", "Exposee via API")
    c3.metric("CI", "Active sur GitHub")

    st.markdown(
        """
- Le pipeline collecte, structure et met a jour les donnees de marche.
- Les modeles sont entraines puis consommes par une API exploitable.
- Le dashboard permet une demonstration fonctionnelle de bout en bout.
- Le projet est testable et executable de facon reproductible avec Docker.
        """
    )

with tabs[1]:
    st.header("6.2 Points a ameliorer et comment")

    st.markdown("### 1) Observabilite")
    st.markdown(
        """
**A ameliorer**
- Suivi centralise des erreurs et des latences.

**Comment**
- Ajouter des metriques techniques standard dans l'API.
- Exposer des indicateurs de fraicheur de donnees et de succes des DAGs dans une vue unique.
- Ajouter des alertes simples sur echecs repetes.
        """
    )

    st.markdown("### 2) Robustesse des DAGs")
    st.markdown(
        """
**A ameliorer**
- Mieux couvrir les cas de bord (services indisponibles, variables manquantes).

**Comment**
- Renforcer les pre-checks dans les DAGs.
- Homogeneiser les conventions d'environnement pour tous les operateurs Docker.
- Ajouter des tests d'integration sur les scenarios d'echec critiques.
        """
    )

    st.markdown("### 3) Qualite modele")
    st.markdown(
        """
**A ameliorer**
- Stabilite de la performance selon periode et symbole.

**Comment**
- Suivre les metriques par fenetre temporelle.
- Introduire des seuils d'acceptation avant de deployer un nouveau modele.
- Conserver une comparaison systematique avec la baseline sur la meme periode.
        """
    )

    st.markdown("### 4) Securite configuration")
    st.markdown(
        """
**A ameliorer**
- Durcir la gestion des secrets selon les environnements.

**Comment**
- Garder les variables dans .env en local.
- En CI et en production, migrer vers des secrets GitHub ou un secret manager dedie.
- Eviter toute valeur sensible en dur dans le code.
        """
    )

with tabs[2]:
    st.header("6.3 Conclusion")

    root_dir = Path(__file__).parents[1]
    img_path = root_dir / "images"

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.image(str(img_path / "CryptoBot Architecture.png"), width=900)

    st.success(
        "Le projet est une base solide de data engineering applique au ML: pipeline automatise, API operationnelle, "
        "interface de demonstration et tests CI. Les ameliorations identifiees sont concretes et actionnables."
    )

st.caption("Page 6 - Synthese orientee impact")
