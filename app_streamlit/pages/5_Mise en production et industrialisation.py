import streamlit as st

st.set_page_config(page_title="5 - Mise en production et industrialisation", layout="wide")

st.title("5️⃣ Mise en production et industrialisation")
st.markdown("Cette page montre des preuves concretes que le projet est executable, testable et maintenable.")

st.markdown("---")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Services principaux", "6")
k2.metric("Orchestrateur", "Airflow")
k3.metric("API", "FastAPI")
k4.metric("CI", "GitHub Actions")

st.markdown("---")

tabs = st.tabs([
    "5.1 Ce qui est en place",
    "5.2 Preuves techniques",
    "5.3 Risques operationnels",
    "5.4 Niveau de maturite"
])

with tabs[0]:
    st.header("5.1 Ce qui est en place")
    st.markdown(
        """
- Ingestion et mise a jour de donnees automatisees via DAGs Airflow.
- Entrainement ML periodique et publication des artefacts modeles.
- Exposition des predictions et du statut via API FastAPI.
- Dashboard Streamlit branche sur base de donnees et API.
- Isolation des services avec Docker Compose.
- Tests automatises executes sur GitHub Actions au push et en pull request.
        """
    )

with tabs[1]:
    st.header("5.2 Preuves techniques")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Execution")
        st.markdown(
            """
- Les DAGs critiques sont valides et testes.
- Le endpoint status expose la fraicheur des donnees et les modeles deployes.
- Le dashboard affiche la prediction t+1h et les probabilites par classe.
            """
        )

    with c2:
        st.subheader("Qualite")
        st.markdown(
            """
- Tests unitaires API et integration API executes en CI.
- Contrats Airflow verifies en CI Linux.
- Gestion de version des artefacts modeles avec convention de nommage datee.
            """
        )

    st.info("Message cle: le projet depasse le prototype local, il est scriptable et verifiable automatiquement.")

with tabs[2]:
    st.header("5.3 Risques operationnels")
    st.markdown(
        """
- Couplage local a certains hostnames Docker si l'environnement change.
- Variabilite des librairies Airflow selon versions de providers.
- Besoin de surveillance active de la fraicheur des donnees et des derivees de modeles.
- Dependance a la disponibilite de l'API source marche.
        """
    )

with tabs[3]:
    st.header("5.4 Niveau de maturite")

    st.markdown("### Evaluation rapide")
    maturity = {
        "Dimension": [
            "Automatisation pipeline",
            "Qualite logicielle",
            "Observabilite",
            "Scalabilite",
            "Securite secrets",
        ],
        "Niveau actuel": ["Intermediaire", "Intermediaire", "Debut", "Debut", "Intermediaire"],
        "Commentaire": [
            "DAGs en place, execution planifiee.",
            "Tests en CI et conventions de code.",
            "A renforcer avec metriques et alertes centralisees.",
            "Architecture conteneurisee mais pas encore cloud-native.",
            "Usage de variables d'environnement, a durcir avec un secret manager.",
        ],
    }
    st.dataframe(maturity, use_container_width=True, hide_index=True)

st.divider()
st.caption("Page 5 - Industrialisation orientee preuves")
