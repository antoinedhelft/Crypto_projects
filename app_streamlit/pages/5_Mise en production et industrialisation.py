import streamlit as st

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(page_title="5 - Mise en production et industrialisation", layout="wide")

st.title("5️⃣ Mise en production et industrialisation")
st.markdown("Cette section illustre comment le projet est passé d’un prototype local à un système complet, automatisé et déployable en production.")

tabs = st.tabs([
    "5.1 API FastAPI",
    "5.2 Airflow / Cron – Automatisation",
    "5.3 CI/CD – GitHub Actions",
    "5.4 Docker – Conteneurisation",
    "5.5 Prometheus / Grafana"
])

# ============================================================
# 5.1 FASTAPI
# ============================================================
with tabs[0]:
    st.header("5.1 Création d’une API pour exposer les prédictions (FastAPI)")
    st.image("https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png", width=1000)

    st.markdown("### 🎯 Objectif")
    st.markdown("""
    - Rendre le modèle de Machine Learning **accessible via une API REST**  
    - Permettre les **prédictions à la demande** depuis un autre service  
    - Standardiser la communication entre les composants du projet
    """)

    st.markdown("### ⚙️ Technologies")
    st.markdown("""
    - **FastAPI** : Framework rapide et moderne  
    - **Pydantic** : Validation stricte des entrées  
    - **Uvicorn** : Serveur ASGI haute performance  
    - **PostgreSQL** : Stockage des données  
    """)

    st.markdown("### 🔑 Avantages")
    st.markdown("""
    - Documentation interactive (Swagger / ReDoc)  
    - Validation automatique des entrées   
    """)

    st.success("💡 En résumé : FastAPI transforme le modèle ML en un **service web robuste et prêt pour la production.**")

    with st.expander("🔍 Voir la démo Swagger intégrée"):
        st.components.v1.iframe("http://localhost:8000/docs", height=600, scrolling=True)


    st.markdown("### 🧩 Exemple de requête pour `/predict`")

    example_json = """{
  "open_price": 60000.0,
  "high_price": 60500.0,
  "low_price": 59500.0,
  "volume_base": 123.45,
  "volume_quote": 7412345.67,
  "price_lag_1h": 59850.0,
  "volume_lag_1h": 110.0,
  "price_lag_2h": 60120.0,
  "volume_lag_2h": 98.7,
  "price_lag_3h": 59780.0,
  "volume_lag_3h": 105.2,
  "price_lag_4h": 60210.0,
  "volume_lag_4h": 99.9,
  "price_lag_5h": 59990.0,
  "volume_lag_5h": 101.1,
  "rolling_mean_24h": 60050.0,
  "rolling_mean_72h": 59800.0,
  "rsi": 52.3,
  "macd_diff": -12.34,
  "atr": 450.6,
  "hour_of_day": 13,
  "day_of_week": 1
}"""

    st.code(example_json, language="json")





# ============================================================
# 5.2 AIRFLOW / CRON
# ============================================================
with tabs[1]:
    st.header("5.2 Automatisation et ordonnancement (Airflow / Cronjobs)")
    st.image("https://airflow.apache.org/images/feature-image.png", width=1000)

    st.markdown("### 🎯 Objectif")
    st.markdown("""
    - **Planifier et automatiser** les tâches de collecte de données, d’entraînement et de prédiction  
    - Garantir la **régularité et la fiabilité** du pipeline ML  
    - Supprimer les interventions manuelles
    """)

    st.markdown("### 🧠 Évolution du projet")
    st.markdown("""
    - D’abord : un **cronjob** lançant les script de mise à jour (Recuperation data et Machine Learnig) 
    - Puis migration vers **Apache Airflow** pour gérer les dépendances et la planification
    """)

    st.markdown("### 🧩 DAGs principaux")
    st.code("""
    ├── initial_load.py          # Chargement initial des données
    ├── crypto_hourly_update.py  # Mise à jour horaire des données toute les heures 
    └── crypto_monthly_train.py  # Réentraînement mensuel des modèles toute les mois
    """, language="bash")

    st.markdown("### ⚙️ Avantages")
    st.markdown("""
    - Visualisation des exécutions passées  
    - Gestion des erreurs et des dépendances  
    """)
    st.success("💡 Airflow a permis de passer d’un script simple à un **pipeline automatisé et supervisé.**")


    with st.expander("🔍 Voir la démo airflow"):
        st.components.v1.iframe("http://localhost:8080", height=600, scrolling=True)


# ============================================================
# 5.3 CI/CD
# ============================================================
with tabs[2]:
    st.header("5.3 CI/CD et pipeline de déploiement (GitHub Actions)")
    st.image("https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png", width=500)

    st.markdown("### ⚙️ Objectif")
    st.markdown("""
    - **Automatiser les tests**  
    - Garantir la **qualité et la stabilité** du code à chaque mise à jour  
    - Accélérer les itérations sans intervention manuelle
    """)

    st.markdown("### 🔄 Étapes du pipeline GitHub Actions")
    st.code("""
    1️⃣ Exécution des tests unitaires (pytest)
    2️⃣ Build des images Docker (api, ml, data)
    """, language="bash")

    st.markdown("### 🚀 Bénéfices")
    st.markdown("""
    - Moins d’erreurs humaines  
    - Déploiements plus rapides  
    - Code testé automatiquement avant production   
    """)
    st.success("💡 Le CI/CD assure un **pipeline robuste, supervisé et prêt à évoluer vers le cloud.**")

# ------------------------------------------------------------
st.divider()
st.caption("📘 Page de présentation – Partie 5 : Mise en production et industrialisation")


# ============================================================
# 5.4 DOCKER
# ============================================================
with tabs[3]:
    st.header("5.4 Conteneurisation et reproductibilité (Docker)")
    st.image("https://www.docker.com/wp-content/uploads/2022/03/Moby-logo.png", width=500)

    st.markdown("### 🧱 Pourquoi Docker ?")
    st.markdown("""
    - **Uniformiser les environnements** : fini le “ça marche chez moi mais pas chez toi”  
    - **Isoler chaque composant** : API, ML, base de données, dashboard, orchestrateur  
    - **Assurer la portabilité** sur tout type de machine  
    - **Faciliter le déploiement automatisé** 
    """)


    st.markdown("### 📦 Composants conteneurisés")
    st.table({
        "Composant": ["API FastAPI", "Pipeline Data", "Pipeline ML", "Streamlit", "Airflow", "Postgres"],
        "Rôle": ["Servir les prédictions", "Collecter les données", "Entraîner les modèles", "Visualiser les résultats", "Orchestrer les tâches", "Stocker les données"]
    })

    st.markdown("### 🚀 Bénéfices obtenus")
    st.markdown("""
    - Lisibilité et modularité du code  
    - Reproductibilité totale de l’environnement  
    - Automatisation des déploiements  
    """)
    st.success("💡 Docker est le socle de la **reproductibilité et de la scalabilité** du projet.")

    # ============================================================
# 5.3 MONITORING
# ============================================================
with tabs[4]:
    st.header("5.5 Monitoring et supervision (Prometheus / Grafana)")
    st.image("https://miro.medium.com/v2/resize:fit:1400/0*gG72vnJnD0WXFnPQ.png", width=1000)

    st.markdown("### 📊 Objectif")
    st.markdown("""
    - Surveiller les **performances système et applicatives**  
    - Alerter en cas d'erreurs ou de surconsommation  
    - Visualiser en temps réel l'état du pipeline
    """)

    st.markdown("### 🧠 Stack technique")
    st.table({
        "Outil": ["Prometheus", "Grafana", "cAdvisor", "Postgres Exporter", "FastAPI Metrics"],
        "Rôle": [
            "Collecte des métriques de tous les services",
            "Visualisation des métriques en temps réel",
            "Suivi CPU/RAM des conteneurs",
            "Suivi base de données PostgreSQL",
            "Exposition des stats API (requêtes, temps de réponse)"
        ]
    })

    st.markdown("### 🚀 Bénéfices")
    st.markdown("""
    - Diagnostic rapide des erreurs et pics d'usage  
    - Anticipation des pannes  
    - Support au déploiement et à la maintenance  
    """)
    st.success("💡 Grâce à Prometheus et Grafana, chaque service est **observable, mesurable et contrôlable.**")