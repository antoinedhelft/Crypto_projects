# Projet Crypto Data Engineering

Ce projet met en place une chaîne data complète autour de données de marché crypto : ingestion horaire depuis Binance, stockage PostgreSQL, orchestration Airflow, entraînement mensuel de modèles ML, exposition des prédictions via FastAPI et visualisation via Streamlit.

L'objectif principal est data engineering : automatiser un flux fiable, rejouable et observable. La partie ML et les interfaces servent à exploiter les données produites par la plateforme, pas à remplacer le coeur pipeline.

## Ce que montre le projet

- ingestion initiale et incrémentale de données OHLCV dans PostgreSQL
- orchestration Airflow avec DAGs de chargement initial, mise à jour horaire et entraînement mensuel
- séparation des composants par service Docker : base, orchestrateur, API, UI, pipelines
- exposition d'une API FastAPI avec endpoint de santé, prédiction et dérive
- monitoring applicatif via `/metrics` côté API
- exécution locale reproductible via `uv` et `docker compose`

## Architecture

1. Le pipeline d'ingestion initialise les paires crypto et charge l'historique disponible.
2. Un DAG Airflow met à jour les bougies chaque heure et garantit que le chargement initial a bien été réalisé.
3. Un pipeline ML entraîne des modèles partagés entre plusieurs paires en prédisant une variation relative du prix, pas un prix absolu.
4. L'API charge les artefacts produits et expose les prédictions ainsi que des indicateurs de fraîcheur et de dérive.
5. Streamlit fournit une interface de démonstration au-dessus des données et des prédictions.

## Choix techniques importants

- La cible du modèle est `next_close_pct_change` et non un prix brut. Cela permet d'entraîner un seul modèle sur plusieurs paires avec des échelles très différentes comme BTCUSDT et ETHUSDT.
- Airflow reste isolé dans son image Docker. Airflow 2.9.2 dépend de `SQLAlchemy < 2`, alors que l'API et les pipelines utilisent `SQLAlchemy >= 2`.
- `pyproject.toml` et `uv.lock` sont la source unique de vérité pour les dépendances. Les anciens `requirements.txt` ont été retirés.

## Démarrage rapide

Pré-requis : Python 3.12, `uv`, Docker Desktop.

1. Clonez le dépôt.
2. Créez un fichier `.env` à partir de `.env.example`.
3. Créez l'environnement local :

```powershell
uv venv --python 3.12
.\.venv\Scripts\Activate
uv sync --group dev --group api --group data_pipeline --group ml_pipeline --group streamlit
```

4. Générez les secrets Airflow si besoin :

```powershell
python -c "import secrets; print(secrets.token_urlsafe(64))"
python -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
```

- La 2e commande génère une clé Fernet valide (à mettre dans `.env`) sans dépendre de `cryptography` en local.
- Alternative : `docker compose run --rm airflow-webserver python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.
- `docker compose run` peut être exécuté avant `docker compose up` : il lance un conteneur éphémère juste pour la commande.

5. Construisez les images et démarrez la stack :

```powershell
docker compose --profile images build --pull
docker compose up -d --build
```

En cas de souci de cache Docker sur les images Airflow/uv :

```powershell
docker compose build --no-cache --pull airflow-init airflow-webserver airflow-scheduler
```

## Accès aux services

- FastAPI : `http://localhost:8000/docs`
- Airflow : `http://localhost:8080`
- Streamlit : `http://localhost:8501`

## Commandes utiles

### Local avec uv

```powershell
uv run pytest -q
uv run python main.py
```

### Rebuild ciblé

- Si vous modifiez l'API ou ses dépendances : `docker compose build api ; docker compose up -d api`
- Si vous modifiez l'image Airflow : `docker compose build airflow-webserver airflow-scheduler airflow-init ; docker compose up -d airflow-webserver airflow-scheduler`
- Si vous modifiez le pipeline data ou ML :

```powershell
docker build -t crypto_data_pipeline:latest -f docker/Dockerfile.data_pipeline .
docker build -t crypto_ml_pipeline:latest -f docker/Dockerfile.ml_pipeline .
```

- Si vous modifiez seulement les DAGs Airflow montés en volume : `docker compose up -d airflow-webserver airflow-scheduler`

### Base PostgreSQL

```powershell
docker exec -it pg__crypto_example psql -U crypto -d crypto_trading
```

## Arrêt et nettoyage

- Arrêt simple : `docker compose down`
- Redémarrage : `docker compose up -d`
- Reset complet du projet : `docker compose down --volumes --remove-orphans`

Pour reconstruire entièrement les images :

```powershell
docker compose build --no-cache
docker compose up -d
```

Évitez `docker system prune -a --volumes` sauf si vous voulez purger l'ensemble de votre environnement Docker local.