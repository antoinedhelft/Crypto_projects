# Pitch Entretien (2-3 minutes)

## Objectif du projet

Construire une plateforme data crypto bout-en-bout, orientée fiabilité opérationnelle :

- ingestion horaire automatisée des données de marché
- stockage propre dans PostgreSQL
- orchestration avec Airflow
- exposition API et observabilité

Le focus est data engineering. La partie ML sert de consommateur concret des données produites.

## Problème traité

Les données de marché crypto arrivent en continu et doivent être :

- historisées correctement
- mises à jour sans doublons
- exploitables pour des usages downstream (prédiction, dashboard, monitoring)

## Architecture en une phrase

Airflow orchestre des containers dédiés (ingestion initiale, update incrémental, training), PostgreSQL centralise les données, FastAPI expose les résultats et Streamlit permet la démonstration.

## Choix techniques que je défends

1. Prédiction en variation relative (`next_close_pct_change`) plutôt qu'en prix absolu.
2. Isolation Airflow dans Docker à cause de la contrainte `SQLAlchemy < 2`.
3. Dépendances unifiées via `uv` (`pyproject.toml` + `uv.lock`) pour la reproductibilité.

## Ce que j'ai industrialisé

1. Exécution locale reproductible avec `uv`.
2. Stack complète démarrable via `docker compose`.
3. Tests unitaires/intégration API exécutables en CI.
4. Checks qualité de données générés à chaque cycle incrémental.
5. Endpoint `/status` enrichi avec `data_quality` pour l'exploitation.
6. Test d'intégration Airflow concret sur les contrats runtime des DAGs.

## Limites actuelles assumées

1. Pas encore de tests d'intégration Streamlit orientés parcours utilisateur.
2. Les tests Airflow sont des contrats runtime, pas une exécution complète scheduler + workers.

## Plan d'amélioration court terme

1. Ajouter un test Airflow E2E avec exécution réelle d'un DAG sur un dataset de test.
2. Ajouter un test Streamlit d'intégration sur les appels API clés.
3. Publier un tableau de bord de santé pipeline (latence, volumétrie, erreurs).

## Script Démo 60 Secondes

"J'ai construit une plateforme data crypto orientée production, pas juste un notebook ML. L'ingestion est orchestrée par Airflow, les données sont stockées dans PostgreSQL, puis consommées par une API FastAPI et une interface Streamlit.

Le point clé côté modélisation est que je prédis une variation relative (`next_close_pct_change`) au lieu d'un prix absolu, ce qui permet d'utiliser un seul modèle sur des actifs d'échelles très différentes comme BTC et ETH.

Côté industrialisation, j'ai un workflow reproductible avec `uv`, des tests API en CI, des tests d'intégration Airflow sur les contrats runtime des DAGs, et des contrôles qualité de données qui tournent à chaque update incrémental.

Enfin, l'endpoint `/status` expose à la fois les versions de modèles, la fraîcheur des données et l'état qualité (`data_quality`), ce qui donne une vision exploitable en production." 