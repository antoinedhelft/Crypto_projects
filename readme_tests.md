# Documentation des Tests

Ce document explique l'organisation et l'utilisation des tests pour le projet JUIL25-BDE-CRYPTO.

## Structure des Tests

```
tests/
├── conftest.py                    # Configuration globale pytest (fixtures)
├── api/
│   ├── test_unitaire/
│   │   ├── test_settings.py       # Tests de configuration
│   │   ├── test_drift.py          # Tests du calcul de dérive (PSI)
│   │   ├── test_feature_builder.py # Tests de création de features
│   │   └── test_inference.py      # Tests du module d'inférence
│   ├── test_integration/
│   │   └── test_api_endpoints.py  # Tests des endpoints HTTP
│   └── test_e2e/
│       └── test_api_with_real_infrastructure.py # Tests E2E avec Docker
├── airflow/
│   ├── test_unitaire/
│   │   └── test_dags_structure.py # Validation de la structure des DAGs
│   ├── test_integration/
│   │   └── test_airflow_execution.py # Tests d'exécution (placeholder)
│   └── test_e2e/
│       └── test_airflow_data_pipeline.py # Tests E2E de remplissage BDD
└── streamlit/
    ├── test_unitaire/
    │   └── test_streamlit_logic.py # Tests de logique Streamlit
    └── test_integration/
        └── test_streamlit_app.py   # Tests d'intégration Streamlit (placeholder)
```

## Marqueurs pytest

Les tests sont organisés avec trois marqueurs principaux :

### `@pytest.mark.unitaire`
- **Tests rapides** (<1 seconde par test)
- **Sans dépendances externes** : pas de DB, pas d'API live, pas d'artefacts
- **Isolés** : testent une fonction ou une classe spécifique
- **Exemples** :
  - Calcul du PSI (drift)
  - Création de features à partir d'un DataFrame synthétique
  - Validation de la configuration

### `@pytest.mark.integration`
- **Tests plus lents** (plusieurs secondes)
- **Avec dépendances légères** : mocks, artefacts, mais pas d'infrastructure complète
- **Bout-en-bout partiel** : testent l'interaction entre plusieurs composants
- **Exemples** :
  - Appels HTTP aux endpoints de l'API (sans BDD réelle)
  - Validation de la structure des DAGs Airflow

### `@pytest.mark.e2e`
- **Tests très lents** (10-20 minutes)
- **Infrastructure complète** : nécessite `docker compose up` avec tous les services
- **Bout-en-bout complet** : teste le système entier en conditions réelles
- **Exemples** :
  - API qui lit la vraie BDD PostgreSQL remplie par Airflow
  - Vérification de l'intégrité des données dans la BDD
  - Prédictions pour tous les symboles en production

## Commandes Utiles

### Lancer tous les tests
`pytest`

### Lancer uniquement les tests unitaires
`pytest -v -m unitaire`

### Lancer uniquement les tests d'intégration
`pytest -v -m integration`

### Lancer les tests d'un composant spécifique

# API uniquement
`pytest -v tests/api/`

# Airflow uniquement
`pytest -v tests/airflow/`

# Streamlit uniquement
`pytest -v tests/streamlit/`


### Lancer les tests avec rapport de couverture
`pytest -v -m unitaire --cov=api --cov-report=term`

### Lancer les tests en mode verbeux avec traceback court
`pytest -v --tb=short`

### Lancer un test spécifique
`pytest -v tests/api/test_unitaire/test_drift.py::test_psi_on_identical_distributions`

## Couverture des Tests

### API (`tests/api/`)

#### Tests Unitaires
- Configuration (`settings.py`)
- Calcul de dérive PSI (`drift.py`)
- Construction de features (`feature_builder.py`)
- Module d'inférence (`inference.py`)

#### Tests d'Intégration
- Endpoint `/health`
- Endpoint `/models`
- Endpoint `/predict_price_only`
- Endpoint `/predict_direction_only`
- Gestion des erreurs (features manquantes)

### Airflow (`tests/airflow/`)

#### Tests Unitaires
- Import des DAGs sans erreur
- Validation de la structure des DAGs
- Détection de cycles dans les dépendances

#### Tests d'Intégration
- À implémenter selon vos besoins

### Streamlit (`tests/streamlit/`)

#### Tests Unitaires
- Existence du fichier `app.py`
- Validation de la syntaxe

#### Tests d'Intégration
- À implémenter selon vos besoins

## Configuration pytest

Le fichier `pytest.ini` définit :

```ini
[pytest]
pythonpath = .
norecursedirs = .git .venv .cryptenv algo_crypto old_tests
testpaths = tests
markers =
    unitaire: Tests unitaires, rapides et sans dépendances externes
    integration: Tests d'intégration, plus lents, peuvent nécessiter des services
```

## Pipeline CI (GitHub Actions)

La pipeline CI (`.github/workflows/ci.yml`) exécute les tests en plusieurs jobs :

### Job 1: Tests Unitaires (prioritaire)
- Lance tous les tests marqués `@pytest.mark.unitaire`
- Génère un rapport de couverture
- **Bloquant** : si ces tests échouent, la pipeline s'arrête

### Job 2: Tests d'Intégration API
- Lance les tests d'intégration de l'API
- Les tests nécessitant des artefacts sont automatiquement skippés en CI

### Job 3: Tests Airflow
- Installe Airflow avec contraintes
- Valide la structure des DAGs

### Job 4: Tests Streamlit
- Vérifie la syntaxe et la logique de l'application

### Job 5: Build Docker
- Compile les images Docker
- Effectue des smoke tests de base

### Job 6: Résumé CI
- Affiche un résumé des résultats
- Échoue si les tests unitaires ou les builds Docker échouent

## Bonnes Pratiques

### 1. Tests Unitaires
- **Isolez** : Chaque test doit être indépendant
- **Mockez** : Utilisez `monkeypatch` ou `unittest.mock` pour les dépendances externes
- **Soyez rapides** : Un test unitaire ne doit pas dépasser 1 seconde
- **Nommez clairement** : `test_<fonction>_<scénario>_<résultat_attendu>`

**Exemple :**
```python
@pytest.mark.unitaire
def test_psi_on_identical_distributions():
    """Le PSI doit être proche de 0 pour des distributions identiques."""
    reference = np.random.normal(0, 1, 1000)
    current = reference.copy()
    psi = population_stability_index(reference, current)
    assert psi < 0.01
```

### 2. Tests d'Intégration
- **Préparez** : Utilisez des fixtures pour créer l'état initial
- **Testez le comportement réel** : Pas de mocks sur les composants testés
- **Vérifiez les contrats** : Validez les schémas de réponse
- **Gérez les artefacts** : Utilisez `pytest.mark.skipif` si des ressources sont absentes

**Exemple :**
```python
@pytest.mark.integration
@REQUIRES_ARTIFACTS
def test_predict_price_only_endpoint(client):
    """L'endpoint /predict_price_only doit retourner un prix prédit."""
    payload = build_payload_from_features()
    response = client.post("/predict_price_only", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()
```

### 3. Fixtures
Utilisez `conftest.py` pour partager des fixtures entre tests :

```python
@pytest.fixture(scope="session")
def client():
    """Client de test FastAPI."""
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)
```

## Debugging

### Afficher les prints pendant les tests
`pytest -v -s`

### Arrêter au premier échec
`pytest -v -x`

### Lancer le débogueur sur échec
`pytest -v --pdb`

### Voir les warnings
`pytest -v --tb=short -W all`


