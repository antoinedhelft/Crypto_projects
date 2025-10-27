"""Fichier de configuration global pour les tests pytest."""

import pytest


@pytest.fixture(scope="session")
def client():
    """
    Fixture pour le client de test FastAPI.

    L'import est fait à l'intérieur pour éviter d'imposer FastAPI/httpx
    comme dépendance pour les tests qui n'en ont pas besoin (ex: tests Airflow).
    """
    try:
        from fastapi.testclient import TestClient
        from api.main import app

        return TestClient(app)
    except ImportError as e:
        pytest.skip(f"Impossible d'importer le client de test FastAPI: {e}")
