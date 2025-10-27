"""Tests unitaires pour le module de configuration de l'API."""

import importlib
from pathlib import Path
import pytest

@pytest.mark.unitaire
def test_models_dir_default_path(monkeypatch):
    """
    Vérifie que `MODELS_DIR` pointe par défaut vers `algo_crypto/` à la racine du projet.
    """
    # S'assurer que la variable d'environnement n'est pas définie pour ce test
    monkeypatch.delenv("MODELS_DIR", raising=False)

    # Recharger le module settings pour prendre en compte l'absence de la variable d'env
    from api import settings
    importlib.reload(settings)

    # Le chemin de base est le parent du dossier 'api'
    expected_path = Path(settings.BASE_DIR) / "algo_crypto"
    assert settings.MODELS_DIR == expected_path

@pytest.mark.unitaire
def test_models_dir_override_by_env(monkeypatch, tmp_path):
    """
    Vérifie que la variable d'environnement `MODELS_DIR` surcharge le chemin par défaut.
    """
    custom_path = tmp_path / "my_custom_models"
    custom_path.mkdir()

    # Définir la variable d'environnement
    monkeypatch.setenv("MODELS_DIR", str(custom_path))

    # Recharger le module settings
    from api import settings
    importlib.reload(settings)

    assert settings.MODELS_DIR == custom_path
