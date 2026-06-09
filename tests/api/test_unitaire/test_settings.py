"""Tests unitaires pour le module de configuration de l'API."""

import importlib
from pathlib import Path
import pytest

@pytest.mark.unitaire
def test_models_dir_default_path(monkeypatch):
    """
    Vérifie que `MODELS_DIR` pointe par défaut vers `algo_crypto/` à la racine du projet.
    """
    # Supprimer la variable d'environnement pour tester le comportement par defaut.
    monkeypatch.delenv("MODELS_DIR", raising=False)

    # Recharger la configuration.
    from api import settings
    importlib.reload(settings)

    if Path("/app").exists():
        expected_path = Path("/app/algo_crypto")
    else:
        expected_path = Path(settings.__file__).resolve().parents[1] / "algo_crypto"
    # Verifier le chemin obtenu.
    assert settings.MODELS_DIR == expected_path

@pytest.mark.unitaire
def test_models_dir_override_by_env(monkeypatch, tmp_path):
    """
    Vérifie que la variable d'environnement `MODELS_DIR` surcharge le chemin par défaut.
    """
    # Creer un dossier personnalise et le definir dans l'environnement.
    custom_path = tmp_path / "my_custom_models"
    custom_path.mkdir()

    # Définir la variable d'environnement
    monkeypatch.setenv("MODELS_DIR", str(custom_path))

    # Recharger le module pour relire l'environnement.
    from api import settings
    importlib.reload(settings)

    # Verifier que la valeur personnalisee est bien utilisee.
    assert settings.MODELS_DIR == custom_path
