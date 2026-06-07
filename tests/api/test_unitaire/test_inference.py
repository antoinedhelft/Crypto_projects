"""Tests unitaires pour le module d'inférence de l'API."""

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock
import numpy as np

from api import inference

@pytest.mark.unitaire
def test_latest_file(monkeypatch, tmp_path: Path):
    """
    Vérifie que `_latest_file` retourne bien le fichier le plus récent.
    """
    # On fait pointer MODELS_DIR vers notre dossier temporaire
    monkeypatch.setattr(inference, "MODELS_DIR", tmp_path)

    # Créer des fichiers factices
    (tmp_path / "model_20251017_1000.joblib").touch()
    latest_file = tmp_path / "model_20251017_1200.joblib"
    latest_file.touch()
    (tmp_path / "model_20251017_1100.joblib").touch()

    found_file = inference._latest_file("model_*.joblib")
    assert found_file == latest_file

@pytest.mark.unitaire
def test_latest_file_returns_none_if_no_files(monkeypatch, tmp_path: Path):
    """
    Vérifie que `_latest_file` retourne None si aucun fichier ne correspond.
    """
    monkeypatch.setattr(inference, "MODELS_DIR", tmp_path)
    found_file = inference._latest_file("model_*.joblib")
    assert found_file is None

@pytest.mark.unitaire
def test_load_model_is_cached(monkeypatch):
    """
    Vérifie que la fonction `load_model` utilise bien un cache (functools.lru_cache)
    pour ne pas recharger le même modèle plusieurs fois.
    """
    # Créer un mock pour joblib.load
    mock_joblib_load = MagicMock(return_value="fake_model")
    monkeypatch.setattr(inference.joblib, "load", mock_joblib_load)

    # Vider le cache de la fonction avant le test
    inference.load_model.cache_clear()

    # Appeler la fonction plusieurs fois avec le même chemin
    path = Path("/fake/path/model.joblib")
    model1 = inference.load_model(path)
    model2 = inference.load_model(path)

    # joblib.load ne doit avoir été appelé qu'une seule fois
    mock_joblib_load.assert_called_once_with(path)
    assert model1 == "fake_model"
    assert model2 == "fake_model"

@pytest.mark.unitaire
def test_load_features_is_cached(monkeypatch, tmp_path):
    """
    Vérifie que la fonction `load_features` utilise également un cache.
    """
    # Créer un fichier de features factice
    features_path = tmp_path / "features.json"
    features_path.write_text(json.dumps(["feat1", "feat2"]))

    # Vider le cache
    inference.load_features.cache_clear()

    # Appeler plusieurs fois
    feats1 = inference.load_features(features_path)
    feats2 = inference.load_features(features_path)

    # Le fichier ne doit être lu qu'une fois. On ne peut pas directement mocker
    # `open`, mais on peut vérifier que l'objet retourné est le même.
    assert feats1 is feats2
    assert feats1 == ["feat1", "feat2"]


@pytest.mark.unitaire
def test_normalize_symbol_accepts_common_input():
    """Les symboles doivent etre nettoyes puis valides avant usage."""
    assert inference._normalize_symbol(" btcusdt ") == "BTCUSDT"
    assert inference._normalize_symbol("eth123") == "ETH123"


@pytest.mark.unitaire
def test_normalize_symbol_rejects_invalid_values():
    """Les symboles incoherents doivent etre rejetes explicitement."""
    with pytest.raises(ValueError):
        inference._normalize_symbol("BTC/USDT")
    with pytest.raises(ValueError):
        inference._normalize_symbol("")
    with pytest.raises(ValueError):
        inference._normalize_symbol("ab")


@pytest.mark.unitaire
def test_status_returns_operational_payload(monkeypatch):
    """L'endpoint status doit retourner version modele + metriques + date d'entrainement."""
    class _FakePath:
        def __init__(self, name):
            self.name = name
            self._exists = True

        def exists(self):
            return self._exists

    monkeypatch.setattr(
        inference,
        "get_model_paths",
        lambda: (_FakePath("reg.joblib"), _FakePath("clf.joblib"), _FakePath("r.json"), _FakePath("c.json")),
    )
    monkeypatch.setattr(inference, "_latest_file", lambda pattern: None)

    import pandas as pd
    fake_df = pd.DataFrame([{"timestamp": "2026-01-01T00:00:00+00:00"}])
    monkeypatch.setattr(inference, "fetch_history", lambda symbol, hours=1: fake_df)

    payload = inference.status(symbol="BTCUSDT")
    assert payload["models"]["regressor"] == "reg.joblib"
    assert payload["models"]["classifier"] == "clf.joblib"
    assert payload["metrics"] is None
    assert payload["data_freshness"]["symbol"] == "BTCUSDT"

