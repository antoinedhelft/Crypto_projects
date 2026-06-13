"""Tests d'integration des endpoints HTTP de l'API.

Ces tests verifient le format attendu des reponses API, sans dependre
ni d'une base reelle ni des artefacts modeles, grace au monkeypatch.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from api import inference


@pytest.mark.integration
def test_health_endpoint(client):
    """L'endpoint /health doit etre accessible."""
    # Appeler l'endpoint.
    response = client.get("/health")
    # Verifier le resultat.
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.integration
def test_status_endpoint(client, monkeypatch):
    """L'endpoint /status doit renvoyer une charge utile operationnelle."""

    # Remplacer les appels externes par des valeurs fixes pour un test fiable.
    class _FakePath:
        def __init__(self, name):
            self.name = name

        def exists(self):
            return True

    monkeypatch.setattr(
        inference,
        "get_model_paths",
        lambda: (_FakePath("reg.joblib"), _FakePath("rf.json")),
    )
    monkeypatch.setattr(inference, "_latest_file", lambda pattern: None)
    monkeypatch.setattr(
        inference,
        "fetch_history",
        lambda symbol, hours=1: pd.DataFrame([{"timestamp": datetime.now(timezone.utc).isoformat()}]),
    )
    monkeypatch.setattr(
        inference,
        "_load_deployed_registry",
        lambda: {
            "deployed": {
                "regressor": "reg.joblib",
                "trained_at": "2026-01-01T00:00:00",
                "score": 0.12,
            },
            "last_candidate": {
                "regressor": "reg_prev.joblib",
                "trained_at": "2025-12-01T00:00:00",
                "score": 0.10,
            },
        },
    )

    # Appeler l'endpoint.
    response = client.get("/status?symbol=BTCUSDT")
    # Verifier que les champs importants sont presents et corrects.
    assert response.status_code == 200
    data = response.json()
    assert data["models"]["regressor"] == "reg.joblib"
    assert data["data_freshness"]["symbol"] == "BTCUSDT"
    assert "data_quality" in data
    assert "model_deployment" in data
    assert data["model_deployment"]["deployed"]["regressor"] == "reg.joblib"


@pytest.mark.integration
def test_available_symbols_endpoint(client, monkeypatch):
    """L'endpoint /symbols doit exposer les paires disponibles dans la base."""
    # Simuler une liste de symboles disponible.
    monkeypatch.setattr(inference, "list_available_symbols", lambda: ["BTCUSDT", "ETHUSDT"])

    # Appeler l'endpoint.
    response = client.get("/symbols")
    # Verifier le contenu retourne.
    assert response.status_code == 200
    data = response.json()
    assert data["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert data["count"] == 2


@pytest.mark.integration
def test_predict_symbol_endpoint(client, monkeypatch):
    """GET /predict/{symbol} doit utiliser la logique _predict_one."""
    # Remplacer le calcul reel par une reponse de prediction simple.
    monkeypatch.setattr(
        inference,
        "_predict_one",
        lambda symbol: {
            "symbol": symbol,
            "timestamp": "2026-01-01T01:00:00+00:00",
            "asof": "2026-01-01T00:00:00+00:00",
            "current_candle": {"open": 100.0, "close": 101.0},
            "model_version": {"regressor": "reg.joblib"},
            "prediction": {
                "next_close_pct_change": 0.42,
                "next_close_predicted": 101.4242,
            },
        },
    )

    # Appeler l'endpoint.
    response = client.get("/predict/BTCUSDT")
    # Verifier la reponse.
    assert response.status_code == 200
    assert response.json()["symbol"] == "BTCUSDT"
    assert "next_close_predicted" in response.json()["prediction"]


@pytest.mark.integration
def test_predict_batch_endpoint_partial_errors(client, monkeypatch):
    """POST /predict/batch doit retourner predictions + errors en cas d'echec partiel."""

    # Simuler un cas mixte: un symbole valide et un symbole en erreur.
    def _fake_predict(symbol):
        if symbol == "BAD":
            raise ValueError("unknown symbol")
        return {
            "symbol": symbol,
            "timestamp": "2026-01-01T01:00:00+00:00",
            "asof": "2026-01-01T00:00:00+00:00",
            "current_candle": {"open": 100.0, "close": 101.0},
            "model_version": {"regressor": "reg.joblib"},
            "prediction": {
                "next_close_pct_change": 0.1,
                "next_close_predicted": 101.101,
            },
        }

    monkeypatch.setattr(inference, "_predict_one", _fake_predict)

    # Appeler l'endpoint.
    response = client.post("/predict/batch", json={"symbols": ["BTCUSDT", "BAD"]})
    # Verifier que la reponse contient a la fois succes et erreurs.
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["symbol"] == "BTCUSDT"
    assert len(data["errors"]) == 1
    assert data["errors"][0]["symbol"] == "BAD"


@pytest.mark.integration
def test_predict_batch_endpoint_validates_inputs(client, monkeypatch):
    """Les symboles invalides doivent etre rejetes clairement."""
    # Simuler une prediction minimale.
    monkeypatch.setattr(inference, "_predict_one", lambda symbol: {"symbol": symbol})

    # Appeler l'endpoint.
    response = client.post("/predict/batch", json={"symbols": ["BTC/USDT", "btc usdt", "BTCUSDT"]})
    # Verifier que les entrees invalides sont bien listees en erreur.
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["symbol"] == "BTCUSDT"
    assert len(data["errors"]) == 2
