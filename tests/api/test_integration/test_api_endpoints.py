"""Tests d'integration des endpoints HTTP de l'API.

Ces tests verifient les contrats HTTP des routes exposees, sans dependre
ni d'une base reelle ni des artefacts modeles, grace au monkeypatch.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from api import inference


@pytest.mark.integration
def test_health_endpoint(client):
    """L'endpoint /health doit etre accessible."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.integration
def test_status_endpoint(client, monkeypatch):
    """L'endpoint /status doit renvoyer une charge utile operationnelle."""

    class _FakePath:
        def __init__(self, name):
            self.name = name

        def exists(self):
            return True

    monkeypatch.setattr(
        inference,
        "get_model_paths",
        lambda: (_FakePath("reg.joblib"), _FakePath("clf.joblib"), _FakePath("rf.json"), _FakePath("cf.json")),
    )
    monkeypatch.setattr(inference, "_latest_file", lambda pattern: None)
    monkeypatch.setattr(
        inference,
        "fetch_history",
        lambda symbol, hours=1: pd.DataFrame([{"timestamp": datetime.now(timezone.utc).isoformat()}]),
    )

    response = client.get("/status?symbol=BTCUSDT")
    assert response.status_code == 200
    data = response.json()
    assert data["models"]["regressor"] == "reg.joblib"
    assert data["models"]["classifier"] == "clf.joblib"
    assert data["data_freshness"]["symbol"] == "BTCUSDT"


@pytest.mark.integration
def test_available_symbols_endpoint(client, monkeypatch):
    """L'endpoint /symbols doit exposer les paires disponibles dans la base."""
    monkeypatch.setattr(inference, "list_available_symbols", lambda: ["BTCUSDT", "ETHUSDT"])

    response = client.get("/symbols")
    assert response.status_code == 200
    data = response.json()
    assert data["symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert data["count"] == 2


@pytest.mark.integration
def test_predict_symbol_endpoint(client, monkeypatch):
    """GET /predict/{symbol} doit utiliser la logique _predict_one."""
    monkeypatch.setattr(
        inference,
        "_predict_one",
        lambda symbol: {
            "symbol": symbol,
            "timestamp": "2026-01-01T01:00:00+00:00",
            "asof": "2026-01-01T00:00:00+00:00",
            "model_version": {"regressor": "reg.joblib", "classifier": "clf.joblib"},
            "prediction": {
                "next_close_pct_change": 0.42,
                "direction": "Hausse",
                "confidence": 67.5,
                "probabilities": {"Baisse": 10.0, "Stable": 22.5, "Hausse": 67.5},
            },
        },
    )

    response = client.get("/predict/BTCUSDT")
    assert response.status_code == 200
    assert response.json()["symbol"] == "BTCUSDT"
    assert response.json()["prediction"]["direction"] == "Hausse"


@pytest.mark.integration
def test_predict_batch_endpoint_partial_errors(client, monkeypatch):
    """POST /predict/batch doit retourner predictions + errors en cas d'echec partiel."""

    def _fake_predict(symbol):
        if symbol == "BAD":
            raise ValueError("unknown symbol")
        return {
            "symbol": symbol,
            "timestamp": "2026-01-01T01:00:00+00:00",
            "asof": "2026-01-01T00:00:00+00:00",
            "model_version": {"regressor": "reg.joblib", "classifier": "clf.joblib"},
            "prediction": {
                "next_close_pct_change": 0.1,
                "direction": "Stable",
                "confidence": 50.0,
                "probabilities": {"Baisse": 25.0, "Stable": 50.0, "Hausse": 25.0},
            },
        }

    monkeypatch.setattr(inference, "_predict_one", _fake_predict)

    response = client.post("/predict/batch", json={"symbols": ["BTCUSDT", "BAD"]})
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["symbol"] == "BTCUSDT"
    assert len(data["errors"]) == 1
    assert data["errors"][0]["symbol"] == "BAD"


@pytest.mark.integration
def test_predict_batch_endpoint_validates_inputs(client, monkeypatch):
    """Les symboles invalides doivent etre rejetes clairement."""
    monkeypatch.setattr(inference, "_predict_one", lambda symbol: {"symbol": symbol})

    response = client.post("/predict/batch", json={"symbols": ["BTC/USDT", "btc usdt", "BTCUSDT"]})
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["symbol"] == "BTCUSDT"
    assert len(data["errors"]) == 2
