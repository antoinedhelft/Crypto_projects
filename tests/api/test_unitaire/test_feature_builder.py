"""Tests unitaires pour la construction des features d'inference."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from api import feature_builder


def _create_synthetic_dataframe(n_rows: int = 120) -> pd.DataFrame:
    """Genere un DataFrame OHLCV synthetique avec chronologie coherente."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    timestamps = [now - timedelta(hours=i) for i in range(n_rows)][::-1]

    # Serie de prix simple et stable pour eviter des artefacts numeriques
    base = np.linspace(60000, 62000, n_rows)
    data = {
        "timestamp": timestamps,
        "open_price": base,
        "high_price": base + 100,
        "low_price": base - 100,
        "close_price": base + np.sin(np.linspace(0, 8, n_rows)) * 50,
        "volume_base": np.linspace(100, 300, n_rows),
        "volume_quote": np.linspace(6_000_000, 9_000_000, n_rows),
    }
    return pd.DataFrame(data)


@pytest.mark.unitaire
def test_latest_feature_row_builds_vectors_with_expected_order(monkeypatch):
    """latest_feature_row doit produire des vecteurs alignes avec les listes de features."""
    df = _create_synthetic_dataframe()

    clf_feats = ["price_lag_1h", "rsi", "symbol_cat", "hour_of_day"]
    reg_feats = ["price_lag_2h", "macd_diff", "symbol_cat", "day_of_week"]

    monkeypatch.setattr(feature_builder, "load_feature_lists", lambda: (clf_feats, reg_feats))
    monkeypatch.setattr(feature_builder, "load_symbol_map", lambda: {"BTCUSDT": 7})
    monkeypatch.setattr(feature_builder, "fetch_history", lambda symbol, hours=80: df.copy())

    row = feature_builder.latest_feature_row("BTCUSDT")

    assert row["classifier_features"] == clf_feats
    assert row["regressor_features"] == reg_feats
    assert len(row["classifier_vector"]) == len(clf_feats)
    assert len(row["regressor_vector"]) == len(reg_feats)

    # Le code symbole doit etre injecte dans symbol_cat
    clf_symbol_idx = clf_feats.index("symbol_cat")
    reg_symbol_idx = reg_feats.index("symbol_cat")
    assert row["classifier_vector"][clf_symbol_idx] == 7.0
    assert row["regressor_vector"][reg_symbol_idx] == 7.0


@pytest.mark.unitaire
def test_latest_feature_row_sets_timestamp_plus_one_hour(monkeypatch):
    """Le timestamp predit doit etre asof + 1 heure."""
    df = _create_synthetic_dataframe()

    feats = ["price_lag_1h", "rsi", "symbol_cat"]
    monkeypatch.setattr(feature_builder, "load_feature_lists", lambda: (feats, feats))
    monkeypatch.setattr(feature_builder, "load_symbol_map", lambda: {"ETHUSDT": 3})
    monkeypatch.setattr(feature_builder, "fetch_history", lambda symbol, hours=80: df.copy())

    row = feature_builder.latest_feature_row("ETHUSDT")

    asof = datetime.fromisoformat(row["asof_timestamp"])
    pred = datetime.fromisoformat(row["timestamp"])
    assert pred - asof == timedelta(hours=1)
