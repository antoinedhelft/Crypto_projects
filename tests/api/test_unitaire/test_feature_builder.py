"""Tests unitaires pour le constructeur de features."""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta, timezone

from api.feature_builder import compute_indicators

def _create_synthetic_dataframe(n_rows: int = 100) -> pd.DataFrame:
    """Génère un DataFrame de bougies synthétiques pour les tests."""
    now = datetime.now(timezone.utc)
    timestamps = [now - timedelta(hours=i) for i in range(n_rows)][::-1]
    
    data = {
        "timestamp": timestamps,
        "open_price": np.random.uniform(40000, 50000, n_rows),
        "high_price": np.random.uniform(50000, 51000, n_rows),
        "low_price": np.random.uniform(39000, 40000, n_rows),
        "close_price": np.random.uniform(40000, 50000, n_rows),
        "volume_base": np.random.uniform(100, 1000, n_rows),
        "volume_quote": np.random.uniform(5e6, 50e6, n_rows),
    }
    return pd.DataFrame(data)

@pytest.mark.unitaire
def test_compute_indicators_creates_all_columns():
    """
    Vérifie que `compute_indicators` ajoute toutes les colonnes de features attendues
    à un DataFrame de bougies.
    """
    input_df = _create_synthetic_dataframe(n_rows=120) # Assez de données pour les fenêtres de 72h
    
    # Exécuter la fonction à tester
    features_df = compute_indicators(input_df.copy())

    # Liste des colonnes attendues
    expected_new_columns = {
        "rolling_mean_24h",
        "rolling_mean_72h",
        "rsi",
        "macd_diff",
        "atr",
        "hour_of_day",
        "day_of_week",
    }
    # Ajouter les colonnes de lag
    for i in range(1, 6):
        expected_new_columns.add(f"price_lag_{i}h")
        expected_new_columns.add(f"volume_lag_{i}h")

    # Vérifier que toutes les colonnes attendues sont bien présentes
    assert expected_new_columns.issubset(features_df.columns)

    # Vérifier qu'il n'y a pas de valeurs infinies (un problème courant avec les divisions)
    assert not np.isinf(features_df.select_dtypes(include=np.number)).any().any()
