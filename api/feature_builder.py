import os
import math
from datetime import datetime, timezone, timedelta
import pandas as pd
from sqlalchemy import create_engine, text
import json
from pathlib import Path
from .settings import MODELS_DIR, DATABASE_URL as API_DATABASE_URL

DATABASE_URL = API_DATABASE_URL

def _latest_file(dir_path: Path, pattern: str) -> Path | None:
    candidates = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None

def load_feature_lists():
    """Charge les listes de features alignées avec les derniers artefacts disponibles.

    Priorité aux fichiers timestampés (regressor_features_*.json, classifier_features_*.json),
    sinon fallback vers les noms non timestampés.
    """
    clf_path = _latest_file(MODELS_DIR, "classifier_features_*.json") or (MODELS_DIR / "classifier_features.json")
    reg_path = _latest_file(MODELS_DIR, "regressor_features_*.json") or (MODELS_DIR / "regressor_features.json")
    if not clf_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {clf_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {reg_path}")
    with open(clf_path, "r") as f:
        clf_feats = json.load(f)
    with open(reg_path, "r") as f:
        reg_feats = json.load(f)
    return clf_feats, reg_feats

def fetch_history(symbol: str, hours: int = 80) -> pd.DataFrame:
    engine = create_engine(DATABASE_URL, future=True)
    q = text("""
        SELECT p.symbol,
               c.open_datetime AS timestamp,
               c.open_price,
               c.high_price,
               c.low_price,
               c.close_price,
               c.volume_base,
               c.volume_quote
        FROM candlestick c
        JOIN pair p ON p.id = c.pair_id
        WHERE p.symbol = :sym
        ORDER BY c.open_datetime DESC
        LIMIT :lim
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"sym": symbol, "lim": hours})
    if df.empty:
        raise ValueError(f"No data for symbol={symbol}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    price = df["close_price"] if "close_price" in df.columns else df["open_price"]
    df["price"] = price

    # Lags (1..5)
    for k in range(1, 6):
        df[f"price_lag_{k}h"] = df["price"].shift(k)
        df[f"volume_lag_{k}h"] = df["volume_base"].shift(k)

    # Rolling means
    df["rolling_mean_24h"] = df["price"].rolling(24, min_periods=12).mean()
    df["rolling_mean_72h"] = df["price"].rolling(72, min_periods=36).mean()

    # RSI (14)
    delta = df["price"].diff()
    gain = (delta.clip(lower=0)).rolling(14, min_periods=7).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0, math.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD diff (12,26,9)
    ema12 = df["price"].ewm(span=12, adjust=False).mean()
    ema26 = df["price"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_diff"] = macd - signal

    # ATR (14)
    prev_close = df["price"].shift(1)
    tr = pd.concat([
        (df["high_price"] - df["low_price"]),
        (df["high_price"] - prev_close).abs(),
        (df["low_price"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=7).mean()

    # Time features
    ts = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek

    return df

def latest_feature_row(symbol: str):
    clf_feats, reg_feats = load_feature_lists()
    needed = set(clf_feats) | set(reg_feats)

    df = fetch_history(symbol)
    df = compute_indicators(df)

    last = df.iloc[-1]

    # Timestamps
    # On essaye de conserver l'info de fuseau; si naïf, on suppose UTC
    last_ts = pd.to_datetime(last["timestamp"])
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    next_ts = last_ts + timedelta(hours=1)

    # Construction du dictionanire des caractéristiques
    feat_dict = {}
    for col in needed:
        if col not in last:
            raise KeyError(f"Missing feature {col}; available: {list(last.index)}")
        feat_dict[col] = float(last[col]) if pd.api.types.is_numeric_dtype(type(last[col])) else last[col]

    # Ordres des vecteurs
    clf_vector = [feat_dict[f] for f in clf_feats]
    reg_vector = [feat_dict[f] for f in reg_feats]

    return {
        "classifier_features": clf_feats,
        "regressor_features": reg_feats,
        "classifier_vector": clf_vector,
        "regressor_vector": reg_vector,
        # Horodatage de la dernière bougie observée
        "asof_timestamp": last_ts.isoformat(),
        # Horodatage de la bougie prédite (t+1h)
        "timestamp": next_ts.isoformat(),
    }