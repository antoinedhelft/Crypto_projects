import os
import json
from datetime import datetime, timezone, timedelta
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from .settings import MODELS_DIR, DATABASE_URL as API_DATABASE_URL

# Import de la fonction partagee entre entrainement et inference.
# Garantit que les memes features sont calculees des deux cotes.
from scripts.ml_pipeline.feature_engineering import compute_symbol_indicators

DATABASE_URL = API_DATABASE_URL


def _latest_file(dir_path: Path, pattern: str):
    candidates = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_feature_lists():
    """Charge les listes de features alignees avec les derniers artefacts disponibles."""
    clf_path = _latest_file(MODELS_DIR, "classifier_features_*.json") or (MODELS_DIR / "classifier_features.json")
    reg_path = _latest_file(MODELS_DIR, "regressor_features_*.json") or (MODELS_DIR / "regressor_features.json")
    if not clf_path.exists():
        raise FileNotFoundError(f"Fichier non trouve: {clf_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Fichier non trouve: {reg_path}")
    with open(clf_path, "r") as f:
        clf_feats = json.load(f)
    with open(reg_path, "r") as f:
        reg_feats = json.load(f)
    return clf_feats, reg_feats


def load_symbol_map() -> dict:
    """Charge le mapping symbol -> code entier sauvegarde a l entrainement.

    Ce mapping garantit que symbol_cat est identique entre entrainement et inference.
    Si aucun fichier n est trouve (avant le premier entrainement), retourne un dict vide
    et symbol_cat sera 0 par defaut.
    """
    path = _latest_file(MODELS_DIR, "symbol_map_*.json")
    if path is None:
        return {}
    with open(path, "r") as f:
        return json.load(f)


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


def latest_feature_row(symbol: str):
    """Construit le vecteur de features pour la derniere bougie disponible d un symbole.

    Utilise compute_symbol_indicators (partagee avec l entrainement) pour garantir
    la coherence des features entre training et inference.
    """
    clf_feats, reg_feats = load_feature_lists()
    needed = set(clf_feats) | set(reg_feats)

    symbol_map = load_symbol_map()
    symbol_code = symbol_map.get(symbol, 0)

    df = fetch_history(symbol, hours=80)

    # Mise en forme attendue par compute_symbol_indicators : index = open_datetime
    df = df.rename(columns={"timestamp": "open_datetime"})
    df["open_datetime"] = pd.to_datetime(df["open_datetime"])
    df = df.set_index("open_datetime")

    df = compute_symbol_indicators(df, symbol_code=symbol_code)

    last = df.iloc[-1]

    last_ts = last.name  # index = open_datetime
    if hasattr(last_ts, "tzinfo") and last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    next_ts = last_ts + timedelta(hours=1)

    feat_dict = {}
    for col in needed:
        if col not in last.index:
            raise KeyError(f"Missing feature '{col}'; available: {list(last.index)}")
        feat_dict[col] = float(last[col])

    clf_vector = [feat_dict[f] for f in clf_feats]
    reg_vector = [feat_dict[f] for f in reg_feats]

    return {
        "classifier_features": clf_feats,
        "regressor_features": reg_feats,
        "classifier_vector": clf_vector,
        "regressor_vector": reg_vector,
        "asof_timestamp": last_ts.isoformat(),
        "timestamp": next_ts.isoformat(),
    }
