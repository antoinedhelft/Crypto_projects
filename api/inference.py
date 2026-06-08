from typing import Dict, Any, List
import joblib
import json
import numpy as np
from pathlib import Path
from functools import lru_cache
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .feature_builder import latest_feature_row, fetch_history, list_available_symbols
from .settings import MODELS_DIR, DEFAULT_REG_MODEL, DEFAULT_CLF_MODEL
import re
from datetime import datetime, timezone

TS_REGEX = re.compile(r"_(\d{8}_\d{4})")

CLASS_NAMES = ["Baisse", "Stable", "Hausse"]  # 0, 1, 2
SYMBOL_REGEX = re.compile(r"^[A-Z0-9]{3,20}$")


def _latest_file(glob_pattern: str):
    """Retourne le fichier le plus recent base sur le timestamp dans son nom."""
    candidates = list(MODELS_DIR.glob(glob_pattern))
    if not candidates:
        return None

    def sort_key(p: Path):
        match = TS_REGEX.search(p.name)
        if match:
            return match.group(1)
        return str(int(p.stat().st_mtime))

    return sorted(candidates, key=sort_key, reverse=True)[0]


def _get_latest_reg_artifacts():
    reg_model = _latest_file("crypto_regressor_lgbm_*.joblib") or (MODELS_DIR / DEFAULT_REG_MODEL)
    reg_feats = _latest_file("regressor_features_*.json") or (MODELS_DIR / "regressor_features.json")
    return reg_model, reg_feats


def _get_latest_clf_artifacts():
    clf_model = _latest_file("crypto_classifier_lgbm_*.joblib") or (MODELS_DIR / DEFAULT_CLF_MODEL)
    clf_feats = _latest_file("classifier_features_*.json") or (MODELS_DIR / "classifier_features.json")
    return clf_model, clf_feats


def _load_deployed_registry() -> dict | None:
    path = MODELS_DIR / "deployed_model.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _load_latest_data_quality() -> dict | None:
    dq_file = MODELS_DIR / "data_quality_latest.json"
    if not dq_file.exists():
        dq_file = _latest_file("data_quality_*.json")
    if not dq_file or not dq_file.exists():
        return None
    with open(dq_file, "r") as f:
        return json.load(f)


_router = APIRouter()


def _normalize_symbol(symbol: str) -> str:
    """Nettoie et valide un symbole crypto avant usage."""
    normalized = symbol.strip().upper()
    if not SYMBOL_REGEX.fullmatch(normalized):
        raise ValueError("Symbol must contain only uppercase letters and digits (3-20 chars)")
    return normalized


def _to_native(value):
    """Convertit les scalaires numpy en types Python natifs."""
    try:
        import numpy as _np
        if isinstance(value, _np.generic):
            return value.item()
    except Exception:
        pass
    return value


@lru_cache(maxsize=8)
def load_model(path: str):
    return joblib.load(path)


@lru_cache(maxsize=4)
def load_features(path: str):
    with open(path, "r") as f:
        return json.load(f)


def get_model_paths():
    """Verifie et retourne les 4 artefacts necessaires pour une prediction complete."""
    reg_path = None
    clf_path = None
    try:
        registry = _load_deployed_registry()
    except Exception:
        registry = None

    if registry and isinstance(registry, dict):
        deployed = registry.get("deployed") or {}
        reg_name = deployed.get("regressor")
        clf_name = deployed.get("classifier")
        if reg_name:
            candidate = MODELS_DIR / reg_name
            if candidate.exists():
                reg_path = candidate
        if clf_name:
            candidate = MODELS_DIR / clf_name
            if candidate.exists():
                clf_path = candidate

    if reg_path is None:
        reg_path, _ = _get_latest_reg_artifacts()
    if clf_path is None:
        clf_path, _ = _get_latest_clf_artifacts()

    # Les listes de features restent calées sur les artefacts les plus récents.
    reg_feat_path = _latest_file("regressor_features_*.json") or (MODELS_DIR / "regressor_features.json")
    clf_feat_path = _latest_file("classifier_features_*.json") or (MODELS_DIR / "classifier_features.json")

    for p in [reg_path, clf_path, reg_feat_path, clf_feat_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
    return reg_path, clf_path, reg_feat_path, clf_feat_path


def _predict_one(symbol: str) -> dict:
    """Logique de prediction factorisee, utilisee par /predict/{symbol} et /predict/batch."""
    symbol = _normalize_symbol(symbol)
    reg_path, clf_path, reg_feat_path, clf_feat_path = get_model_paths()

    reg_model = load_model(str(reg_path))
    clf_model = load_model(str(clf_path))

    feat_data = latest_feature_row(symbol)

    # Regression : predit target_pct (variation % du close suivant)
    reg_pred_pct = float(reg_model.predict([feat_data["regressor_vector"]])[0])

    # Classification : predit la direction (0=Baisse, 1=Stable, 2=Hausse)
    if hasattr(clf_model, "predict_proba"):
        clf_proba = clf_model.predict_proba([feat_data["classifier_vector"]])[0].tolist()
        clf_pred_idx = int(np.argmax(clf_proba))
        confidence = round(max(clf_proba) * 100, 1)
    else:
        clf_pred_idx = int(_to_native(clf_model.predict([feat_data["classifier_vector"]])[0]))
        clf_proba = None
        confidence = None

    return {
        "symbol": symbol,
        "timestamp": feat_data["timestamp"],
        "asof": feat_data.get("asof_timestamp"),
        # Nom du fichier modele utilise : permet de tracer quelle version a fait la prediction
        "model_version": {
            "regressor": reg_path.name,
            "classifier": clf_path.name,
        },
        "deployment": _load_deployed_registry(),
        "prediction": {
            "next_close_pct_change": round(reg_pred_pct, 4),
            "direction": CLASS_NAMES[clf_pred_idx],
            "confidence": confidence,
            "probabilities": {
                CLASS_NAMES[i]: round(clf_proba[i] * 100, 1)
                for i in range(len(CLASS_NAMES))
            } if clf_proba else None,
        },
    }


@_router.get("/predict/{symbol}")
def predict_symbol(symbol: str):
    """Prediction pour un symbole : variation % du prix et direction (Baisse/Stable/Hausse)."""
    try:
        return _predict_one(symbol)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class BatchRequest(BaseModel):
    symbols: List[str]


@_router.post("/predict/batch")
def predict_batch(request: BatchRequest):
    """Prediction pour plusieurs symboles en un seul appel.

    Plus efficace que N appels /predict/{symbol} depuis Streamlit car les modeles
    sont charges une seule fois en cache (lru_cache).
    """
    results = []
    errors = []
    if not request.symbols:
        raise HTTPException(status_code=400, detail="symbols cannot be empty")
    if len(request.symbols) > 20:
        raise HTTPException(status_code=400, detail="symbols batch is limited to 20 items")

    seen = set()
    for raw_symbol in request.symbols:
        try:
            symbol = _normalize_symbol(raw_symbol)
            if symbol in seen:
                continue
            seen.add(symbol)
        except ValueError as e:
            errors.append({"symbol": raw_symbol, "error": str(e)})
            continue
        try:
            results.append(_predict_one(symbol))
        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})

    return {"predictions": results, "errors": errors}


@_router.get("/symbols")
def available_symbols():
    """Liste les paires de cryptos présentes en base."""
    try:
        symbols = list_available_symbols()
        return {"symbols": symbols, "count": len(symbols)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@_router.get("/status")
def status(symbol: str = "BTCUSDT"):
    """Expose un statut exploitable en prod: version modele, metriques, fraicheur data.

    Pourquoi ce endpoint est plus utile que /models:
    - /models donnait juste nom/taille/date mtime (peu actionnable)
    - /status donne les infos utiles a l'exploitation: qualite et fraicheur
    """
    try:
        reg_path, clf_path, _, _ = get_model_paths()

        # Chercher le dernier fichier metrics_*.json pour les KPI de qualite
        metrics_file = _latest_file("metrics_*.json")
        metrics = None
        if metrics_file and metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

        # Fraicheur des donnees: derniere bougie disponible pour un symbole de reference
        try:
            df = fetch_history(symbol, hours=1)
            last_ts = datetime.fromisoformat(str(df.iloc[-1]["timestamp"]).replace("Z", "+00:00"))
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            lag_hours = round((now - last_ts).total_seconds() / 3600, 2)
            freshness = {
                "symbol": symbol,
                "last_candle": last_ts.isoformat(),
                "hours_since_update": lag_hours,
            }
        except Exception:
            freshness = None

        data_quality = None
        try:
            data_quality_report = _load_latest_data_quality()
            if data_quality_report is None:
                data_quality = {
                    "available": False,
                    "reason": "no_data_quality_report_found",
                }
            else:
                data_quality = {
                    "available": True,
                    "report": data_quality_report,
                }
        except Exception:
            data_quality = {
                "available": False,
                "reason": "data_quality_report_unreadable",
            }

        model_deployment = None
        try:
            model_deployment = _load_deployed_registry()
        except Exception:
            model_deployment = None

        return {
            "models": {
                "regressor": reg_path.name,
                "classifier": clf_path.name,
            },
            "metrics": metrics,
            "data_freshness": freshness,
            "data_quality": data_quality,
            "model_deployment": model_deployment,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
