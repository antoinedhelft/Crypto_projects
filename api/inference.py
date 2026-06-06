from typing import Dict, Any, List
import joblib
import json
import numpy as np
from pathlib import Path
from functools import lru_cache
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .feature_builder import latest_feature_row
from .settings import MODELS_DIR, DEFAULT_REG_MODEL, DEFAULT_CLF_MODEL
import re

TS_REGEX = re.compile(r"_(\d{8}_\d{4})")

CLASS_NAMES = ["Baisse", "Stable", "Hausse"]  # 0, 1, 2


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


_router = APIRouter()


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
    reg_path, reg_feat_path = _get_latest_reg_artifacts()
    clf_path, clf_feat_path = _get_latest_clf_artifacts()
    for p in [reg_path, clf_path, reg_feat_path, clf_feat_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
    return reg_path, clf_path, reg_feat_path, clf_feat_path


def _predict_one(symbol: str) -> dict:
    """Logique de prediction factorisee, utilisee par /predict/{symbol} et /predict/batch."""
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
    for symbol in request.symbols:
        try:
            results.append(_predict_one(symbol))
        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})

    return {"predictions": results, "errors": errors}


@_router.get("/models")
def list_available_models():
    """Liste les modeles disponibles avec leur taille et date de modification."""
    try:
        models = []
        for f in MODELS_DIR.glob("*.joblib"):
            stat = f.stat()
            models.append({
                "name": f.name,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": stat.st_mtime,
            })
        return {"models": sorted(models, key=lambda m: m["modified"], reverse=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
