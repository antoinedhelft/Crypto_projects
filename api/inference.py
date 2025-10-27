from typing import Dict, Any
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

def _latest_file(glob_pattern: str):
    """
    Retourne le fichier le plus récent basé sur le timestamp dans son nom.
    Format attendu: *_YYYYMMDD_HHMM.extension
    Si aucun timestamp n'est trouvé, utilise la date de modification du fichier.
    """
    candidates = list(MODELS_DIR.glob(glob_pattern))
    if not candidates:
        return None
    
    def sort_key(p: Path):
        # Essayer d'extraire le timestamp du nom du fichier
        match = TS_REGEX.search(p.name)
        if match:
            return match.group(1)  # Retourne le timestamp comme chaîne (ex: "20251017_1200")
        # Sinon, fallback sur la date de modification du fichier
        return str(int(p.stat().st_mtime))
    
    candidates_sorted = sorted(candidates, key=sort_key, reverse=True)
    return candidates_sorted[0]

def _get_latest_reg_artifacts():
    """Sélectionne le dernier modèle de régression et sa liste de features. Retourne (reg_model_path, reg_features_path)."""
    reg_model = _latest_file("crypto_regressor_lgbm_*.joblib") or (MODELS_DIR / DEFAULT_REG_MODEL)
    reg_feats = _latest_file("regressor_features_*.json") or (MODELS_DIR / "regressor_features.json")
    return reg_model, reg_feats

def _get_latest_clf_artifacts():
    """Sélectionne le dernier modèle de classification et sa liste de features. Retourne (clf_model_path, clf_features_path)."""
    clf_model = _latest_file("crypto_classifier_lgbm_*.joblib") or (MODELS_DIR / DEFAULT_CLF_MODEL)
    clf_feats = _latest_file("classifier_features_*.json") or (MODELS_DIR / "classifier_features.json")
    return clf_model, clf_feats

_router = APIRouter()

def _to_native(value):
    """Convertit les scalaires numpy (np.generic) en types Python natifs (int/float/str)."""
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
    with open(path, 'r') as f:
        return json.load(f)

def get_model_paths():
    """Compat: conserve une version combinée pour l'endpoint /predict/{symbol} qui nécessite les deux modèles.
    Retourne (reg_model, clf_model, reg_feats, clf_feats) et vérifie l'existence des 4 artefacts.
    """
    reg_path, reg_feat_path = _get_latest_reg_artifacts()
    clf_path, clf_feat_path = _get_latest_clf_artifacts()
    for p in [reg_path, clf_path, reg_feat_path, clf_feat_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
    return reg_path, clf_path, reg_feat_path, clf_feat_path

@_router.get("/predict/{symbol}")
def predict_symbol(symbol: str):
    """Prédiction automatique basée sur les dernières données du symbole"""
    try:
        # Charger les modèles et caractéristiques
        reg_path, clf_path, reg_feat_path, clf_feat_path = get_model_paths()
        
        reg_model = load_model(str(reg_path))
        clf_model = load_model(str(clf_path))
        
        # Construire les caractéristiques depuis la BDD
        feat_data = latest_feature_row(symbol)
        
        # Prédiction régression
        reg_pred = reg_model.predict([feat_data["regressor_vector"]])[0]
        
        # Prédiction classification
        class_names = ["Baisse", "Stable", "Hausse"]  # Ordre: 0, 1, 2 (comme dans ton training)
        
        if hasattr(clf_model, "predict_proba"):
            clf_proba = clf_model.predict_proba([feat_data["classifier_vector"]])[0]
            clf_pred_idx = int(np.argmax(clf_proba)) 
            clf_pred_label = class_names[clf_pred_idx]
            clf_proba = clf_proba.tolist()
        else:
            clf_pred_idx = int(clf_model.predict([feat_data["classifier_vector"]])[0])
            clf_pred_label = class_names[clf_pred_idx]
            clf_proba = None
        
        return {
            "symbol": symbol,
            "timestamp": feat_data["timestamp"],
            "asof": feat_data.get("asof_timestamp"),
            "prediction": {
                "next_close_price": round(float(reg_pred), 2),
                "direction": clf_pred_label,
                "confidence": round(max(clf_proba) * 100, 1) if clf_proba else None
            },
            "details": {
                "classification_idx": clf_pred_idx,
                "probabilities": {
                    "Baisse": round(clf_proba[0] * 100, 1) if clf_proba else None,
                    "Stable": round(clf_proba[1] * 100, 1) if clf_proba else None,
                    "Hausse": round(clf_proba[2] * 100, 1) if clf_proba else None
                } if clf_proba else None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@_router.get("/models")
def list_available_models():
    """Liste les modèles disponibles"""
    try:
        models = []
        for f in MODELS_DIR.glob("*.joblib"):
            stat = f.stat()
            models.append({
                "name": f.name,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": stat.st_mtime
            })
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints séparés pour régression et classification
@_router.post("/predict_price_only")
def predict_price_only(data: dict):
    """Prédiction de prix uniquement (régression)"""
    try:
        reg_path, reg_feat_path = _get_latest_reg_artifacts()
        if not reg_path.exists():
            raise FileNotFoundError(f"Required file not found: {reg_path}")
        if not reg_feat_path.exists():
            raise FileNotFoundError(f"Required file not found: {reg_feat_path}")
        reg_feat_list = load_features(str(reg_feat_path))
        
        missing = [f for f in reg_feat_list if f not in data]
        if missing:
            raise HTTPException(400, f"Features manquantes: {missing}")
        
        reg_vector = [data[f] for f in reg_feat_list]
        reg_model = load_model(str(reg_path))
        reg_pred = reg_model.predict([reg_vector])[0]
        
        return {
            "predicted_price": float(reg_pred),
            "model": "regression_only"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@_router.post("/predict_direction_only")
def predict_direction_only(data: dict):
    """Prédiction de direction uniquement (classification)"""
    try:
        clf_path, clf_feat_path = _get_latest_clf_artifacts()
        if not clf_path.exists():
            raise FileNotFoundError(f"Required file not found: {clf_path}")
        if not clf_feat_path.exists():
            raise FileNotFoundError(f"Required file not found: {clf_feat_path}")
        clf_feat_list = load_features(str(clf_feat_path))
        
        missing = [f for f in clf_feat_list if f not in data]
        if missing:
            raise HTTPException(400, f"Features manquantes: {missing}")
        clf_vector = [data[f] for f in clf_feat_list]
        clf_model = load_model(str(clf_path))

        # Mapping souhaité: 0=Baisse, 1=Stable, 2=Hausse
        code_to_label = {0: "Baisse", 1: "Stable", 2: "Hausse"}

        # Prédiction brute (peut être un entier numpy)
        raw_pred = clf_model.predict([clf_vector])[0]
        raw_pred = _to_native(raw_pred)

        # Construire un mapping classes -> libellé lisible
        label_mapping = {}
        if hasattr(clf_model, "classes_"):
            for cls in clf_model.classes_:
                try:
                    label_mapping[cls] = code_to_label.get(int(cls), str(cls))
                except Exception:
                    label_mapping[cls] = str(cls)

        # Libellé final
        if raw_pred in label_mapping:
            pred_label = label_mapping[raw_pred]
        else:
            try:
                pred_label = code_to_label.get(int(raw_pred), str(raw_pred))
            except Exception:
                pred_label = str(raw_pred)

        # Probabilités (si dispo) libellées + confiance
        proba_map = None
        confidence = None
        if hasattr(clf_model, "predict_proba"):
            proba = clf_model.predict_proba([clf_vector])[0]
            if hasattr(clf_model, "classes_"):
                proba_map = {}
                for i, cls in enumerate(clf_model.classes_):
                    lbl = label_mapping.get(cls, code_to_label.get(cls, str(cls)))
                    proba_map[lbl] = round(float(proba[i]) * 100, 2)
            else:
                # Ordre implicite 0,1,2
                proba_map = {
                    "Baisse": round(float(proba[0]) * 100, 2) if len(proba) > 0 else None,
                    "Stable": round(float(proba[1]) * 100, 2) if len(proba) > 1 else None,
                    "Hausse": round(float(proba[2]) * 100, 2) if len(proba) > 2 else None,
                }
            # Confiance = max des probabilités
            confidence = max([v for v in proba_map.values() if v is not None], default=None)

        return {
            "predicted_direction_label": pred_label,
            "confidence_percentage": confidence,
            "probabilities": proba_map,
            "model": "classification_only"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
class CandleData(BaseModel):
    current_price: float
    features: Dict[str, Any]
