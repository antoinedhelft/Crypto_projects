from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import numpy as np
import pandas as pd

from .feature_builder import fetch_history, load_feature_lists
from scripts.ml_pipeline.feature_engineering import compute_symbol_indicators

# Exclure les colonnes qui ne sont pas des features d'entree du modele.
NON_MODEL_FEATURES = {
    "close_price", "open_price", "high_price", "low_price",
    "target_price", "target_pct", "symbol", "symbol_cat", "volume_quote",
}

router = APIRouter()


def _compute_for_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs de drift avec le meme moteur que l'entrainement."""
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "open_datetime"})
    df["open_datetime"] = pd.to_datetime(df["open_datetime"] if "open_datetime" in df.columns else df.index)
    if "open_datetime" in df.columns:
        df = df.set_index("open_datetime")
    return compute_symbol_indicators(df, symbol_code=0)


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI (Population Stability Index) entre deux distributions."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    edges = np.unique(np.quantile(np.concatenate([expected, actual]), np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0

    exp_hist, _ = np.histogram(expected, bins=edges)
    act_hist, _ = np.histogram(actual, bins=edges)

    exp_pct = exp_hist / max(exp_hist.sum(), 1)
    act_pct = act_hist / max(act_hist.sum(), 1)

    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _status_from_psi(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "warning"
    return "drift"


@router.get("/drift/{symbol}")
def drift_get(
    symbol: str,
    recent_hours: int = 168,
    baseline_hours: int = 720,
    bins: int = 10,
    non_overlap: bool = False,
):
    """Mesure la derive sur les features reellement utilisees par le modele.

    - recent_hours: fenetre recente (par defaut 7 jours)
    - baseline_hours: fenetre de reference (par defaut 30 jours)
    - non_overlap=true: compare 7 derniers jours vs 30 jours precedents (recommande)
    """
    try:
        if non_overlap:
            need = recent_hours + baseline_hours
            df_all = _compute_for_drift(fetch_history(symbol, hours=need))
            if len(df_all) >= need:
                df_base = df_all.iloc[-(recent_hours + baseline_hours):-recent_hours]
                df_recent = df_all.iloc[-recent_hours:]
            else:
                df_recent = _compute_for_drift(fetch_history(symbol, hours=recent_hours))
                df_base = _compute_for_drift(fetch_history(symbol, hours=baseline_hours))
        else:
            df_recent = _compute_for_drift(fetch_history(symbol, hours=recent_hours))
            df_base = _compute_for_drift(fetch_history(symbol, hours=baseline_hours))

        # Features reelles du modele (source de verite = artefacts features_*.json)
        try:
            clf_feats, reg_feats = load_feature_lists()
            model_features = sorted((set(clf_feats) | set(reg_feats)) - NON_MODEL_FEATURES)
        except FileNotFoundError:
            # Fallback si aucun artefact n'existe encore
            model_features = [
                "rsi", "macd_diff", "atr", "atr_pct",
                "rolling_mean_24h", "rolling_mean_72h",
                "price_lag_1h", "price_lag_2h",
                "hour_of_day", "day_of_week", "volume_base",
            ]

        report: Dict[str, Any] = {}
        for col in model_features:
            if col in df_recent.columns and col in df_base.columns:
                psi = _psi(
                    df_base[col].to_numpy(dtype=float),
                    df_recent[col].to_numpy(dtype=float),
                    bins=bins,
                )
                report[col] = {
                    "psi": round(psi, 4),
                    "status": _status_from_psi(psi),
                }

        values = [v["psi"] for v in report.values() if pd.notnull(v["psi"])]
        overall = round(float(np.mean(values)), 4) if values else None
        alert = (
            "unknown" if overall is None
            else "stable" if overall < 0.1
            else "warning" if overall < 0.25
            else "drift - re-entrainement recommande"
        )

        resp = {
            "symbol": symbol,
            "recent_hours": recent_hours,
            "baseline_hours": baseline_hours,
            "bins": bins,
            "overall_psi": overall,
            "alert": alert,
            "features_measured": len(report),
            "features": report,
            "note": "PSI < 0.1 stable | 0.1-0.25 warning | >0.25 drift",
        }
        if non_overlap:
            resp["window_mode"] = "non_overlap"
        return resp
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
