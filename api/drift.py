from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from .feature_builder import fetch_history
from scripts.ml_pipeline.feature_engineering import compute_symbol_indicators

def _compute_for_drift(df):
    """Wrapper pour le drift : compute_symbol_indicators sans symbol_cat."""
    df = df.rename(columns={"timestamp": "open_datetime"}) if "timestamp" in df.columns else df
    import pandas as pd
    df["open_datetime"] = pd.to_datetime(df["open_datetime"] if "open_datetime" in df.columns else df.index)
    if "open_datetime" in df.columns:
        df = df.set_index("open_datetime")
    return compute_symbol_indicators(df, symbol_code=0)

router = APIRouter()

class DriftRequest(BaseModel):
	symbol: str
	recent_hours: int = 168  # 7 jours
	baseline_hours: int = 720  # 30 jours
	features: List[str] = []  # si vide, on prendra un set par dÃ©faut
	bins: int = 10

def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
	# PSI pour variables continues: discretiser en bins sur l'union
	expected = expected[~np.isnan(expected)]
	actual = actual[~np.isnan(actual)]
	if len(expected) == 0 or len(actual) == 0:
		return np.nan
	# bornes sur population combinÃ©e pour stabilitÃ©
	combined = np.concatenate([expected, actual])
	quantiles = np.linspace(0, 1, bins + 1)
	edges = np.unique(np.quantile(combined, quantiles))
	# Ã©viter < 2 edges
	if len(edges) < 2:
		return 0.0
	exp_hist, _ = np.histogram(expected, bins=edges)
	act_hist, _ = np.histogram(actual, bins=edges)
	exp_pct = exp_hist / max(exp_hist.sum(), 1)
	act_pct = act_hist / max(act_hist.sum(), 1)
	# Ã©viter divisions par zÃ©ro
	exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
	act_pct = np.where(act_pct == 0, 1e-6, act_pct)
	psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
	return float(psi)

@router.get("/drift/{symbol}")
def drift_get(symbol: str, recent_hours: int = 168, baseline_hours: int = 720, bins: int = 10, non_overlap: bool = False):
	# baseline_hours 720 = la rÃ©fÃ©rence 30 jours
	# recent_hours 168 = pÃ©riode rÃ©cente Ã  surveiller 7 jours
	# Population Stability Index : score de dÃ©rive entre deux distributions
	try:
		# Charger historique nÃ©cessaire
		if non_overlap:
			# FenÃªtres non chevauchantes: baseline = fenÃªtre juste avant la fenÃªtre rÃ©cente
			need = recent_hours + baseline_hours
			df_all = fetch_history(symbol, hours=need)
			df_all = _compute_for_drift(df_all)
			if len(df_all) >= need:
				# baseline: les baseline_hours avant la fenÃªtre rÃ©cente
				df_base = df_all.iloc[-(recent_hours + baseline_hours):-recent_hours]
				# recent: les recent_hours derniÃ¨res
				df_recent = df_all.iloc[-recent_hours:]
			else:
				# Pas assez de lignes, on retombe sur le mode chevauchant
				df_recent = fetch_history(symbol, hours=recent_hours)
				df_base = fetch_history(symbol, hours=baseline_hours)
				df_recent = _compute_for_drift(df_recent)
				df_base = _compute_for_drift(df_base)
		else:
			# Mode par dÃ©faut (peut chevaucher): derniÃ¨res N heures pour chaque fenÃªtre
			df_recent = fetch_history(symbol, hours=recent_hours)
			df_base = fetch_history(symbol, hours=baseline_hours)
			df_recent = _compute_for_drift(df_recent)
			df_base = _compute_for_drift(df_base)

		# CaractÃ©ristiques par dÃ©faut
		default_feats = [
			"close_price", "volume_base", "volume_quote",
			"rsi", "macd_diff", "atr",
		]

		report: Dict[str, Any] = {}
		for col in default_feats:
			if col in df_recent.columns and col in df_base.columns:
				psi = population_stability_index(df_base[col].to_numpy(dtype=float), df_recent[col].to_numpy(dtype=float), bins=bins)
				report[col] = {"psi": psi}

		# Score global simple: moyenne des PSI
		values = [v["psi"] for v in report.values() if pd.notnull(v["psi"]) ]
		overall = float(np.mean(values)) if values else None

		resp = {
			"symbol": symbol,
			"recent_hours": recent_hours,
			"baseline_hours": baseline_hours,
			"bins": bins,
			"overall_psi": overall,
			"features": report,
			"notes": "RÃ¨gle (indicative): PSI < 0.1 faible; 0.1-0.25 modÃ©rÃ©; >0.25 important."
		}
		if non_overlap:
			resp["window_mode"] = "non_overlap"
		return resp
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))
