import os
import sys
import pandas as pd
from pathlib import Path
import json
import datetime

# Ajouter le path racine
sys.path.append(str(Path(__file__).parent.parent))

print("[DEBUG] Pipeline ML démarré")
sys.stdout.flush()

# Remplacer imports relatifs par absolus:
try:
    from scripts.ml_pipeline.data_loading import load_candles_from_db
    from scripts.ml_pipeline.feature_engineering import build_features
    from scripts.ml_pipeline.config import (
        MODEL_REG_PATH, MODEL_CLF_PATH,
        FEATURES_REG_JSON, FEATURES_CLF_JSON, METRICS_JSON, SYMBOL_MAP_JSON
    )
    from scripts.ml_pipeline.models.regression import train_regressor
    from scripts.ml_pipeline.models.classification import train_classifier
    from scripts.ml_pipeline.backtest import backtest_classification
    print("[DEBUG] Imports OK")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def main():
    print("[DEBUG] Début main()")
    sys.stdout.flush()
    
    try:
        print("[DEBUG] Chargement données...")
        sys.stdout.flush()
        df_raw = load_candles_from_db()
        print(f"[DEBUG] {len(df_raw)} lignes chargées")
        sys.stdout.flush()
        
        if df_raw.empty:
            print("[ERROR] Aucune donnée chargée!")
            return
            
        print("[DEBUG] Feature engineering...")
        sys.stdout.flush()
        df_features, symbol_map = build_features(df_raw)
        # Sauvegarder le symbol_map immediatement pour que l'API puisse l'utiliser
        # meme si le training echoue apres cette etape
        with open(str(SYMBOL_MAP_JSON), 'w') as f:
            json.dump(symbol_map, f, indent=2)
        print(f"[DEBUG] symbol_map sauvé: {symbol_map}")
        # Ordonner par symbole puis temps pour construire un split chrono par symbole
        idx_name = df_features.index.name or 'open_datetime'
        df_features = (
            df_features
            .reset_index()
            .rename(columns={df_features.index.name: 'open_datetime'})
            .sort_values(['symbol', 'open_datetime'])
            .set_index('open_datetime')
        )
        print(f"[DEBUG] Features construites: {df_features.shape}")
        print(f"[DEBUG] Colonnes: {list(df_features.columns)}")
        sys.stdout.flush()
        
        # Construire un masque d'entraînement chrono par symbole (~80% par symbole)
        df_tmp = df_features.copy()
        df_tmp['row_idx'] = df_tmp.groupby('symbol').cumcount()
        df_tmp['grp_size'] = df_tmp.groupby('symbol')['symbol'].transform('size')
        df_tmp['train_cut'] = (df_tmp['grp_size'] * 0.8).astype(int)
        train_mask = df_tmp['row_idx'] < df_tmp['train_cut']
        df_tmp = df_tmp.drop(columns=['row_idx', 'grp_size', 'train_cut'])

    # Aucun clipping de la cible. On conserve toute l'amplitude des prix
    # pour éviter de plafonner artificiellement des actifs comme BTCUSDT.
        
        print("[DEBUG] Entraînement régresseur...")
        sys.stdout.flush()
        reg_result = train_regressor(df_features, FEATURES_REG_JSON, MODEL_REG_PATH, train_mask=train_mask)
        
        print("[DEBUG] Entraînement classificateur...")
        sys.stdout.flush()
        clf_result = train_classifier(df_features, FEATURES_CLF_JSON, MODEL_CLF_PATH, train_mask=train_mask)
        
        # Sauvegarde des métriques du run courant
        metrics_new = {
            "trained_at": datetime.datetime.now().isoformat(),
            "regression": {
                "mae_pct": reg_result["mae"],
                "r2": reg_result["r2"],
            },
            "classification": {
                "f1_macro": clf_result["report"]["macro avg"]["f1-score"],
                "accuracy": clf_result["report"]["accuracy"],
            },
        }

        # Comparaison avec les métriques précédentes si elles existent
        previous_metrics_files = sorted(Path(METRICS_JSON).parent.glob("metrics_*.json"))
        if previous_metrics_files:
            latest = previous_metrics_files[-1]
            with open(latest) as f:
                metrics_prev = json.load(f)
            prev_mae = metrics_prev.get("regression", {}).get("mae_pct")
            prev_f1  = metrics_prev.get("classification", {}).get("f1_macro")
            new_mae  = metrics_new["regression"]["mae_pct"]
            new_f1   = metrics_new["classification"]["f1_macro"]
            print(f"[METRICS] Régression MAE%  : {prev_mae:.4f} → {new_mae:.4f} "
                  f"({'✓ amélioration' if new_mae < prev_mae else '✗ dégradation'})")
            print(f"[METRICS] Classification F1: {prev_f1:.4f} → {new_f1:.4f} "
                  f"({'✓ amélioration' if new_f1 > prev_f1 else '✗ dégradation'})")
        else:
            print("[METRICS] Premier entraînement — pas de comparaison possible.")

        with open(str(METRICS_JSON), 'w') as f:
            json.dump(metrics_new, f, indent=2)
        print(f"[METRICS] Métriques sauvées dans {METRICS_JSON}")

        print("[DEBUG] Training terminé avec succès!")
        
    except Exception as e:
        print(f"[ERROR] Exception dans main(): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()