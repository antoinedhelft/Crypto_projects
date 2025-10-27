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
        FEATURES_REG_JSON, FEATURES_CLF_JSON, METRICS_JSON
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
        df_features = build_features(df_raw)
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
        
        print("[DEBUG] Training terminé avec succès!")
        
    except Exception as e:
        print(f"[ERROR] Exception dans main(): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()