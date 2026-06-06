import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import shutil


def _build_dynamic_labels(df_features):
    """Construit les labels de classification avec des seuils ATR-dynamiques par paire.

    Probleme du seuil fixe +-0.5% :
    - Sur BTC (faible volatilite relative), +-0.5% est un mouvement significatif.
    - Sur une altcoin volatile, +-0.5% est du bruit et classe la plupart des mouvements
      en Hausse/Baisse alors que c est de la fluctuation normale.
    - Resultat : desequilibre de classes (64% Stable) et mauvaise generalisation inter-paires.

    Solution ATR-dynamique :
    - Le seuil est proportionnel a atr_pct = ATR / close * 100, soit la volatilite
      relative de chaque bougie.
    - Un multiple de 0.5x l ATR% median par paire definit la zone "Stable".
    - Ainsi une altcoin avec 2% d ATR moyen aura un seuil de +-1%, tandis que BTC
      avec 0.3% d ATR moyen aura un seuil de +-0.15%.
    - Objectif : obtenir ~30-40% de Stable au lieu de 64%, equilibrant les 3 classes.
    """
    price_change = ((df_features['target_price'] - df_features['close_price'])
                    / df_features['close_price'] * 100)

    # Calcul du seuil par paire : 0.5 x ATR% median de la paire.
    # La mediane est preferee a la moyenne pour etre robuste aux spikes de volatilite.
    atr_median_by_symbol = df_features.groupby('symbol')['atr_pct'].median() * 0.5
    thresholds = df_features['symbol'].map(atr_median_by_symbol)

    y = pd.Series(1, index=df_features.index, dtype=int)  # defaut = Stable
    y[price_change < -thresholds] = 0  # Baisse
    y[price_change >  thresholds] = 2  # Hausse

    return y


def train_classifier(df_features, features_path, model_path, train_mask=None):
    """Entrainement de la classification LightGBM sans fuite : reglage sur train uniquement,
    evaluation sur le test en attente, puis reajustement sur toutes les donnees pour le modele enregistre.
    """

    # Exclusion des colonnes de data leakage et des colonnes sources des targets.
    # symbol_cat (entier encode de la paire) est inclus : LightGBM peut apprendre
    # des patterns specifiques a chaque paire sans risque de leakage.
    features_clf = [col for col in df_features.columns if col not in [
        'symbol', 'target_price', 'target_pct', 'close_price'
    ]]
    X = df_features[features_clf]

    # Labels avec seuils ATR-dynamiques (voir _build_dynamic_labels pour le raisonnement)
    y = _build_dynamic_labels(df_features)

    # Alignement des index apres dropna eventuels
    mask = y.notna()
    X, y = X[mask], y[mask]

    print(f"[DEBUG] Features classification ({len(features_clf)}): {features_clf}")
    print(f"[DEBUG] Distribution classes: {y.value_counts().sort_index().to_dict()}")

    # Sauvegarde de la liste des features pour l inference
    with open(str(features_path), 'w') as f:
        json.dump(features_clf, f)

    # Separation temporelle pour eviter le leakage : 80% train / 20% test en ordre chronologique
    if train_mask is not None:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]
    else:
        split_point = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split_point], y.iloc[:split_point]
        X_test, y_test = X.iloc[split_point:], y.iloc[split_point:]

    # Recherche d hyperparametres avec validation croisee temporelle.
    # class_weight='balanced' compense le desequilibre residuel entre Baisse/Hausse/Stable.
    param_dist_clf = {
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63],
        'max_depth': [10, 20],
        'class_weight': ['balanced'],
    }

    lgbm_clf = lgb.LGBMClassifier(random_state=42, n_jobs=1, verbose=-1)
    tscv = TimeSeriesSplit(n_splits=3)
    random_search_clf = RandomizedSearchCV(
        lgbm_clf,
        param_dist_clf,
        n_iter=10,
        scoring='f1_macro',
        cv=tscv,
        verbose=0,
        n_jobs=-1,
        random_state=42,
    )
    random_search_clf.fit(X_train, y_train)

    print(f"Meilleurs parametres de classification: {random_search_clf.best_params_}")
    best_clf_model = random_search_clf.best_estimator_

    # Evaluation sur le jeu de test (OOS = out-of-sample, donnees jamais vues)
    y_pred = best_clf_model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=['Baisse', 'Stable', 'Hausse'],
        output_dict=True,
    )
    print("Rapport de classification final:")
    print(classification_report(y_test, y_pred, target_names=['Baisse', 'Stable', 'Hausse']))

    # Refit final sur l ensemble des donnees (train + test) pour maximiser
    # la quantite de donnees du modele mis en production
    final_model = lgb.LGBMClassifier(**best_clf_model.get_params())
    final_model.fit(X, y)
    joblib.dump(final_model, str(model_path))

    return {
        "features": features_clf,
        "report": report,
        "test_index": list(X_test.index),
        "y_pred_test": [int(v) for v in y_pred],  # 0=Baisse, 1=Stable, 2=Hausse
    }
