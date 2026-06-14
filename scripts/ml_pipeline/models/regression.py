import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import joblib
import json

def train_regressor(df_features, features_path, model_path, train_mask=None):
    """Entraînement de la régression LightGBM optimisée pour prédire target_pct (% changement prix).
    
    === Design Decisions ===
    
    **Features unifiées avec classification** :
    - Inclut atr_pct : capture volatilité relative → utile pour prédire amplitude suivante.
    - Réduit redondance architecture (avant : features différentes selon modèle).
    
    **Objectif MAE/Huber** :
    - Robuste à la distribution de target_pct (peut avoir des queues longues).
    - 'huber' est aussi testé pour réduire l'impact des outliers extrêmes.
    
    **Regularisation (reg_alpha, reg_lambda)** :
    - L1 (alpha) : favorise la parcimonie (certaines features = 0).
    - L2 (lambda) : pénalise les gros poids (smoothness).
    - Gridé dans RandomizedSearchCV pour éviter l'overfitting.
    
    **Métriques (MAE + RMSE + R² + baseline + direction)** :
    - MAE : erreur médiane (robuste, interpretable en %).
    - RMSE : détecte outliers (si RMSE >> MAE → gros écarts rares).
    - R² : variance expliquée (0 = modèle nul, 1 = parfait).
    - Baseline persistance : prédire 0% de variation (close[t+1] ~= close[t]).
    - Direction accuracy : part des signes correctement prédits (hausse/baisse).
    
    **n_iter=10 sur hyperparams** :
    - Equité avec autres modèles (avant : n_iter=5 insuffisant).
    - RandomizedSearchCV : 10 combos aléatoires d'hyperparams testées sur 3 folds CV.
    """

    # Caractéristiques (exclusion des colonnes de data leakage)
    features_reg = [col for col in df_features.columns if col not in ['symbol', 'target_price', 'target_pct', 'close_price']]
    X = df_features[features_reg]
    # Prédire la variation % plutôt que le prix absolu : scale-indépendant entre paires
    y = df_features['target_pct']

    print(f"[DEBUG] Features ({len(features_reg)}): {features_reg}")

    # Liste des caractéristiques persistantes
    with open(str(features_path), 'w') as f:
        json.dump(features_reg, f)

    # Séparation temporelle avant le entrainement pour éviter les fuites (utiliser un masque externe si fourni)
    if train_mask is not None:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]
    else:
        # Fallback: TimeSeriesSplit pour garantir la séparation temporelle
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, test_idx in tscv.split(X):
            pass  # Garder les indices du dernier fold
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        print(f"[DEBUG] Fallback TimeSeriesSplit: {len(train_idx)} train / {len(test_idx)} test")

    # Recherche des hyperparamètres avec RandomizedSearchCV
    # Aligné avec la classification : même nombre d'iterations (n_iter=10)
    param_dist = {
        'objective': ['mae', 'huber'],
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63],
        'max_depth': [10, 20],
        'reg_alpha': [0.0, 0.1, 0.5],
        'reg_lambda': [0.0, 0.1, 0.5],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0],
        'min_child_samples': [20, 50, 100],
    }

    lgbm_reg = lgb.LGBMRegressor(
        # Alternative : objective='quantile', alpha=0.5 pour médiane (encore plus robuste outliers),
        # mais moins documenté dans LightGBM et coûts calcul plus élevés.
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    tscv = TimeSeriesSplit(n_splits=3)
    random_search_reg = RandomizedSearchCV(
        lgbm_reg,
        param_dist,
        n_iter=10,  # Augmenté de 5 à 10 pour égalité avec classificateur
        scoring='neg_mean_absolute_error',
        cv=tscv,
        verbose=0,
        n_jobs=-1,
        random_state=42,
    )
    random_search_reg.fit(X_train, y_train)

    print(f"Meilleurs paramètres de régression(MAE): {random_search_reg.best_params_}")
    best_reg_model = random_search_reg.best_estimator_  # already refit on TRAIN

    # Evaluation sur le test (OOS = out-of-sample)
    y_pred = best_reg_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Détecte les outliers (puissance 2)

    # Baseline naive: persistance => target_pct predit = 0.0
    y_base = np.zeros(len(y_test), dtype=float)
    mae_base = mean_absolute_error(y_test, y_base)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_base))

    # Exactitude directionnelle (on ignore les vrais 0 rares, classes quasi absentes)
    direction_acc = float((np.sign(y_pred) == np.sign(y_test)).mean())
    
    print(f"Erreur absolue moyenne (MAE en %): {mae:.4f}")
    print(f"Erreur quadratique (RMSE en %): {rmse:.4f}")
    print(f"Coefficient de détermination (R2): {r2:.4f}")
    print(f"Baseline MAE (0%): {mae_base:.4f}")
    print(f"Baseline RMSE (0%): {rmse_base:.4f}")
    print(f"Direction accuracy: {direction_acc:.4f}")
    if mae_base > 0:
        gain_vs_baseline = (mae_base - mae) / mae_base * 100.0
        print(f"Gain vs baseline (MAE): {gain_vs_baseline:.2f}%")
    
    # Alerte si RMSE >> MAE : indique des outliers importants
    if rmse > mae * 1.5:
        print(f"[WARN] RMSE/MAE ratio = {rmse/mae:.2f} → présence d'outliers significatifs")

    # Refit sur toutes les données (train + test) pour maximiser la quantité de données
    final_model = lgb.LGBMRegressor(**best_reg_model.get_params())
    final_model.fit(X, y)

    # Enregistrer le modèle final
    joblib.dump(final_model, str(model_path))

    return {
        "mae": mae,
        "rmse": rmse,
        "mae_baseline": mae_base,
        "rmse_baseline": rmse_base,
        "direction_accuracy": direction_acc,
        "r2": r2,
        "features": features_reg
    }
