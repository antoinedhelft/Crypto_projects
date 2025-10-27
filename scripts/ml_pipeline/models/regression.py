import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json

def train_regressor(df_features, features_path, model_path, train_mask=None):
    """Entraînement de la régression LightGBM sans fuite de données : réglage sur l'entraînement uniquement, 
    évaluation sur le test en attente.
    Renvoie des métriques et enregistre un modèle réajusté (entraîné sur des données complètes) pour l'inférence.
    """

    # Caractéristiques (exclusion des colonnes de data leakage)
    features_reg = [col for col in df_features.columns if col not in ['symbol', 'target_price', 'close_price']]
    X = df_features[features_reg]
    y = df_features['target_price']

    print(f"[DEBUG] Features ({len(features_reg)}): {features_reg}")

    # Liste des caractéristiques persistantes
    with open(str(features_path), 'w') as f:
        json.dump(features_reg, f)

    # Séparation temporelle avant le entrainement pour éviter les fuites (utiliser un masque externe si fourni)
    if train_mask is not None:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]
    else:
        split_point = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split_point], y.iloc[:split_point]
        X_test, y_test = X.iloc[split_point:], y.iloc[split_point:]

    # recherche des hyperparamètres avec RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63],
        'max_depth': [10, 20],
    }

    lgbm_reg = lgb.LGBMRegressor(objective='mae', random_state=42, n_jobs=1, verbose=-1)
    tscv = TimeSeriesSplit(n_splits=3)
    random_search_reg = RandomizedSearchCV(
        lgbm_reg,
        param_dist,
        n_iter=5,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        verbose=0,
        n_jobs=-1,
        random_state=42,
    )
    random_search_reg.fit(X_train, y_train)

    print(f"Meilleurs paramètres de régression(MAE): {random_search_reg.best_params_}")
    best_reg_model = random_search_reg.best_estimator_  # already refit on TRAIN

    # Evaluation sur le test
    y_pred = best_reg_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Erreur absolue moyenne (MAE): {mae:.4f}")
    print(f"Coefficient de détermination (R2): {r2:.4f}")

    # Optionnel: refit sur toutes les données pour la sauvegarde
    final_model = lgb.LGBMRegressor(**best_reg_model.get_params())
    final_model.fit(X, y)

    # Enregistrer le modèle final
    joblib.dump(final_model, str(model_path))

    return {"mae": mae, "r2": r2, "features": features_reg}
