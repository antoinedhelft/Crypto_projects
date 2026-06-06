import pandas as pd
import numpy as np
import ta


def compute_symbol_indicators(df: pd.DataFrame, symbol_code: int = 0) -> pd.DataFrame:
    """Calcule tous les indicateurs pour un DataFrame mono-symbole.

    Cette fonction est la SOURCE DE VÉRITÉ partagée entre :
    - l'entraînement (pipeline_train.py via build_features)
    - l'inférence (feature_builder.py dans l'API)
    Toute modification ici s'applique automatiquement aux deux contextes,
    garantissant que le modèle reçoit exactement les mêmes features à l'entraînement
    et à la prédiction.

    Args:
        df: DataFrame indexé sur open_datetime, colonnes attendues :
            close_price, high_price, low_price, volume_base
        symbol_code: code entier de la paire (issu du symbol_map sauvegardé à l'entraînement)
    """
    # --- Lags de prix (4h) ---
    # 4 lags capturent la dynamique récente sans ajouter de bruit au-delà de 4h.
    for lag in range(1, 5):
        df[f'price_lag_{lag}h'] = df['close_price'].shift(lag)

    # Les lags de volume sont absents intentionnellement :
    # déjà représentés par volume_base courant + ATR (proxy de liquidité).

    # --- Moyennes mobiles ---
    # 24h = tendance intra-journalière, 72h = tendance 3 jours
    df['rolling_mean_24h'] = df['close_price'].rolling(window=24).mean()
    df['rolling_mean_72h'] = df['close_price'].rolling(window=72).mean()

    # --- Indicateurs techniques ---
    # RSI : force relative des variations, plage 0-100 (>70 suracheté, <30 survendu)
    df['rsi'] = ta.momentum.RSIIndicator(df['close_price']).rsi()
    # MACD diff : divergence entre EMA12 et EMA26 — direction et force de tendance
    df['macd_diff'] = ta.trend.MACD(df['close_price']).macd_diff()
    # ATR : amplitude moyenne sur 14 périodes = proxy de volatilité absolue
    df['atr'] = ta.volatility.AverageTrueRange(
        df['high_price'], df['low_price'], df['close_price']
    ).average_true_range()

    # --- Feature catégorielle : code du symbole ---
    # LightGBM utilise ce code pour apprendre des patterns spécifiques à chaque paire
    # (ex: BTC réagit différemment à RSI=70 qu'une altcoin volatile).
    # Le code est assigné alphabétiquement à l'entraînement et sauvegardé dans
    # symbol_map.json pour être réutilisé à l'identique à l'inférence.
    df['symbol_cat'] = symbol_code

    # --- Caractéristiques temporelles ---
    # Patterns horaires (ouverture marchés US/Asie) et hebdomadaires (weekend = faible volume)
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    # --- ATR normalisé ---
    # atr / close * 100 = volatilité relative, comparable entre BTC et altcoins.
    # Utilisé par le classificateur pour calculer des seuils de classe dynamiques par paire.
    df['atr_pct'] = df['atr'] / df['close_price'] * 100

    return df


def build_features(df_raw: pd.DataFrame):
    """Construit le DataFrame de features pour l'entraînement (toutes paires).

    Returns:
        df_features: DataFrame avec toutes les features + targets
        symbol_map: dict {symbol: code} à sauvegarder pour l'inférence
    """
    df_raw['open_datetime'] = pd.to_datetime(df_raw['open_datetime'])
    df_raw.set_index('open_datetime', inplace=True)

    # Mapping alphabétique symbol → code entier.
    # L'ordre alphabétique garantit que le même symbole aura toujours le même code
    # sur des datasets différents, à condition que les paires ne changent pas.
    symbols = sorted(df_raw['symbol'].unique())
    symbol_map = {sym: code for code, sym in enumerate(symbols)}

    df_list = []
    for symbol in symbols:
        df = df_raw[df_raw['symbol'] == symbol].copy()
        df = compute_symbol_indicators(df, symbol_code=symbol_map[symbol])

        # --- Cibles (uniquement à l'entraînement) ---
        # target_price : prix de clôture de la bougie T+1
        df['target_price'] = df['close_price'].shift(-1)
        # target_pct : variation % entre close[T] et close[T+1]
        # Le régresseur prédit target_pct plutôt que target_price pour être
        # scale-indépendant entre paires (BTC à 100k vs altcoin à 0.001€).
        df['target_pct'] = (df['target_price'] - df['close_price']) / df['close_price'] * 100

        df_list.append(df)

    df_features = pd.concat(df_list)
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_features.dropna(inplace=True)

    return df_features, symbol_map
