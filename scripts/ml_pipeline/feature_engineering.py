import pandas as pd
import numpy as np
import ta

def build_features(df_raw: pd.DataFrame):

    
    # Permet de prédire le prix de fermetuire de l'heure suivante
    df_raw['open_datetime'] = pd.to_datetime(df_raw['open_datetime'])
    df_raw.set_index('open_datetime', inplace=True)
    df_list = []

    for symbol in df_raw['symbol'].unique():
        df = df_raw[df_raw['symbol'] == symbol].copy()
        
        # Lag et moyennes mobiles (1H)
        ### création de 5 colonnes de lag pour le prix et le volume (valeurs des 1, 2, 3, 4, 5 heures précédentes)
        for lag in range(1, 6):
            df[f'price_lag_{lag}h'] = df['close_price'].shift(lag)
            df[f'volume_lag_{lag}h'] = df['volume_base'].shift(lag)
        ### création de 2 colonnes de moyennes mobiles (24h et 72h)
        df['rolling_mean_24h'] = df['close_price'].rolling(window=24).mean()
        df['rolling_mean_72h'] = df['close_price'].rolling(window=72).mean()

        # Indicateurs techniques
        # Indice de Force Relative : mesure l'ampleur et la rapidité des changemets récents.
        #  > 70 suracheté, < 30 survendu étendue : 0-100
        df['rsi'] = ta.momentum.RSIIndicator(df['close_price']).rsi()
        # Convergence et divergence des moyennes mobiles : Indique la direction et la force de tendance entre 2 moyennes mobiles.
        df['macd_diff'] = ta.trend.MACD(df['close_price']).macd_diff()
        # Average True Range : indique l'ampleur moyenne des mouvements sur une période donnée et donc la volatilité.
        # Un ATR élevé indique une forte volatilité, un ATR faible indique une faible volatilité.
        df['atr'] = ta.volatility.AverageTrueRange(df['high_price'], df['low_price'], df['close_price']).average_true_range()
        
        # Caractéristiques temporelles
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # Cible de régression: le prix de fermeture suivant
        ### création une colonne target_price qui est le prix de clôture de l'heure suivante
        df['target_price'] = df['close_price'].shift(-1)
        
        df_list.append(df)

    df_features = pd.concat(df_list)

    # Nettoyage des valeurs infinies et manquantes
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_features.dropna(inplace=True)

    return df_features