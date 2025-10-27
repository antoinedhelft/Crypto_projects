from sqlalchemy import create_engine
import pandas as pd
from .config import DB_URL

def load_candles_from_db():
    """Charge 4 ans de données crypto depuis Postgres"""
    query = """
    SELECT p.symbol, c.open_datetime, c.open_price, c.high_price, 
           c.low_price, c.close_price, c.volume_base, c.volume_quote
    FROM candlestick c
    JOIN pair p ON p.id = c.pair_id
    WHERE c.open_datetime >= NOW() - INTERVAL '4 years'
    ORDER BY p.symbol, c.open_datetime ASC
    """
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    if df.empty:
        raise ValueError("Aucune donnée trouvée dans la base")
    
    print(f"Données chargées: {len(df)} lignes sur 4 ans")
    return df
