import os, time
from datetime import datetime, timedelta
import pandas as pd
from binance.client import Client
from sqlalchemy import select
from sqlalchemy.orm import Session
from .db_utils import init_db, SessionLocal, engine
from .models import Exchange, Crypto, Pair, Candlestick as Candle

# Ce script fait le chargement initial:
# 1) choix des paires Binance eligibles
# 2) recuperation de 4 ans de bougies
# 3) validation des lignes
# 4) insertion en base

YEARS = 4
INTERVAL = Client.KLINE_INTERVAL_1HOUR
STABLES = {"USDT","USDC","BUSD","DAI","TUSD","PAX","USDP","FDUSD","GUSD"}
TOP_N = 3
client = Client()

def has_min_history(symbol: str, interval: str, years: int) -> bool:
    """Retourne True si le symbole a >= years d'historique disponible pour l'intervalle."""
    cutoff_ms = int((datetime.now() - timedelta(days=years * 365)).timestamp() * 1000)
    # Tente d’utiliser la méthode utilitaire de python-binance (souvent disponible)
    try:
        earliest = getattr(client, "_get_earliest_valid_timestamp")(symbol, interval)
        return earliest <= cutoff_ms
    except Exception:
        # Fallback: teste s’il existe au moins 1 bougie à la date cutoff
        try:
            data = client.get_klines(symbol=symbol, interval=interval, startTime=cutoff_ms, limit=1)
            return len(data) > 0 and int(data[0][0]) <= cutoff_ms
        except Exception:
            return False

def get_top_symbols(limit=TOP_N):
    """Retourne les paires USDT les plus liquides avec historique suffisant."""
    tickers = client.get_ticker()
    candidates = []
    for t in tickers:
        symbol = t['symbol']
        # On ne garde que les paires finissant par USDT
        if not symbol.endswith("USDT"):
            continue
        base = symbol[:-4]
        if base.upper() in STABLES:
            continue
        try:
            vol = float(t.get("quoteVolume", 0))
            if vol > 0:
                candidates.append((symbol, vol))
        except Exception:
            continue
    candidates.sort(key=lambda x: x[1], reverse=True)

    selected = []
    for sym, _ in candidates:
        if has_min_history(sym, INTERVAL, YEARS):
            selected.append(sym)
        else:
            print(f"Skip {sym}: less than {YEARS} years of history.")
        if len(selected) >= limit:
            break

    print(f"Top {limit} paires (non stables) avec >= {YEARS} ans d'historique: {selected}")
    return selected

def ensure_refs(db: Session):
    """Cree les references minimales (exchange/crypto) si elles n'existent pas."""
    exch = db.execute(select(Exchange).where(Exchange.name=="Binance")).scalar_one_or_none()
    if not exch:
        exch = Exchange(name="Binance")
        db.add(exch)
    # Insère quelques cryptos de base (pas obligatoire mais pratique)
    base_cryptos = [("Bitcoin","BTC"),("Ethereum","ETH"),("Tether","USDT")]
    for name,sym in base_cryptos:
        if not db.execute(select(Crypto).where(Crypto.symbol==sym)).scalar_one_or_none():
            db.add(Crypto(name=name, symbol=sym))
    db.commit()
    return exch.id

def ensure_pair(db: Session, exchange_id: int, symbol: str):
    """Cree (si necessaire) la paire pour un symbole et retourne son pair_id."""
    base = symbol[:-4]
    quote = "USDT"
    base_crypto = db.execute(select(Crypto).where(Crypto.symbol==base)).scalar_one_or_none()
    if not base_crypto:
        base_crypto = Crypto(name=base, symbol=base)
        db.add(base_crypto)
        db.commit()
    quote_crypto = db.execute(select(Crypto).where(Crypto.symbol==quote)).scalar_one()
    pair = db.execute(select(Pair).where(Pair.symbol==symbol, Pair.exchange_id==exchange_id)).scalar_one_or_none()
    if not pair:
        pair = Pair(
            exchange_id=exchange_id,
            base_crypto_id=base_crypto.id,
            quote_crypto_id=quote_crypto.id,
            symbol=symbol,
            is_active=True
        )
        db.add(pair)
        db.commit()
    return pair.id

def fetch_symbol(symbol: str, years: int):
    """Recupere l'historique d'un symbole sur `years` annees, mois par mois."""
    end = datetime.now()
    start = end - timedelta(days=years*365)
    cur = start
    frames = []
    while cur < end:
        nxt = (cur.replace(day=1)+timedelta(days=32)).replace(day=1)
        if nxt > end: nxt = end
        kl = client.get_historical_klines(
            symbol, INTERVAL,
            cur.strftime("%d %b %Y %H:%M:%S"),
            nxt.strftime("%d %b %Y %H:%M:%S")
        )
        if kl:
            df = pd.DataFrame(kl, columns=[
                'open_time','open','high','low','close','volume',
                'close_time','quote_asset_volume','number_of_trades',
                'taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'
            ])
            frames.append(df)
            print(symbol, cur.strftime('%Y-%m'), len(df))
        cur = nxt
        time.sleep(0.1)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out.drop_duplicates(subset=['open_time'], inplace=True)
    return out

def insert_candles(db: Session, pair_id: int, df: pd.DataFrame):
    """Valide puis insere les bougies pour une paire.

    Les lignes invalides sont rejetees avant insert et comptabilisees par raison.
    """
    if df is None or df.empty:
        return
    df['open_datetime'] = pd.to_datetime(df['open_time'], unit='ms', utc=True, errors='coerce')
    df['close_datetime'] = pd.to_datetime(df['close_time'], unit='ms', utc=True, errors='coerce')
    rows = []
    rejected = 0
    rejected_reasons: dict[str, int] = {}

    def _reject(reason: str) -> None:
        # Comptage des rejets pour comprendre rapidement pourquoi des lignes sont ignorees.
        nonlocal rejected
        rejected += 1
        rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

    for _, r in df.iterrows():
        try:
            open_dt = r['open_datetime']
            close_dt = r['close_datetime']

            open_price = float(r['open'])
            high_price = float(r['high'])
            low_price = float(r['low'])
            close_price = float(r['close'])
            volume_base = float(r['volume'])
            volume_quote = float(r['quote_asset_volume'])
            trade_count = int(r['number_of_trades'])
            taker_buy_base = float(r['taker_buy_base_asset_volume'])
            taker_buy_quote = float(r['taker_buy_quote_asset_volume'])
        except Exception:
            _reject("parse_error")
            continue

        # Controle 1: timestamps valides et ordonnes.
        if pd.isna(open_dt) or pd.isna(close_dt):
            _reject("invalid_timestamp")
            continue
        if close_dt <= open_dt:
            _reject("invalid_time_order")
            continue

        # Controle 2: coherence des prix OHLC.
        if (
            open_price <= 0
            or high_price <= 0
            or low_price <= 0
            or close_price <= 0
        ):
            _reject("non_positive_price")
            continue
        if high_price < low_price:
            _reject("high_lower_than_low")
            continue
        if high_price < max(open_price, close_price):
            _reject("high_below_open_close")
            continue
        if low_price > min(open_price, close_price):
            _reject("low_above_open_close")
            continue

        # Controle 3: coherence des volumes et du nombre de trades.
        if volume_base < 0 or volume_quote < 0:
            _reject("negative_volume")
            continue
        if trade_count < 0:
            _reject("negative_trade_count")
            continue
        if taker_buy_base < 0 or taker_buy_quote < 0:
            _reject("negative_taker_volume")
            continue
        if taker_buy_base > volume_base + 1e-12:
            _reject("taker_base_gt_total")
            continue
        if taker_buy_quote > volume_quote + 1e-12:
            _reject("taker_quote_gt_total")
            continue

        rows.append(Candle(
            pair_id=pair_id,
            open_datetime=open_dt,
            close_datetime=close_dt,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume_base=volume_base,
            volume_quote=volume_quote,
            trade_count=trade_count,
            taker_buy_base_volume=taker_buy_base,
            taker_buy_quote_volume=taker_buy_quote
        ))

    if rows:
        db.bulk_save_objects(rows)
        db.commit()

    if rejected > 0:
        print(f"Insertion {len(rows)} bougies, rejet {rejected} lignes (pair_id={pair_id}).")
        print(f"Détail rejets: {rejected_reasons}")
    else:
        print(f"Insertion {len(rows)} bougies.")

def main():
    """Point d'entree du chargement initial."""
    # Etape A: creer les tables si absentes.
    init_db()  # Crée les tables à partir de models.py si absentes
    with SessionLocal() as db:
        # Etape B: preparer les references puis determiner les symbols a charger.
        exch_id = ensure_refs(db)
        symbols = get_top_symbols()

        # Etape C: chargement complet pour chaque symbole retenu.
        for sym in symbols:
            pair_id = ensure_pair(db, exch_id, sym)
            df = fetch_symbol(sym, YEARS)
            insert_candles(db, pair_id, df)

if __name__ == "__main__":
    main()