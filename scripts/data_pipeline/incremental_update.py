from pathlib import Path
from datetime import datetime, timedelta, timezone
import time
import pandas as pd
from binance.client import Client
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from .db_utils import SessionLocal, init_db, engine  
from .models import Pair, Candlestick as Candle, Exchange, Crypto  


YEARS = 4
TOP_N = 3
INTERVAL = Client.KLINE_INTERVAL_1HOUR
# Base assets considérés comme stables (à exclure côté baseAsset pour garder des paires type BTCUSDT, ETHUSDT, ...)
STABLES = {"USDT","USDC","BUSD","DAI","TUSD","PAX","USDP","FDUSD","GUSD","USDE"}

client = Client()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

def log(msg: str):
    print(f"[{datetime.now().isoformat()}] {msg}")

def get_top_symbols(limit=TOP_N):
    """Retourne les 'limit' paires USDT triées par volume, filtrées sur spot TRADING et hors stablecoins.
    Utilise exchangeInfo pour éviter les symboles invalides. Retour: filtre simple par suffixe USDT.
    """
    try:
        info = client.get_exchange_info()
        spot_ok = {
            s["symbol"]
            for s in info.get("symbols", [])
            if s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("isSpotTradingAllowed", True)
            and s.get("baseAsset", "").upper() not in STABLES
        }
    except Exception as e:
        log(f"exchangeInfo indisponible, fallback simple: {e}")
        spot_ok = None

    tickers = client.get_ticker()
    candidates = []
    for t in tickers:
        symbol = t.get("symbol")
        if not symbol or not symbol.endswith("USDT"):
            continue
        base = symbol[:-4]
        if base.upper() in STABLES:
            continue
        if spot_ok is not None and symbol not in spot_ok:
            continue
        try:
            vol = float(t.get("quoteVolume", 0))
            if vol > 0:
                candidates.append((symbol, vol))
        except Exception:
            continue
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [s for s,_ in candidates[:limit]]
    log(f"Top USDT (hors stables): {selected}")
    return selected

def ensure_exchange_and_quote(db: Session):
    exch = db.execute(select(Exchange).where(Exchange.name=="Binance")).scalar_one_or_none()
    if not exch:
        exch = Exchange(name="Binance")
        db.add(exch)
    quote = db.execute(select(Crypto).where(Crypto.symbol=="USDT")).scalar_one_or_none()
    if not quote:
        quote = Crypto(name="Tether", symbol="USDT")
        db.add(quote)
    db.commit()
    return exch.id

def ensure_pair(db: Session, exchange_id: int, symbol: str):
    base_sym = symbol[:-4]
    base_crypto = db.execute(select(Crypto).where(Crypto.symbol==base_sym)).scalar_one_or_none()
    if not base_crypto:
        base_crypto = Crypto(name=base_sym, symbol=base_sym)
        db.add(base_crypto)
        db.commit()
    quote_crypto = db.execute(select(Crypto).where(Crypto.symbol=="USDT")).scalar_one()
    pair = db.execute(
        select(Pair).where(Pair.symbol==symbol, Pair.exchange_id==exchange_id)
    ).scalar_one_or_none()
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

def earliest_open_api(symbol: str):
    """Retourne le timestamp (UTC) de la première bougie disponible pour le symbole, ou None si indisponible."""
    try:
        kl = client.get_historical_klines(symbol, INTERVAL, "1 Jan 1900", limit=1)
        if not kl:
            return None
        return pd.to_datetime(kl[0][0], unit='ms', utc=True)
    except Exception as e:
        log(f"{symbol} earliest_open_api erreur: {e}")
        return None

def has_four_years_history_cached(db: Session, pair_id: int, symbol: str):
    """Retourne True si on dispose d'au moins 4 ans d'historique pour la paire.
    - Si la paire a déjà des bougies en base: compare MIN(open_datetime) à now-4ans.
    - Si aucune bougie en base (nouvelle paire): interroge l'API pour connaître la première bougie.
    """
    required_date = datetime.now(timezone.utc) - timedelta(days=YEARS*365)
    earliest_db = db.execute(
        select(func.min(Candle.open_datetime)).where(Candle.pair_id == pair_id)
    ).scalar_one_or_none()
    if earliest_db is not None:
        return earliest_db <= required_date
    # Pas de données en base → interroge l'API pour connaître la première bougie
    eo = earliest_open_api(symbol)
    if eo is None:
        return False
    return eo <= required_date

def fetch_full_history(symbol: str, years=YEARS):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=years*365)
    cur = start
    parts = []
    log(f"{symbol}: full fetch {years}y")
    while cur < end:
        nxt = (cur.replace(day=1)+timedelta(days=32)).replace(day=1)
        if nxt > end: nxt = end
        try:
            kl = client.get_historical_klines(
                symbol, INTERVAL,
                cur.strftime("%d %b %Y %H:%M:%S"),
                nxt.strftime("%d %b %Y %H:%M:%S")
            )
            if kl:
                dfm = pd.DataFrame(kl, columns=[
                    'open_time','open','high','low','close','volume',
                    'close_time','quote_asset_volume','number_of_trades',
                    'taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'
                ])
                parts.append(dfm)
                log(f"{symbol} {cur.strftime('%Y-%m')} {len(dfm)}")
        except Exception as e:
            log(f"{symbol} erreur {cur.strftime('%Y-%m')}: {e}")
        cur = nxt
        time.sleep(0.1)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df.drop_duplicates(subset=['open_time'], inplace=True)
    return df

def fetch_range(symbol: str, start_dt: datetime, end_dt: datetime):
    """Récupère les klines [start_dt, end_dt] par segments mensuels pour éviter les timeouts."""
    cur = start_dt
    parts = []
    log(f"{symbol}: fetch range {start_dt.date()} -> {end_dt.date()}")
    while cur < end_dt:
        nxt = (cur.replace(day=1)+timedelta(days=32)).replace(day=1)
        if nxt > end_dt:
            nxt = end_dt
        try:
            kl = client.get_historical_klines(
                symbol, INTERVAL,
                cur.strftime("%d %b %Y %H:%M:%S"),
                nxt.strftime("%d %b %Y %H:%M:%S")
            )
            if kl:
                dfm = pd.DataFrame(kl, columns=[
                    'open_time','open','high','low','close','volume',
                    'close_time','quote_asset_volume','number_of_trades',
                    'taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'
                ])
                parts.append(dfm)
                log(f"{symbol} {cur.strftime('%Y-%m')} {len(dfm)}")
        except Exception as e:
            log(f"{symbol} erreur {cur.strftime('%Y-%m')}: {e}")
        cur = nxt
        time.sleep(0.1)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df.drop_duplicates(subset=['open_time'], inplace=True)
    return df

def fetch_incremental(symbol: str, start_ms: int):
    try:
        kl = client.get_historical_klines(symbol, INTERVAL, str(start_ms))
        if not kl:
            return None
        df = pd.DataFrame(kl, columns=[
            'open_time','open','high','low','close','volume',
            'close_time','quote_asset_volume','number_of_trades',
            'taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'
        ])
        df.drop_duplicates(subset=['open_time'], inplace=True)
        return df
    except Exception as e:
        log(f"{symbol} incr erreur: {e}")
        return None

def df_to_candles(df, pair_id):
    if df is None or df.empty:
        return []
    df['open_datetime'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close_datetime'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    rows = []
    for _,r in df.iterrows():
        rows.append(Candle(
            pair_id=pair_id,
            open_datetime=r['open_datetime'],
            close_datetime=r['close_datetime'],
            open_price=float(r['open']),
            high_price=float(r['high']),
            low_price=float(r['low']),
            close_price=float(r['close']),
            volume_base=float(r['volume']),
            volume_quote=float(r['quote_asset_volume']),
            trade_count=int(r['number_of_trades']),
            taker_buy_base_volume=float(r['taker_buy_base_asset_volume']),
            taker_buy_quote_volume=float(r['taker_buy_quote_asset_volume'])
        ))
    return rows

def get_last_open_datetime(db: Session, pair_id: int):
    return db.execute(
        select(func.max(Candle.open_datetime)).where(Candle.pair_id == pair_id)
    ).scalar_one_or_none()

def run_incremental_cycle():
    init_db()

    with SessionLocal() as db:
        exch_id = ensure_exchange_and_quote(db)
        # 1) Récupérer un pool de candidats plus large que TOP_N
        candidate_pool = get_top_symbols(limit=50)
        log(f"Candidats (max 50): {candidate_pool[:10]}...")

        # 2) Sélectionner au plus TOP_N symboles en respectant la contrainte 4 ans pour les nouveaux
        selected: list[str] = []
        failures = 0
        for symbol in candidate_pool:
            if len(selected) >= TOP_N or failures >= 5:
                break

            # Pair déjà existante ? → on l'accepte directement, on fera l'incrémental
            pair_row = db.execute(select(Pair).where(Pair.exchange_id==exch_id, Pair.symbol==symbol)).scalar_one_or_none()
            if pair_row:
                selected.append(symbol)
                continue

            # Nouvelle paire: vérifier qu'au moins 4 ans sont disponibles (côté API)
            required_date = datetime.now(timezone.utc) - timedelta(days=YEARS*365)
            eo = earliest_open_api(symbol)
            if eo is None or eo > required_date:
                failures += 1
                log(f"{symbol}: ignorée (pas 4 ans). Echecs={failures}")
                continue
            selected.append(symbol)

        log(f"Sélection finale (TOP {TOP_N} avec contrainte 4 ans pour nouvelles): {selected}")

        # 3) Traiter uniquement ces symboles (nouveaux → full 4y; existants → incrémental)
        for symbol in selected:
            pair_row = db.execute(select(Pair).where(Pair.exchange_id==exch_id, Pair.symbol==symbol)).scalar_one_or_none()
            if not pair_row:
                # Créer la paire et backfill 4 ans
                pair_id = ensure_pair(db, exch_id, symbol)
                start_dt = datetime.now(timezone.utc) - timedelta(days=YEARS*365)
                df_full = fetch_range(symbol, start_dt, datetime.now(timezone.utc))
                candles = df_to_candles(df_full, pair_id)
                if candles:
                    db.bulk_save_objects(candles)
                    db.commit()
                    log(f"{symbol}: initial {len(candles)} bougies (4 ans).")
                else:
                    log(f"{symbol}: aucun historique récupéré malgré éligibilité. A vérifier.")
                continue

            # Incrémental
            pair_id = pair_row.id
            last_open = get_last_open_datetime(db, pair_id)
            if last_open:
                next_needed = last_open + timedelta(hours=1)
                now_utc = datetime.now(timezone.utc)
                if next_needed.tzinfo is not None:
                    if next_needed > now_utc - timedelta(hours=1):
                        log(f"{symbol}: à jour.")
                        continue
                else:
                    if next_needed > now_utc.replace(tzinfo=None) - timedelta(hours=1):
                        log(f"{symbol}: à jour.")
                        continue
                start_ms = int(next_needed.timestamp()*1000)
                df_inc = fetch_incremental(symbol, start_ms)
                candles = df_to_candles(df_inc, pair_id)
                if candles:
                    db.bulk_save_objects(candles)
                    db.commit()
                    log(f"{symbol}: +{len(candles)} incrément.")
                else:
                    log(f"{symbol}: aucune nouvelle bougie.")
            else:
                # Paire existante sans données → backfill 4 ans
                start_dt = datetime.now(timezone.utc) - timedelta(days=YEARS*365)
                df_full = fetch_range(symbol, start_dt, datetime.now(timezone.utc))
                candles = df_to_candles(df_full, pair_id)
                if candles:
                    db.bulk_save_objects(candles)
                    db.commit()
                    log(f"{symbol}: (rattrapage) {len(candles)} bougies (4 ans).")

def main():
    run_incremental_cycle()

if __name__ == "__main__":
    main()