from pathlib import Path
from datetime import datetime, timedelta, timezone
import time
import json
import os
import pandas as pd
from binance.client import Client
from sqlalchemy import select, func, text
from sqlalchemy.orm import Session
from .db_utils import SessionLocal, init_db, engine  
from .models import Pair, Candlestick as Candle, Exchange, Crypto  

# Ce script fait la mise a jour horaire de la table candlestick:
# 1) selection des paires a traiter
# 2) recuperation des bougies Binance (initiale ou incrementale)
# 3) validation des lignes avant insertion
# 4) generation d'un rapport de qualite des donnees

YEARS = 4
TOP_N = 3
INTERVAL = Client.KLINE_INTERVAL_1HOUR
# Base assets considérés comme stables (à exclure côté baseAsset pour garder des paires type BTCUSDT, ETHUSDT, ...)
STABLES = {"USDT","USDC","BUSD","DAI","TUSD","PAX","USDP","FDUSD","GUSD","USDE"}

client = Client()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "algo_crypto")))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    """Affiche un message horodate pour faciliter le debug en production."""
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
    """Cree (si besoin) les references Exchange/Crypto minimales utilisees par les paires."""
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
    """Cree (si absente) la paire exchange+symbol et retourne son pair_id."""
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
    """Recupere l'historique complet sur `years` annees, segmente mois par mois."""
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
    """Recupere uniquement les nouvelles bougies a partir d'un timestamp millisecondes."""
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

def df_to_candles(df, pair_id, symbol: str = "unknown"):
    """Convertit un DataFrame Binance en objets SQLAlchemy Candle.

    Cette fonction applique aussi les regles de validation metier avant insertion
    pour eviter de polluer la base avec des lignes incoherentes.
    """
    if df is None or df.empty:
        return []
    df['open_datetime'] = pd.to_datetime(df['open_time'], unit='ms', utc=True, errors='coerce')
    df['close_datetime'] = pd.to_datetime(df['close_time'], unit='ms', utc=True, errors='coerce')
    rows = []
    rejected = 0
    rejected_reasons: dict[str, int] = {}

    def _reject(reason: str) -> None:
        # Compte les lignes rejetees par type d'erreur pour l'observabilite.
        nonlocal rejected
        rejected += 1
        rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

    for _,r in df.iterrows():
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

        # Controle 1: timestamps presents et ordonnes.
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

        # Controle 3: coherence des volumes et compteurs.
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

    if rejected > 0:
        log(f"{symbol}: {len(rows)} bougies valides, {rejected} rejetees avant insertion.")
        log(f"{symbol}: detail rejets = {rejected_reasons}")
    return rows

def get_last_open_datetime(db: Session, pair_id: int):
    """Retourne la derniere bougie connue pour une paire."""
    return db.execute(
        select(func.max(Candle.open_datetime)).where(Candle.pair_id == pair_id)
    ).scalar_one_or_none()


def _compute_data_quality_report(db: Session) -> dict:
    """Construit un rapport de qualite global apres chargement.

    Objectif: verifier rapidement si les donnees inserees restent exploitables.
    """
    # Volume global de donnees presentes.
    total_rows = int(db.execute(select(func.count(Candle.id))).scalar_one() or 0)

    # Nombre de symbols distincts couverts dans candlestick.
    symbol_count = int(
        db.execute(
            select(func.count(func.distinct(Pair.symbol)))
            .select_from(Candle)
            .join(Pair, Pair.id == Candle.pair_id)
        ).scalar_one()
        or 0
    )

    # Fraicheur: derniere bougie disponible en base.
    latest_open = db.execute(select(func.max(Candle.open_datetime))).scalar_one_or_none()
    if latest_open is not None and latest_open.tzinfo is None:
        latest_open = latest_open.replace(tzinfo=timezone.utc)

    # Check 1: champs critiques nuls (ne devrait jamais arriver).
    null_count = int(
        db.execute(
            text(
                """
                SELECT COUNT(*)
                FROM candlestick
                WHERE pair_id IS NULL
                   OR open_datetime IS NULL
                   OR close_datetime IS NULL
                   OR open_price IS NULL
                   OR high_price IS NULL
                   OR low_price IS NULL
                   OR close_price IS NULL
                   OR volume_base IS NULL
                   OR volume_quote IS NULL
                """
            )
        ).scalar_one()
        or 0
    )

    # Check 2: prix non positifs.
    invalid_price_count = int(
        db.execute(
            text(
                """
                SELECT COUNT(*)
                FROM candlestick
                WHERE open_price <= 0
                   OR high_price <= 0
                   OR low_price <= 0
                   OR close_price <= 0
                """
            )
        ).scalar_one()
        or 0
    )

    # Check 3: doublons de bougies par paire+timestamp.
    duplicate_pair_time_groups = int(
        db.execute(
            text(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT pair_id, open_datetime, COUNT(*) AS n
                    FROM candlestick
                    GROUP BY pair_id, open_datetime
                    HAVING COUNT(*) > 1
                ) d
                """
            )
        ).scalar_one()
        or 0
    )

    # Fraicheur exprimee en heures de retard par rapport a maintenant.
    now_utc = datetime.now(timezone.utc)
    freshness_hours = None
    if latest_open is not None:
        freshness_hours = round((now_utc - latest_open).total_seconds() / 3600, 2)

    # Statut de chaque controle, puis statut global.
    checks = {
        "null_critical_fields": {
            "count": null_count,
            "status": "ok" if null_count == 0 else "fail",
        },
        "non_positive_prices": {
            "count": invalid_price_count,
            "status": "ok" if invalid_price_count == 0 else "fail",
        },
        "duplicate_pair_open_datetime": {
            "count": duplicate_pair_time_groups,
            "status": "ok" if duplicate_pair_time_groups == 0 else "fail",
        },
        "freshness_hours": {
            "value": freshness_hours,
            "status": (
                "unknown"
                if freshness_hours is None
                else "ok"
                if freshness_hours <= 2.0
                else "warn"
            ),
        },
    }

    overall_status = "ok"
    if any(v.get("status") == "fail" for v in checks.values()):
        overall_status = "fail"
    elif any(v.get("status") == "warn" for v in checks.values()):
        overall_status = "warn"

    return {
        "generated_at": now_utc.isoformat(),
        "table": "candlestick",
        "row_count": total_rows,
        "symbol_count": symbol_count,
        "latest_open_datetime": latest_open.isoformat() if latest_open else None,
        "overall_status": overall_status,
        "checks": checks,
    }


def _persist_data_quality_report(report: dict) -> None:
    """Sauvegarde le rapport de qualite en version historisee + latest."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    history_file = MODELS_DIR / f"data_quality_{timestamp}.json"
    latest_file = MODELS_DIR / "data_quality_latest.json"

    history_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Data quality report saved: {latest_file.name} (status={report.get('overall_status')})")

def run_incremental_cycle():
    """Orchestre un cycle complet d'update incremental."""
    # S'assure que les tables existent avant tout traitement.
    init_db()

    with SessionLocal() as db:
        # Etape A: references minimales (exchange + quote asset).
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
                candles = df_to_candles(df_full, pair_id, symbol=symbol)
                if candles:
                    db.bulk_save_objects(candles)
                    db.commit()
                    log(f"{symbol}: initial {len(candles)} bougies (4 ans).")
                else:
                    log(f"{symbol}: aucun historique récupéré malgré éligibilité. A vérifier.")
                continue

            # Incrémental: on ne récupère que les bougies manquantes depuis la dernière heure connue.
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
                candles = df_to_candles(df_inc, pair_id, symbol=symbol)
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
                candles = df_to_candles(df_full, pair_id, symbol=symbol)
                if candles:
                    db.bulk_save_objects(candles)
                    db.commit()
                    log(f"{symbol}: (rattrapage) {len(candles)} bougies (4 ans).")

        # Etape finale: produire le rapport de qualite pour l'observabilite.
        try:
            dq_report = _compute_data_quality_report(db)
            _persist_data_quality_report(dq_report)
        except Exception as e:
            log(f"Data quality report generation failed: {e}")

def main():
    """Point d'entree CLI du script incremental."""
    run_incremental_cycle()

if __name__ == "__main__":
    main()