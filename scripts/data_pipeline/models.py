from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey, UniqueConstraint, Index

# Définition des tables de la BDD PostgreSQL

Base = declarative_base()

class Exchange(Base):
    __tablename__ = "exchange"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)

class Crypto(Base):
    __tablename__ = "crypto"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    symbol = Column(String(15), unique=True, nullable=False)

class Pair(Base):
    __tablename__ = "pair"
    id = Column(Integer, primary_key=True)
    exchange_id = Column(Integer, ForeignKey("exchange.id"), nullable=False)
    base_crypto_id = Column(Integer, ForeignKey("crypto.id"), nullable=False)
    quote_crypto_id = Column(Integer, ForeignKey("crypto.id"), nullable=False)
    symbol = Column(String(30), nullable=False)
    is_active = Column(Boolean, default=True)
    __table_args__ = (
        UniqueConstraint("exchange_id", "symbol", name="uq_exchange_symbol"),
        Index("idx_pair_symbol", "symbol"),  # Index simple sur symbol
    )

class Candlestick(Base):  
    __tablename__ = "candlestick"
    id = Column(Integer, primary_key=True)
    pair_id = Column(Integer, ForeignKey("pair.id"), nullable=False)
    open_datetime = Column(DateTime(timezone=True), nullable=False)
    close_datetime = Column(DateTime(timezone=True), nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume_base = Column(Float, nullable=False)
    volume_quote = Column(Float, nullable=False)
    trade_count = Column(Integer)
    taker_buy_base_volume = Column(Float)
    taker_buy_quote_volume = Column(Float)
    __table_args__ = (
        UniqueConstraint("pair_id", "open_datetime", name="uq_pair_time"),
        Index("idx_candle_pair_time", "pair_id", "open_datetime"),  # Index pair+time
    )