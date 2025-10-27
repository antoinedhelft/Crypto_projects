"""Tests End-to-End pour l'API.

Ces tests nécessitent que Docker Compose soit lancé avec tous les services.
Ils testent l'intégration complète : BDD → API → Prédictions.

Pour lancer ces tests :
1. docker compose up -d
2. Attendre que les services soient prêts (~2 min)
3. pytest -v -m e2e tests/api/test_e2e/
4. docker compose down
"""

import pytest
import requests
import psycopg2
import os
import time

# Marqueur pour les tests E2E
pytestmark = pytest.mark.e2e

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://airflow:airflow@localhost:5432/crypto_db")


@pytest.fixture(scope="module")
def db_connection():
    """Connexion à la base de données PostgreSQL."""
    max_retries = 30
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            yield conn
            conn.close()
            return
        except psycopg2.OperationalError:
            if i == max_retries - 1:
                raise
            time.sleep(2)


def test_api_health():
    """Vérifie que l'API est accessible."""
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_database_has_candles_data(db_connection):
    """Vérifie que la base de données contient des données de bougies."""
    cursor = db_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM candlestick JOIN pair ON candlestick.pair_id = pair.id WHERE symbol='BTCUSDT'")
    count = cursor.fetchone()[0]
    
    assert count > 0, "La table 'candlestick' devrait contenir des données pour BTCUSDT"
    print(f"✅ La BDD contient {count} bougies pour BTCUSDT")


def test_database_has_recent_data(db_connection):
    """Vérifie que les données sont récentes (moins de 24h)."""
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT MAX(open_datetime) 
        FROM candlestick 
        JOIN pair ON candlestick.pair_id = pair.id
        WHERE symbol='BTCUSDT'
    """)
    latest_timestamp = cursor.fetchone()[0]
    
    assert latest_timestamp is not None, "Aucune donnée trouvée dans la table 'candlestick'"
    
    # Vérifier que la donnée la plus récente date de moins de 24h
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    time_diff = now - latest_timestamp
    
    assert time_diff < timedelta(hours=24), \
        f"Les données sont trop anciennes (dernière mise à jour : {latest_timestamp})"
    
    print(f"✅ Dernière donnée : {latest_timestamp} (il y a {time_diff.total_seconds() / 3600:.1f}h)")


def test_predict_endpoint_with_real_data():
    """Teste l'endpoint /predict/{symbol} avec les données réelles de la BDD."""
    response = requests.get(f"{API_URL}/predict/BTCUSDT", timeout=30)
    
    assert response.status_code == 200, f"L'API a retourné une erreur : {response.text}"
    
    data = response.json()
    
    # Vérifier la structure de la réponse
    assert "symbol" in data
    assert data["symbol"] == "BTCUSDT"
    
    assert "prediction" in data
    prediction = data["prediction"]
    
    assert "next_close_price" in prediction
    assert isinstance(prediction["next_close_price"], (int, float))
    assert prediction["next_close_price"] > 0
    
    assert "direction" in prediction
    assert prediction["direction"] in ["Baisse", "Stable", "Hausse"]
    
    assert "confidence" in prediction
    if prediction["confidence"] is not None:
        assert 0 <= prediction["confidence"] <= 100
    
    print(f"✅ Prédiction : {prediction['next_close_price']} USD, Direction : {prediction['direction']}")


def test_models_endpoint_lists_models():
    """Vérifie que l'endpoint /models retourne bien des modèles."""
    response = requests.get(f"{API_URL}/models")
    assert response.status_code == 200
    
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    
    # On devrait avoir au moins 2 modèles (regressor + classifier)
    assert len(data["models"]) >= 2, "Au moins 2 modèles devraient être présents"
    
    print(f"✅ {len(data['models'])} modèles disponibles")


def test_predict_multiple_symbols(db_connection):
    """Teste les prédictions pour tous les symboles présents dans la BDD."""
    cursor = db_connection.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM candlestick JOIN pair ON candlestick.pair_id = pair.id ORDER BY symbol")
    symbols = [row[0] for row in cursor.fetchall()]
    
    assert len(symbols) > 0, "Aucun symbole trouvé dans la BDD"
    
    print(f"\n📊 Test de prédiction pour {len(symbols)} symboles...")
    
    successful_predictions = 0
    for symbol in symbols:
        try:
            response = requests.get(f"{API_URL}/predict/{symbol}", timeout=30)
            if response.status_code == 200:
                successful_predictions += 1
                data = response.json()
                print(f"  ✅ {symbol}: {data['prediction']['next_close_price']} USD")
            else:
                print(f"  ⚠️  {symbol}: Erreur {response.status_code}")
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")
    
    # Au moins 50% des symboles devraient avoir des prédictions valides
    success_rate = successful_predictions / len(symbols)
    assert success_rate >= 0.5, \
        f"Seulement {successful_predictions}/{len(symbols)} prédictions ont réussi"
