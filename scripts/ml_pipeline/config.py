from pathlib import Path
from datetime import datetime
import os

# Détection d'environnement avec Path (plus propre)
if os.getenv('DOCKER_ENV') or Path('/app').exists():
    # Dans Docker
    ALGO_DIR = Path("/app/algo_crypto")
else:
    # En local - remonter à la racine du projet
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/ml_pipeline -> scripts -> racine
    ALGO_DIR = PROJECT_ROOT / "algo_crypto"

# Créer le dossier si nécessaire
ALGO_DIR.mkdir(exist_ok=True)

# Database config
DB_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://crypto:crypto@localhost:5432/crypto_trading')

# Variables de compatibilité (inutilisées mais importées)
SQLITE_FALLBACK = None
USE_SQLALCHEMY = True

# Versioning avec timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_REG_PATH = ALGO_DIR / f"crypto_regressor_lgbm_{timestamp}.joblib"
MODEL_CLF_PATH = ALGO_DIR / f"crypto_classifier_lgbm_{timestamp}.joblib"
FEATURES_REG_JSON = ALGO_DIR / f"regressor_features_{timestamp}.json"
FEATURES_CLF_JSON = ALGO_DIR / f"classifier_features_{timestamp}.json"
METRICS_JSON = ALGO_DIR / f"metrics_{timestamp}.json"
# Mapping symbol -> code entier, sauvegarde pour que l'inference utilise les memes codes
SYMBOL_MAP_JSON = ALGO_DIR / f"symbol_map_{timestamp}.json"

# Paths des modèles actuels (liens symboliques)
MODEL_REG_CURRENT = ALGO_DIR / "crypto_regressor_lgbm.joblib"
MODEL_CLF_CURRENT = ALGO_DIR / "crypto_classifier_lgbm.joblib"

print(f"[DEBUG] Modèles seront sauvés dans: {ALGO_DIR}")