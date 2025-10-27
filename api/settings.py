import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
# Autorise le remplacement MODELS_DIR depuis l'environnement; retour à repo_root/algo_crypto
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "algo_crypto")))

ALGO_DIR = Path(
    os.getenv("MODELS_DIR")
    or ("/app/algo_crypto" if Path("/app").exists() else (Path(__file__).resolve().parents[1] / "algo_crypto"))
)

# Variables d'environnement avec fallback
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg2://crypto:crypto@postgres:5432/crypto_trading"
)

# Noms des modèles par défaut
DEFAULT_REG_MODEL = "crypto_regressor_lgbm.joblib"
DEFAULT_CLF_MODEL = "crypto_classifier_lgbm.joblib"