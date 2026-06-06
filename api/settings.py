import os
from pathlib import Path

# Repertoire des modeles ML : priorite a la variable d'environnement (Docker),
# sinon algo_crypto/ a la racine du projet (local)
MODELS_DIR = Path(
    os.getenv("MODELS_DIR")
    or ("/app/algo_crypto" if Path("/app").exists() else str(Path(__file__).resolve().parents[1] / "algo_crypto"))
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://crypto:crypto@postgres:5432/crypto_trading"
)

# Noms des modeles par defaut (utilises si aucun fichier timestampe n'est trouve)
DEFAULT_REG_MODEL = "crypto_regressor_lgbm.joblib"
DEFAULT_CLF_MODEL = "crypto_classifier_lgbm.joblib"