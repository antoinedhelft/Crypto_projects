import os
from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.models import Variable
from docker.types import Mount
from sqlalchemy import create_engine, text

NETWORK = "juil25-bde-crypto-main_default"

def _db_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://crypto:crypto@postgres:5432/crypto_trading",
    )

def _initial_load_done() -> bool:
    """Autorise la mise à jour si il y a eu le chargement initial.
    Si la variable est False, on vérifie dans la base de données (la table candlestick existe et a des lignes);
    SI elle est trouvée et possède des lignes, on met la variable à True pour débloquer les prochaines exécutions.
    """
    flag = Variable.get("INITIAL_LOAD_DONE", default_var="false").lower() == "true"
    if flag:
        return True
    # solutio nde secours pour vérifier la BDD
    try:
        engine = create_engine(_db_url(), future=True)
        with engine.connect() as conn:
            exists = conn.execute(text("SELECT to_regclass('public.candlestick') IS NOT NULL")).scalar()
            if not exists:
                return False
            cnt = conn.execute(text("SELECT COUNT(*) FROM public.candlestick"))
            n = cnt.scalar() or 0
            if n > 0:
                Variable.set("INITIAL_LOAD_DONE", "true")
                return True
            return False
    except Exception:
        # En cas d'erreur de BDD, on ignore par prudence
        return False


with DAG(
    dag_id="crypto_hourly_update",
    description="Mise à jour horaire des données de marché (candles) pour toutes les cryptomonnaies",
    start_date=datetime(2025,1,1),
    schedule_interval="0 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"owner":"crypto","retries":1},
    is_paused_upon_creation=False,
) as dag:
    gate_initial_done = ShortCircuitOperator(
        task_id="gate_initial_load_done",
        python_callable=_initial_load_done,
    )

    update = DockerOperator(
        task_id="update_candles",
        image="crypto_data_pipeline:latest",
        command="-m scripts.data_pipeline.incremental_update",
        docker_url="unix://var/run/docker.sock",
        api_version="auto",
        network_mode=NETWORK,
        mount_tmp_dir=False,
        environment={
            "DATABASE_URL":"postgresql+psycopg2://crypto:crypto@postgres:5432/crypto_trading",
            "PYTHONUNBUFFERED":"1",
            "MODELS_DIR":"/app/algo_crypto"
        },
        mounts=[Mount(source="models_data", target="/app/algo_crypto", type="volume")],
        auto_remove=True
    )

    gate_initial_done >> update