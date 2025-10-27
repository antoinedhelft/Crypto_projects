import os
from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.sensors.python import PythonSensor
from airflow.models import Variable
from sqlalchemy import create_engine, text
from docker.types import Mount

NETWORK = "juil25-bde-crypto_default"


def _db_url() -> str:
    # Réutilise l'environnement défini sur airflow par docker-compose
    return os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://crypto:crypto@postgres:5432/crypto_trading",
    )


def _postgres_ready() -> bool:
    """Renvoi True quand Postgres répond à SELECT 1."""
    try:
        engine = create_engine(_db_url(), future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

# fonction qui vérifie si la table candlestick est vide ou inexistante
# stock dans une variable airflow si le chargement initial a déjà été fait
def _should_run_initial_load() -> bool:
    """Condition de cour-circuit condition: run seulement si n'est pas fait et que la table est vide oumanquante."""
    done_flag = Variable.get("INITIAL_LOAD_DONE", default_var="false").lower() == "true"
    if done_flag:
        # Already done -> skip downstream
        return False

    engine = create_engine(_db_url(), future=True)
    with engine.connect() as conn:
        # Vérifie si la table candlestick existe
        exists = conn.execute(
            text("SELECT to_regclass('public.candlestick') IS NOT NULL")
        ).scalar()
        if not exists:
            # si pas de table -> considère la BDD vide -> lance le dag
            return True
        # si la table existte -> vérifie le nombre de lignes
        cnt = conn.execute(text("SELECT COUNT(*) FROM public.candlestick")).scalar()
        if cnt == 0:
            return True
        # si la BDD est déjà remplie -> défini une variable qui renvoie l'état de la BDD
        Variable.set("INITIAL_LOAD_DONE", "true")
        return False


def _mark_initial_load_done() -> None:
    Variable.set("INITIAL_LOAD_DONE", "true")


with DAG(
    dag_id="crypto_initial_load",
    description="Chargement initial des données historiques de cryptomonnaies dans la BDD",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@once",  # Se déclenche automatiquement au premier lancement du DAG
    catchup=False,
    is_paused_upon_creation=False,
) as dag:
    wait_for_postgres = PythonSensor(
        task_id="wait_for_postgres",
        python_callable=_postgres_ready,
        poke_interval=10,
        timeout=10 * 60,  # 10 minutes max
        mode="reschedule",
    )

    gate_db_empty = ShortCircuitOperator(
        task_id="gate_db_empty",
        python_callable=_should_run_initial_load,
    )

    initial_load = DockerOperator(
        task_id="initial_load",
        image="crypto_data_pipeline:latest",
        command="-m scripts.data_pipeline.initial_load",
        docker_url="unix://var/run/docker.sock",
        api_version="auto",
        network_mode=NETWORK,
        mount_tmp_dir=False,
        environment={
            "DATABASE_URL":"postgresql+psycopg2://crypto:crypto@postgres:5432/crypto_trading",
            "PYTHONUNBUFFERED":"1"
        },
        mounts=[Mount(source="models_data", target="/app/algo_crypto", type="volume")],
        auto_remove=True
    )

    mark_done = PythonOperator(
        task_id="mark_initial_load_done",
        python_callable=_mark_initial_load_done,
    )

    wait_for_postgres >> gate_db_empty >> initial_load >> mark_done