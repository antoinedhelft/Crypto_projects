import os
from datetime import datetime
from pathlib import Path
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import ShortCircuitOperator, get_current_context
from airflow.models import Variable
from docker.types import Mount

NETWORK = "juil25-bde-crypto-main_default"

def _should_train() -> bool:
    """Décide si le job de training doit s'exécuter.

    Retourne True si:
    - FORCE_TRAIN (Variable) == true, ou dag_run.conf.force == true, OU
    - le chargement initial est terminé (INITIAL_LOAD_DONE == true), ET
      (nous sommes le 1er du mois) OU (aucun modèle n'est présent dans MODELS_DIR)
    """
    ctx = get_current_context()
    dag_run = ctx.get("dag_run")

    # 1) Run déclenché manuellement -> toujours exécuter
    if dag_run and getattr(dag_run, "external_trigger", False):
        print("[gate] Run manuel détecté -> run")
        return True

    # 2) Overrides explicites (Variable Airflow ou conf du run)
    force_var = Variable.get("FORCE_TRAIN", default_var="false").lower() == "true"
    
    force_conf = False
    try:
        force_conf = bool((dag_run and dag_run.conf and dag_run.conf.get("force")))
    except Exception:
        force_conf = False
    if force_var or force_conf:
        print(f"[gate] FORCE_TRAIN activé (var={force_var}, conf={force_conf}) -> run")
        return True

    # 3) Vérification chargement initial
    if Variable.get("INITIAL_LOAD_DONE", default_var="false").lower() != "true":
        print("[gate] INITIAL_LOAD_DONE != true -> skip")
        return False

    # 4) Jour du mois
    logical_date = ctx.get("logical_date") or datetime.now()
    if getattr(logical_date, "day", None) == 1:
        print("[gate] 1er du mois -> run")
        return True

    # 5) Présence de modèles existants
    models_dir = Path(os.environ.get("MODELS_DIR", "/app/algo_crypto"))
    has_models = any(models_dir.glob("*.joblib"))
    print(f"[gate] MODELS_DIR={models_dir} has_models={has_models} -> {'skip' if has_models else 'run'}")
    return not has_models

with DAG(
    dag_id="crypto_monthly_ml_train",
    description="Entraînement mensuel des modèles ML (régression et classification) sur les données crypto",
    start_date=datetime(2025,1,1),
    schedule_interval="0 0 1 * *", # Check horaire: lance le 1er du mois OU si aucun modèle n'existe
    catchup=False,
    max_active_runs=1,
    default_args={"owner":"crypto","retries":1},
    is_paused_upon_creation=False,
) as dag:
    gate = ShortCircuitOperator(
        task_id="gate_initial_and_need_training",
        python_callable=_should_train,
    )
    train_models_task = DockerOperator(
        task_id='train_models',
        image='crypto_ml_pipeline:latest',
        command=["python", "/app/scripts/ml_pipeline/pipeline_train.py"],
        docker_url="unix://var/run/docker.sock",
        api_version="auto",
        network_mode=NETWORK,
        force_pull=False,
        mount_tmp_dir=False,
        environment={
            'DATABASE_URL': 'postgresql+psycopg2://crypto:crypto@postgres:5432/crypto_trading',
            'PYTHONPATH': '/app',
            'MODELS_DIR': '/app/algo_crypto'
        },
        mounts=[
            Mount(source='models_data', target='/app/algo_crypto', type='volume')
        ],
        auto_remove=True,
        dag=dag,
    )
    gate >> train_models_task