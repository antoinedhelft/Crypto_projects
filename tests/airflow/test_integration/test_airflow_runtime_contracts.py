"""Tests d'integration Airflow sur les regles d'execution des DAGs.

Ces tests verifient des proprietes runtime importantes (images Docker, commandes,
ordonnancement des taches) sans necessiter un scheduler Airflow actif.
"""

import os
from pathlib import Path

import pytest


AIRFLOW_AVAILABLE = False
if os.name != "nt":
    # Bloc setup: forcer un environnement Airflow minimal pour charger les DAGs en test.
    AIRFLOW_TEST_HOME = Path(__file__).resolve().parents[3] / ".airflow_test"
    AIRFLOW_TEST_HOME.mkdir(parents=True, exist_ok=True)
    AIRFLOW_TEST_DB = (AIRFLOW_TEST_HOME / "airflow_test.db").resolve()

    os.environ.setdefault("AIRFLOW_HOME", str(AIRFLOW_TEST_HOME))
    os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
    os.environ.setdefault("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN", f"sqlite:///{AIRFLOW_TEST_DB.as_posix()}")
    # Necessaire uniquement pour le parsing des DAGs en CI.
    # Aucune connexion reelle n'est faite vers cette URL dans ces tests.
    os.environ.setdefault("DATABASE_URL", os.environ.get("TEST_DATABASE_URL", "__CI_TEST_DATABASE_URL__"))

    try:
        from airflow.models import DagBag
        from airflow.providers.docker.operators.docker import DockerOperator
        AIRFLOW_AVAILABLE = True
    except ImportError:
        AIRFLOW_AVAILABLE = False


DAGS_DIR = Path(__file__).resolve().parents[3] / "airflow" / "dags"

SKIP_NO_AIRFLOW = pytest.mark.skipif(
    not AIRFLOW_AVAILABLE,
    reason="Ces tests d'integration Airflow tournent en CI Linux avec les dependances Airflow",
)


@pytest.mark.integration
@SKIP_NO_AIRFLOW
def test_hourly_update_dag_uses_expected_docker_settings():
    # Charger les DAGs puis recuperer le DAG horaire.
    dag_bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
    assert len(dag_bag.import_errors) == 0, f"Erreurs d'import detectees: {dag_bag.import_errors}"

    # Lire le DAG directement depuis la collection chargee en memoire.
    dag = dag_bag.dags.get("crypto_hourly_update")
    assert dag is not None, "Le DAG crypto_hourly_update doit exister"

    # Recuperer la tache Docker de mise a jour.
    update_task = dag.get_task("update_candles")
    # Verifier l'image, la commande et le reseau utilises.
    assert isinstance(update_task, DockerOperator)
    assert update_task.image == "crypto_data_pipeline:latest"
    assert update_task.command == "-m scripts.data_pipeline.incremental_update"
    assert update_task.network_mode == "juil25-bde-crypto-main_default"


@pytest.mark.integration
@SKIP_NO_AIRFLOW
def test_initial_load_dag_task_order_is_gated():
    # Charger le DAG d'initialisation.
    dag_bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
    assert len(dag_bag.import_errors) == 0, f"Erreurs d'import detectees: {dag_bag.import_errors}"

    # Lire le DAG directement depuis la collection chargee en memoire.
    dag = dag_bag.dags.get("crypto_initial_load")
    assert dag is not None, "Le DAG crypto_initial_load doit exister"

    # Recuperer les taches principales du flux.
    wait_task = dag.get_task("wait_for_postgres")
    gate_task = dag.get_task("gate_db_empty")
    load_task = dag.get_task("initial_load")
    mark_task = dag.get_task("mark_initial_load_done")

    # Verifier l'ordre des dependances entre taches.
    assert gate_task.task_id in wait_task.downstream_task_ids
    assert load_task.task_id in gate_task.downstream_task_ids
    assert mark_task.task_id in load_task.downstream_task_ids
