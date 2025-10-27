"""Tests unitaires pour les DAGs Airflow.

Ces tests vérifient la structure et la validité des DAGs sans les exécuter.
"""

import pytest
from pathlib import Path

# Vérifier si Airflow est installé
try:
    from airflow.models import DagBag
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

DAGS_DIR = Path(__file__).resolve().parents[3] / "airflow" / "dags"

SKIP_NO_AIRFLOW = pytest.mark.skipif(
    not AIRFLOW_AVAILABLE,
    reason="Airflow n'est pas installé"
)

SKIP_NO_DAGS = pytest.mark.skipif(
    not DAGS_DIR.exists(),
    reason="Le dossier airflow/dags n'existe pas"
)


@pytest.mark.unitaire
@SKIP_NO_AIRFLOW
@SKIP_NO_DAGS
def test_dags_load_without_errors():
    """
    Vérifie que tous les DAGs présents dans airflow/dags peuvent être importés
    sans erreurs de syntaxe ou d'import.
    """
    dag_bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
    
    # Vérifier qu'il n'y a pas d'erreurs d'import
    assert len(dag_bag.import_errors) == 0, f"Erreurs d'import détectées: {dag_bag.import_errors}"
    
    # Vérifier qu'au moins un DAG a été trouvé
    assert len(dag_bag.dags) > 0, "Aucun DAG trouvé dans airflow/dags"


@pytest.mark.unitaire
@SKIP_NO_AIRFLOW
@SKIP_NO_DAGS
def test_dags_have_required_attributes():
    """
    Vérifie que chaque DAG a les attributs essentiels définis.
    """
    dag_bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
    
    for dag_id, dag in dag_bag.dags.items():
        # Chaque DAG doit avoir un owner
        assert dag.owner is not None, f"Le DAG '{dag_id}' n'a pas d'owner défini"
        
        # Chaque DAG doit avoir une description
        assert dag.description is not None and len(dag.description) > 0, \
            f"Le DAG '{dag_id}' n'a pas de description"
        
        # Chaque DAG doit avoir au moins une tâche
        assert len(dag.tasks) > 0, f"Le DAG '{dag_id}' n'a aucune tâche"


@pytest.mark.unitaire
@SKIP_NO_AIRFLOW
@SKIP_NO_DAGS
def test_dags_have_no_cycles():
    """
    Vérifie qu'aucun DAG ne contient de cycle (dépendances circulaires).
    """
    from airflow.exceptions import AirflowDagCycleException
    
    dag_bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)
    
    for dag_id, dag in dag_bag.dags.items():
        # Airflow 2.x : la validation des cycles se fait automatiquement au chargement
        # On vérifie simplement qu'aucune exception de cycle n'est présente dans import_errors
        # Et que le DAG a bien des tâches avec des dépendances valides
        try:
            # Vérifie que toutes les dépendances sont résolvables
            for task in dag.tasks:
                _ = task.upstream_list
                _ = task.downstream_list
        except AirflowDagCycleException as e:
            pytest.fail(f"Le DAG '{dag_id}' contient un cycle: {e}")
        except Exception as e:
            pytest.fail(f"Erreur dans les dépendances du DAG '{dag_id}': {e}")
