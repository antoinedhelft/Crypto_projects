import json
import logging
import os
from urllib import request


LOGGER = logging.getLogger(__name__)


def slack_failure_alert(context) -> None:
    """Envoie une alerte Slack quand une tache echoue definitivement.

    Airflow n'appelle ce callback qu'au moment ou la tache passe en etat FAILED,
    donc apres epuisement des retries configures.
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        LOGGER.warning("SLACK_WEBHOOK_URL absent: alerte Slack non envoyee.")
        return

    dag_id = context.get("dag").dag_id if context.get("dag") else "unknown_dag"
    task_instance = context.get("task_instance")
    task_id = task_instance.task_id if task_instance else "unknown_task"
    run_id = context.get("run_id", "unknown_run")
    try_number = getattr(task_instance, "try_number", "n/a")
    log_url = getattr(task_instance, "log_url", "")
    exception = context.get("exception")

    message = {
        "text": (
            ":red_circle: Echec Airflow\n"
            f"DAG: {dag_id}\n"
            f"Tache: {task_id}\n"
            f"Run: {run_id}\n"
            f"Essai final: {try_number}\n"
            f"Erreur: {exception}\n"
            f"Logs: {log_url}"
        )
    }

    req = request.Request(
        webhook_url,
        data=json.dumps(message).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=10) as response:
            if response.status >= 400:
                LOGGER.error("Echec envoi Slack, status=%s", response.status)
    except Exception as exc:
        LOGGER.exception("Impossible d'envoyer l'alerte Slack: %s", exc)