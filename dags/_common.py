"""Shared DAG defaults. Tasks are thin wrappers around newspipe.tasks functions."""

from __future__ import annotations

from datetime import timedelta

DEFAULT_ARGS = {
    "owner": "data-engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
}


def run_id(context) -> str:
    """Tie warehouse lineage (ops.pipeline_runs) to the Airflow run."""
    return f"airflow-{context['dag'].dag_id}-{context['ts_nodash']}"
