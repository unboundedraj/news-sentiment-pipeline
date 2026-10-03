"""Weekly: walk-forward backtest -> MLflow -> register -> promote if better -> rescore.

Waits for the latest successful `news_daily` run so the model always trains on a
quality-gated feature table.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from _common import DEFAULT_ARGS, run_id
from airflow.decorators import dag, task
from airflow.sensors.external_task import ExternalTaskSensor


@dag(
    dag_id="model_retrain",
    schedule="0 23 * * 5",  # Fridays, one hour after news_daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["newspipe", "ml"],
    doc_md=__doc__,
)
def model_retrain():
    wait_for_daily = ExternalTaskSensor(
        task_id="wait_for_news_daily",
        external_dag_id="news_daily",
        external_task_id="quality_gate",
        execution_delta=timedelta(hours=1),
        mode="reschedule",
        timeout=6 * 3600,
    )

    @task
    def backtest_and_register(**context) -> dict:
        from newspipe.config import load_settings
        from newspipe.train import run_backtest

        result = run_backtest(load_settings(), run_id(context))
        return {"best_model": result["best_model"], **result.get("mlflow", {})}

    @task
    def rescore(info: dict) -> int:
        from newspipe.config import load_settings
        from newspipe.score import score_new_days

        # A newly promoted champion re-scores live days under its own version.
        return score_new_days(load_settings()) if info.get("promoted") else 0

    info = backtest_and_register()
    wait_for_daily >> info
    rescore(info)


model_retrain()
