"""One-off backfill: historical archive + full market history -> warehouse -> features -> DQ.

Trigger manually. Every task is idempotent (upserts keyed by natural keys), so the DAG
can be re-run safely after a partial failure.
"""

from __future__ import annotations

from datetime import datetime

from _common import DEFAULT_ARGS, run_id
from airflow.decorators import dag, task


@dag(
    dag_id="news_backfill",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["newspipe", "backfill"],
    doc_md=__doc__,
)
def news_backfill():
    @task
    def ingest_historical(**context) -> int:
        from newspipe import tasks
        from newspipe.config import load_settings

        return tasks.ingest_historical(load_settings(), run_id(context))

    @task
    def ingest_market(**context) -> int:
        from newspipe import tasks
        from newspipe.config import load_settings

        return tasks.ingest_market(load_settings(), run_id(context))

    @task
    def build_warehouse(**context) -> list:
        from newspipe import tasks
        from newspipe.config import load_settings

        return tasks.build_warehouse(load_settings(), run_id(context))["label_audit"]

    @task
    def build_features(**context) -> dict:
        from newspipe import tasks
        from newspipe.config import load_settings

        return tasks.build_features(load_settings(), run_id(context))

    @task
    def quality_gate(**context) -> dict:
        from newspipe import tasks
        from newspipe.config import load_settings

        return tasks.check_quality(load_settings(), run_id(context))

    built = build_warehouse()
    [ingest_historical(), ingest_market()] >> built
    built >> build_features() >> quality_gate()


news_backfill()
