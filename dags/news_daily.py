"""Daily: ingest RSS + market increment -> rebuild silver/gold -> features -> DQ gate -> score.

RSS and market ingestion run in parallel; the warehouse build waits for both. The
quality gate raises on blocking failures, so scoring never runs on bad data.
"""

from __future__ import annotations

from datetime import datetime

from _common import DEFAULT_ARGS, run_id
from airflow.decorators import dag, task


@dag(
    dag_id="news_daily",
    schedule="0 22 * * 1-5",  # after the US close, weekdays
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["newspipe", "ingestion", "scoring"],
    doc_md=__doc__,
)
def news_daily():
    @task
    def ingest_rss(**context) -> int:
        from newspipe import tasks
        from newspipe.config import load_settings

        return tasks.ingest_rss(load_settings(), run_id(context))

    @task
    def ingest_market(**context) -> int:
        from newspipe import tasks
        from newspipe.config import load_settings

        return tasks.ingest_market(load_settings(), run_id(context))

    @task
    def build_warehouse(**context) -> dict:
        from newspipe import tasks
        from newspipe.config import load_settings

        stats = tasks.build_warehouse(load_settings(), run_id(context))
        return {k: v for k, v in stats.items() if k != "label_audit"}

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

    @task
    def score(**context) -> int:
        from newspipe.config import load_settings
        from newspipe.score import score_new_days

        return score_new_days(load_settings())

    built = build_warehouse()
    [ingest_rss(), ingest_market()] >> built
    built >> build_features() >> quality_gate() >> score()


news_daily()
