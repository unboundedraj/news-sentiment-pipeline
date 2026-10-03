"""`newspipe` command line - the same tasks the Airflow DAGs run.

newspipe backfill        historical archive + market history + build + features + checks
newspipe daily           RSS + market increment + build + features + checks + scoring
newspipe train           walk-forward backtest, MLflow logging, champion promotion
newspipe score           score unscored days with the champion model
newspipe reproduce       original notebook protocol on the same data
newspipe status          row counts, watermarks, label audit, last runs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from newspipe import tasks
from newspipe.config import load_settings
from newspipe.warehouse import connect, new_run_id


def _backfill(s, run_id, args):
    tasks.ingest_historical(s, run_id)
    tasks.ingest_market(s, run_id)
    tasks.build_warehouse(s, run_id)
    tasks.build_features(s, run_id)
    tasks.check_quality(s, run_id)


def _daily(s, run_id, args):
    from newspipe.score import score_new_days

    tasks.ingest_rss(s, run_id)
    tasks.ingest_market(s, run_id)
    tasks.build_warehouse(s, run_id)
    tasks.build_features(s, run_id)
    tasks.check_quality(s, run_id)
    score_new_days(s)


def _train(s, run_id, args):
    from newspipe.train import run_backtest

    result = run_backtest(s, run_id, target=args.target, only=args.only, track=not args.no_track)
    print(result["summary"].to_string(index=False))
    if "mlflow" in result:
        print(json.dumps(result["mlflow"], indent=2))


def _score(s, run_id, args):
    from newspipe.score import score_new_days

    print(score_new_days(s, only_live=not args.all))


def _reproduce(s, run_id, args):
    from newspipe.reproduce import reproduce_original

    print(json.dumps(reproduce_original(s), indent=2))


def _status(s, run_id, args):
    with connect(s.warehouse) as con:
        for title, sql in [
            (
                "row counts",
                """
                SELECT 'bronze.historical_headlines' t, count(*) n FROM bronze.historical_headlines
                UNION ALL SELECT 'bronze.market_bars', count(*) FROM bronze.market_bars
                UNION ALL SELECT 'bronze.rss_items', count(*) FROM bronze.rss_items""",
            ),
            ("watermarks", "SELECT * FROM ops.watermarks ORDER BY source"),
            ("label audit", "SELECT * FROM gold.label_audit"),
            (
                "last runs",
                """SELECT task, status, round(duration_s, 1) s, rows_written
                             FROM ops.pipeline_runs ORDER BY started_at DESC LIMIT 8""",
            ),
        ]:
            print(f"== {title} ==")
            try:
                print(con.sql(sql))
            except Exception as exc:  # table may not exist yet
                print(f"  (unavailable: {exc.__class__.__name__})")


COMMANDS = {
    "backfill": _backfill,
    "daily": _daily,
    "train": _train,
    "score": _score,
    "reproduce": _reproduce,
    "status": _status,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="newspipe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--target", choices=["next_day_up", "same_day_up"])
    p.add_argument("--only", nargs="*", help="train: restrict to these candidates")
    p.add_argument("--no-track", action="store_true", help="train: skip MLflow")
    p.add_argument("--all", action="store_true", help="score: include archive days")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )
    for noisy in ("urllib3", "alembic", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    COMMANDS[args.command](load_settings(), new_run_id(), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
