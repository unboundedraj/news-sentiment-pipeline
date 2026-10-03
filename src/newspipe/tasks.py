"""Pipeline tasks. Each is idempotent and tracked in ops.pipeline_runs.

The Airflow DAGs (dags/) and the CLI both call these functions, so the orchestration
layer stays thin and everything is testable without a scheduler.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from newspipe import features, quality, transform
from newspipe.config import Settings
from newspipe.sources import historical, market, rss
from newspipe.warehouse import connect, get_watermark, set_watermark, tracked, upsert

log = logging.getLogger("newspipe.tasks")


def ingest_historical(s: Settings, run_id: str) -> int:
    cfg = s["historical"]
    with tracked(s.warehouse, run_id, "ingest_historical") as t:
        path = historical.fetch(cfg["url"], s.path("raw"), cfg.get("sha256"))
        frame = historical.ingest(path, cfg["encoding"])
        quality.HistoricalBronze.validate(frame, lazy=True)
        with connect(s.warehouse) as con:
            t["rows_written"] = upsert(con, "bronze.historical_headlines", frame)
        t["details"] = {"days": int(frame.trade_date.nunique()), "sha256": frame._source_sha256[0]}
    return t["rows_written"]


def ingest_market(s: Settings, run_id: str, today: date | None = None) -> int:
    cfg = s["market"]
    with tracked(s.warehouse, run_id, "ingest_market") as t:
        with connect(s.warehouse) as con:
            last = con.execute(
                "SELECT max(trade_date) FROM bronze.market_bars WHERE symbol = ?", [cfg["symbol"]]
            ).fetchone()[0]
            start, end = market.incremental_window(
                last, date.fromisoformat(cfg["start"]), cfg["refetch_overlap_days"], today
            )
            bars = market.fetch_yahoo(cfg["symbol"], start, end)
            quality.MarketBars.validate(bars, lazy=True)
            bars["_provider"] = cfg["provider"]
            bars["_ingested_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
            t["rows_written"] = upsert(con, "bronze.market_bars", bars)
            if not bars.empty:
                set_watermark(con, f"market:{cfg['symbol']}", bars.trade_date.max())
        t["details"] = {"window": [str(start), str(end)]}
    return t["rows_written"]


def ingest_rss(s: Settings, run_id: str, fetcher=rss.fetch_feed) -> int:
    total = 0
    with tracked(s.warehouse, run_id, "ingest_rss") as t:
        per_feed = {}
        with connect(s.warehouse) as con:
            for name, url in s["rss"]["feeds"].items():
                try:
                    items = fetcher(url, name)
                except Exception as exc:  # one broken feed must not stop the others
                    log.warning("feed %s failed: %s", name, exc)
                    per_feed[name] = f"error: {exc}"
                    continue
                quality.RssItems.validate(items, lazy=True)
                con.register("_items", items)
                new = con.execute(
                    "SELECT count(*) FROM _items WHERE item_id NOT IN (SELECT item_id FROM bronze.rss_items)"
                ).fetchone()[0]
                con.execute(
                    "INSERT INTO bronze.rss_items SELECT item_id, feed, title, link, published_at, "
                    "_ingested_at FROM _items ON CONFLICT (item_id) DO NOTHING"
                )
                con.unregister("_items")
                if items.published_at.notna().any():
                    prev = get_watermark(con, f"rss:{name}")
                    newest = items.published_at.max()
                    set_watermark(con, f"rss:{name}", max(newest, prev) if prev else newest)
                per_feed[name] = int(new)
                total += int(new)
        t["rows_written"] = total
        t["details"] = per_feed
    return total


def build_warehouse(s: Settings, run_id: str) -> dict:
    with tracked(s.warehouse, run_id, "build_warehouse") as t, connect(s.warehouse) as con:
        boundary = s["historical"]["segment_boundary"]
        stats = {
            "headlines": transform.build_silver_headlines(con, boundary),
            "market_days": transform.build_silver_market(con, s["market"]["symbol"]),
            "target_days": transform.build_gold_targets(con),
        }
        audit = transform.build_label_audit(con, boundary)
        stats["label_audit"] = audit[["segment", "verdict"]].to_dict("records")
        t["rows_written"], t["details"] = stats["headlines"], stats
    return stats


def build_features(s: Settings, run_id: str) -> dict:
    with tracked(s.warehouse, run_id, "build_features") as t, connect(s.warehouse) as con:
        scored = features.update_sentiment(con)
        days = features.build_daily_features(
            con, s["features"]["version"], s["features"]["max_headlines_per_day"]
        )
        t["rows_written"], t["details"] = days, {"newly_scored_headlines": scored}
    return {"newly_scored": scored, "feature_days": days}


def check_quality(s: Settings, run_id: str) -> dict:
    with tracked(s.warehouse, run_id, "check_quality") as t, connect(s.warehouse) as con:
        results = quality.run_checks(con, run_id)
        t["details"] = results
    return results
