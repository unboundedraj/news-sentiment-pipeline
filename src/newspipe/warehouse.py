"""DuckDB warehouse: schema DDL, connection helper, idempotent upserts, run ledger."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger("newspipe")

DDL = """
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS ops;

-- bronze: as received -------------------------------------------------------
CREATE TABLE IF NOT EXISTS bronze.historical_headlines (
    trade_date      DATE,
    headline_rank   INTEGER,
    headline_raw    VARCHAR,
    provided_label  INTEGER,
    _source_sha256  VARCHAR,
    _ingested_at    TIMESTAMP,
    PRIMARY KEY (trade_date, headline_rank)
);
CREATE TABLE IF NOT EXISTS bronze.market_bars (
    symbol       VARCHAR,
    trade_date   DATE,
    open         DOUBLE,
    high         DOUBLE,
    low          DOUBLE,
    close        DOUBLE,
    volume       DOUBLE,
    _provider    VARCHAR,
    _ingested_at TIMESTAMP,
    PRIMARY KEY (symbol, trade_date)
);
CREATE TABLE IF NOT EXISTS bronze.rss_items (
    item_id       VARCHAR PRIMARY KEY,   -- sha1 of feed + guid/link
    feed          VARCHAR,
    title         VARCHAR,
    link          VARCHAR,
    published_at  TIMESTAMP,
    _ingested_at  TIMESTAMP
);

-- ops: observability ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ops.pipeline_runs (
    run_id VARCHAR, task VARCHAR, status VARCHAR, started_at TIMESTAMP,
    duration_s DOUBLE, rows_written BIGINT, details JSON
);
CREATE TABLE IF NOT EXISTS ops.watermarks (
    source VARCHAR PRIMARY KEY, high_watermark TIMESTAMP, updated_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ops.dq_results (
    run_id VARCHAR, dataset VARCHAR, check_name VARCHAR, passed BOOLEAN,
    observed VARCHAR, checked_at TIMESTAMP
);
"""


@contextmanager
def connect(path: Path, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        if not read_only:
            con.execute(DDL)
        yield con
    finally:
        con.close()


def upsert(con: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame) -> int:
    """INSERT OR REPLACE by primary key - re-running a load never duplicates rows."""
    if frame.empty:
        return 0
    con.register("_upsert_frame", frame)
    cols = ", ".join(frame.columns)
    con.execute(f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM _upsert_frame")
    con.unregister("_upsert_frame")
    return len(frame)


def get_watermark(con: duckdb.DuckDBPyConnection, source: str) -> datetime | None:
    row = con.execute(
        "SELECT high_watermark FROM ops.watermarks WHERE source = ?", [source]
    ).fetchone()
    return row[0] if row else None


def set_watermark(con: duckdb.DuckDBPyConnection, source: str, value: datetime) -> None:
    con.execute(
        "INSERT OR REPLACE INTO ops.watermarks VALUES (?, ?, now()::TIMESTAMP)", [source, value]
    )


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


@contextmanager
def tracked(warehouse: Path, run_id: str, task: str) -> Iterator[dict]:
    """Record a task's outcome in ops.pipeline_runs (also on failure)."""
    info: dict = {"rows_written": None, "details": {}}
    started = datetime.now(timezone.utc).replace(tzinfo=None)
    t0 = time.perf_counter()
    status = "success"
    try:
        yield info
    except Exception as exc:
        status = "failed"
        info["details"]["error"] = repr(exc)
        raise
    finally:
        with connect(warehouse) as con:
            con.execute(
                "INSERT INTO ops.pipeline_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    task,
                    status,
                    started,
                    time.perf_counter() - t0,
                    info["rows_written"],
                    json.dumps(info["details"], default=str),
                ],
            )
        log.info(
            "%s %s (%.1fs) rows=%s", task, status, time.perf_counter() - t0, info["rows_written"]
        )
