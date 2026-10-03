"""Data quality: pandera schemas for in-memory frames + SQL checks on warehouse tables.

Every check result is appended to ops.dq_results. `run_checks` raises if a blocking
check fails, which stops the Airflow task (and therefore everything downstream).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import duckdb
import pandera.pandas as pa

log = logging.getLogger("newspipe.quality")

# ----------------------------------------------------------------- frame schemas
HistoricalBronze = pa.DataFrameSchema(
    {
        "trade_date": pa.Column("object", nullable=False),
        "headline_rank": pa.Column(int, pa.Check.in_range(1, 25)),
        "headline_raw": pa.Column(str, pa.Check.str_length(min_value=1)),
        "provided_label": pa.Column(int, pa.Check.isin([0, 1])),
    },
    unique=["trade_date", "headline_rank"],
    strict=False,
)

MarketBars = pa.DataFrameSchema(
    {
        "symbol": pa.Column(str),
        "close": pa.Column(float, pa.Check.gt(0)),
        "open": pa.Column(float, pa.Check.gt(0), nullable=True),
        "high": pa.Column(float, nullable=True),
        "low": pa.Column(float, nullable=True),
    },
    checks=[
        pa.Check(lambda d: (d["high"].isna() | (d["high"] >= d["low"])).all(), name="high_gte_low")
    ],
    unique=["symbol", "trade_date"],
    strict=False,
)

RssItems = pa.DataFrameSchema(
    {
        "item_id": pa.Column(str, unique=True),
        "title": pa.Column(str, pa.Check.str_length(min_value=1)),
        "feed": pa.Column(str),
    },
    strict=False,
)


# ------------------------------------------------------------------- SQL checks
@dataclass
class SqlCheck:
    name: str
    sql: str  # must return a single boolean
    blocking: bool = True


WAREHOUSE_CHECKS = [
    SqlCheck(
        "archive_has_at_least_20_headlines_per_day",
        "SELECT bool_and(n >= 20) FROM (SELECT count(*) n FROM bronze.historical_headlines GROUP BY trade_date)",
    ),
    SqlCheck(
        # 3 source days (2009-09-15, 2009-12-24, 2011-04-21) ship 22-23 headlines.
        "archive_has_exactly_25_headlines_per_day",
        "SELECT bool_and(n = 25) FROM (SELECT count(*) n FROM bronze.historical_headlines GROUP BY trade_date)",
        blocking=False,
    ),
    SqlCheck(
        "every_archive_day_maps_to_a_session",
        """SELECT count(*) = 0 FROM gold.targets t
           JOIN (SELECT DISTINCT trade_date FROM bronze.historical_headlines) a
             ON a.trade_date = t.news_date
           WHERE t.same_day_up IS NULL""",
    ),
    SqlCheck(
        # Longest legitimate closure: 9/11 (NYSE shut 2001-09-11..14 -> 7-day gap).
        # Other >4-day gaps: President Ford's day of mourning (2007-01-02) and
        # Hurricane Sandy (2012-10-29/30). Anything longer means missing bars.
        "market_has_no_gaps_over_7_days",
        "SELECT coalesce(max(date_diff('day', prev_trade_date, trade_date)), 0) <= 7 FROM silver.market_daily",
    ),
    SqlCheck(
        "returns_within_plausible_range",
        "SELECT bool_and(abs(return_1d) < 0.25) FROM silver.market_daily",
    ),
    SqlCheck(
        "bytes_artifacts_removed",
        "SELECT count(*) = 0 FROM silver.headlines WHERE regexp_matches(headline, '^b[''\"]')",
    ),
    SqlCheck(
        "every_headline_scored",
        """SELECT count(*) = 0 FROM silver.headlines h
           ANTI JOIN silver.headline_sentiment s USING (headline_hash)""",
    ),
    SqlCheck(
        "features_one_row_per_day",
        "SELECT count(*) = count(DISTINCT news_date) FROM gold.daily_features",
    ),
    SqlCheck(
        "provided_label_trustworthy",
        "SELECT bool_and(verdict LIKE 'label == same_day%') FROM gold.label_audit",
        blocking=False,  # documented finding, not a pipeline failure - targets are recomputed
    ),
]


class DataQualityError(Exception):
    pass


def run_checks(con: duckdb.DuckDBPyConnection, run_id: str, checks=WAREHOUSE_CHECKS) -> dict:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    results = {}
    for c in checks:
        try:
            passed = bool(con.execute(c.sql).fetchone()[0])
            observed = None
        except duckdb.Error as exc:  # missing table etc. counts as a failure
            passed, observed = False, str(exc)[:200]
        results[c.name] = passed
        con.execute(
            "INSERT INTO ops.dq_results VALUES (?, 'warehouse', ?, ?, ?, ?)",
            [run_id, c.name, passed, json.dumps(observed), now],
        )
        level = logging.INFO if passed else (logging.ERROR if c.blocking else logging.WARNING)
        log.log(level, "dq %-40s %s", c.name, "PASS" if passed else "FAIL")
    blocking_failures = [c.name for c in checks if c.blocking and not results[c.name]]
    if blocking_failures:
        raise DataQualityError(f"blocking data quality checks failed: {blocking_failures}")
    return results
