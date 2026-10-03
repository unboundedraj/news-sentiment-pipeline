"""Silver and gold builds.

silver.headlines        unified, cleaned headlines from every source (archive + RSS)
silver.market_daily     trading days with returns and direction
gold.targets            per news day: same-session and next-session market direction,
                        recomputed from raw bars (never taken from the archive label)
gold.label_audit        how the archive's provided label lines up with the real market
"""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

from newspipe.text import clean_headline, headline_hash, normalize

log = logging.getLogger("newspipe.transform")


def build_silver_headlines(con: duckdb.DuckDBPyConnection, segment_boundary: str) -> int:
    archive = con.execute(
        """
        SELECT trade_date AS news_date, headline_rank, headline_raw,
               CASE WHEN trade_date < CAST(? AS DATE) THEN 'archive_uk_press'
                    ELSE 'archive_reddit_worldnews' END AS source
        FROM bronze.historical_headlines
        """,
        [segment_boundary],
    ).df()
    rss = con.execute(
        """
        SELECT CAST(published_at AS DATE) AS news_date,
               CAST(row_number() OVER (PARTITION BY CAST(published_at AS DATE)
                                       ORDER BY published_at, item_id) AS INTEGER) AS headline_rank,
               title AS headline_raw,
               'rss_' || feed AS source
        FROM bronze.rss_items
        WHERE published_at IS NOT NULL
        """
    ).df()
    frame = pd.concat([archive, rss], ignore_index=True)
    frame["headline"] = frame["headline_raw"].map(clean_headline)
    frame["headline_norm"] = frame["headline"].map(normalize)
    frame["headline_hash"] = frame["headline"].map(headline_hash)
    frame["had_bytes_artifact"] = frame["headline_raw"].str.match(r"""^b['"]""")
    frame = frame[frame["headline_norm"] != ""]
    # The archive repeats some headlines within a day (e.g. "has cubie killed fees" x4).
    frame = frame.sort_values(["news_date", "source", "headline_rank"]).drop_duplicates(
        ["news_date", "headline_hash"]
    )
    con.register("_silver", frame.drop(columns=["headline_raw"]))
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.headlines AS
        SELECT news_date, source, headline_rank, headline, headline_norm, headline_hash,
               had_bytes_artifact
        FROM _silver
        ORDER BY news_date, source, headline_rank
        """
    )
    con.unregister("_silver")
    return len(frame)


def build_silver_market(con: duckdb.DuckDBPyConnection, symbol: str) -> int:
    con.execute(
        """
        CREATE OR REPLACE TABLE silver.market_daily AS
        SELECT
            trade_date,
            close,
            lag(close) OVER w                                   AS prev_close,
            close / lag(close) OVER w - 1                       AS return_1d,
            CASE WHEN close >= lag(close) OVER w THEN 1 ELSE 0 END AS direction_up,
            lag(trade_date) OVER w                              AS prev_trade_date,
            lead(trade_date) OVER w                             AS next_trade_date
        FROM bronze.market_bars
        WHERE symbol = ?
        WINDOW w AS (ORDER BY trade_date)
        QUALIFY lag(close) OVER w IS NOT NULL
        """,
        [symbol],
    )
    return con.execute("SELECT count(*) FROM silver.market_daily").fetchone()[0]


def build_gold_targets(con: duckdb.DuckDBPyConnection) -> int:
    """Map every news day to the market sessions around it.

    same_day_up : direction of the session on the news day (null on weekends/holidays)
    next_day_up : direction of the first session strictly after the news day - the only
                  target that is a genuine *forecast* from that day's headlines.
    prev_day_up : direction of the last session strictly before the news day (used only
                  by the label audit)
    """
    con.execute(
        """
        CREATE OR REPLACE TABLE gold.targets AS
        WITH days AS (SELECT DISTINCT news_date FROM silver.headlines)
        SELECT
            d.news_date,
            same.direction_up                                   AS same_day_up,
            same.return_1d                                      AS same_day_return,
            nxt.trade_date                                      AS next_session_date,
            nxt.direction_up                                    AS next_day_up,
            nxt.return_1d                                       AS next_day_return,
            prv.direction_up                                    AS prev_day_up
        FROM days d
        LEFT JOIN silver.market_daily same ON same.trade_date = d.news_date
        ASOF LEFT JOIN silver.market_daily nxt ON nxt.trade_date > d.news_date
        ASOF LEFT JOIN silver.market_daily prv ON prv.trade_date < d.news_date
        ORDER BY d.news_date
        """
    )
    return con.execute("SELECT count(*) FROM gold.targets").fetchone()[0]


def build_label_audit(con: duckdb.DuckDBPyConnection, segment_boundary: str) -> pd.DataFrame:
    """Measure which market move the archive's provided label actually describes."""
    audit = con.execute(
        """
        WITH labels AS (
            SELECT DISTINCT trade_date AS news_date, provided_label,
                   CASE WHEN trade_date < CAST(? AS DATE) THEN 'archive_uk_press'
                        ELSE 'archive_reddit_worldnews' END AS segment
            FROM bronze.historical_headlines
        )
        SELECT
            segment,
            min(l.news_date)                                         AS first_date,
            max(l.news_date)                                         AS last_date,
            count(*)                                                 AS days,
            round(avg((l.provided_label = t.same_day_up)::INT), 4)  AS agree_same_day,
            round(avg((l.provided_label = t.prev_day_up)::INT), 4)  AS agree_prev_day,
            round(avg((l.provided_label = t.next_day_up)::INT), 4)  AS agree_next_day
        FROM labels l
        JOIN gold.targets t USING (news_date)
        WHERE t.same_day_up IS NOT NULL
        GROUP BY segment
        ORDER BY first_date
        """,
        [segment_boundary],
    ).df()

    def verdict(r: pd.Series) -> str:
        best = max(
            ("same_day", r.agree_same_day),
            ("prev_day", r.agree_prev_day),
            ("next_day", r.agree_next_day),
            key=lambda kv: kv[1],
        )
        return f"label == {best[0]} DJIA direction" if best[1] >= 0.95 else "label unexplained"

    audit["verdict"] = audit.apply(verdict, axis=1)
    con.register("_audit", audit)
    con.execute("CREATE OR REPLACE TABLE gold.label_audit AS SELECT * FROM _audit")
    con.unregister("_audit")
    for r in audit.itertuples():
        log.info(
            "label audit %s: same=%.3f prev=%.3f next=%.3f -> %s",
            r.segment,
            r.agree_same_day,
            r.agree_prev_day,
            r.agree_next_day,
            r.verdict,
        )
    return audit
