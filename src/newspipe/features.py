"""Feature store.

silver.headline_sentiment   VADER scores per distinct headline (keyed by headline_hash).
                            Computed incrementally: only hashes never seen before are scored.
gold.daily_features         one row per news day: sentiment aggregates + the day's text,
                            tagged with a feature_version so models can pin a definition.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import duckdb
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger("newspipe.features")

SENTIMENT_DDL = """
CREATE TABLE IF NOT EXISTS silver.headline_sentiment (
    headline_hash VARCHAR PRIMARY KEY,
    compound DOUBLE, pos DOUBLE, neg DOUBLE, neu DOUBLE,
    scored_at TIMESTAMP
)
"""

NUMERIC_FEATURES = [
    "n_headlines",
    "sent_mean",
    "sent_min",
    "sent_max",
    "sent_std",
    "frac_negative",
    "frac_positive",
    "neg_mean",
    "pos_mean",
]

# Columns a served model receives (everything else in the training frame is label/context).
SERVING_COLUMNS = ["doc_text", *NUMERIC_FEATURES]

_analyzer: SentimentIntensityAnalyzer | None = None


def analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score_headlines(texts: pd.Series) -> pd.DataFrame:
    scores = [analyzer().polarity_scores(t) for t in texts]
    return pd.DataFrame(scores, index=texts.index)[["compound", "pos", "neg", "neu"]]


def update_sentiment(con: duckdb.DuckDBPyConnection) -> int:
    """Score only headlines that have not been scored before (incremental)."""
    con.execute(SENTIMENT_DDL)
    todo = con.execute(
        """
        SELECT DISTINCT h.headline_hash, first(h.headline) AS headline
        FROM silver.headlines h
        ANTI JOIN silver.headline_sentiment s USING (headline_hash)
        GROUP BY h.headline_hash
        """
    ).df()
    if todo.empty:
        return 0
    scored = pd.concat([todo[["headline_hash"]], score_headlines(todo["headline"])], axis=1)
    scored["scored_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    con.register("_scored", scored)
    con.execute("INSERT INTO silver.headline_sentiment SELECT * FROM _scored")
    con.unregister("_scored")
    log.info("scored %d new headlines", len(scored))
    return len(scored)


def build_daily_features(con: duckdb.DuckDBPyConnection, version: str, max_per_day: int) -> int:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE gold.daily_features AS
        WITH ranked AS (
            SELECT h.*, s.compound, s.pos, s.neg,
                   row_number() OVER (PARTITION BY news_date
                                      ORDER BY source, headline_rank) AS rn
            FROM silver.headlines h
            JOIN silver.headline_sentiment s USING (headline_hash)
        )
        SELECT
            news_date,
            '{version}'                                         AS feature_version,
            string_agg(DISTINCT source, ',')                    AS sources,
            count(*)                                            AS n_headlines,
            avg(compound)                                       AS sent_mean,
            min(compound)                                       AS sent_min,
            max(compound)                                       AS sent_max,
            coalesce(stddev_samp(compound), 0)                  AS sent_std,
            avg((compound <= -0.05)::INT)                       AS frac_negative,
            avg((compound >= 0.05)::INT)                        AS frac_positive,
            avg(neg)                                            AS neg_mean,
            avg(pos)                                            AS pos_mean,
            string_agg(headline_norm, ' ' ORDER BY rn)          AS doc_text
        FROM ranked
        WHERE rn <= {int(max_per_day)}
        GROUP BY news_date
        ORDER BY news_date
        """
    )
    return con.execute("SELECT count(*) FROM gold.daily_features").fetchone()[0]


def training_frame(con: duckdb.DuckDBPyConnection, target: str) -> pd.DataFrame:
    """Model-ready rows: features joined to a recomputed target (rows without it dropped)."""
    if target not in {"next_day_up", "same_day_up"}:
        raise ValueError(f"unsupported target {target}")
    return con.execute(
        f"""
        SELECT f.*, t.{target} AS target, t.same_day_up, t.prev_day_up
        FROM gold.daily_features f
        JOIN gold.targets t USING (news_date)
        WHERE t.{target} IS NOT NULL
          AND f.sources LIKE 'archive%'
        ORDER BY news_date
        """
    ).df()
