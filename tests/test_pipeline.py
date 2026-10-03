import duckdb
import pandas as pd
import pytest

from newspipe import tasks
from newspipe.quality import DataQualityError, run_checks
from newspipe.sources import rss
from newspipe.warehouse import connect


def _q(s, sql):
    with duckdb.connect(str(s.warehouse), read_only=True) as con:
        return con.execute(sql).fetchall()


def test_label_audit_detects_shifted_segment(built):
    audit = dict(_q(built, "SELECT segment, verdict FROM gold.label_audit"))
    assert audit == {
        "archive_uk_press": "label == prev_day DJIA direction",
        "archive_reddit_worldnews": "label == same_day DJIA direction",
    }


def test_byte_artifacts_are_cleaned(built):
    flagged, remaining = _q(
        built,
        """SELECT sum(had_bytes_artifact::INT),
                  count(*) FILTER (WHERE regexp_matches(headline, '^b[''"]'))
           FROM silver.headlines""",
    )[0]
    assert flagged > 0 and remaining == 0


def test_targets_are_causal(built):
    bad = _q(
        built,
        """SELECT count(*) FROM gold.targets
           WHERE next_session_date <= news_date""",
    )[0][0]
    assert bad == 0


def test_ingestion_is_idempotent(built):
    before = _q(built, "SELECT count(*) FROM bronze.historical_headlines")[0][0]
    tasks.ingest_historical(built, "again")
    tasks.ingest_market(built, "again", today=pd.Timestamp("2009-12-31").date())
    after = _q(built, "SELECT count(*) FROM bronze.historical_headlines")[0][0]
    bars = _q(built, "SELECT count(*), count(DISTINCT trade_date) FROM bronze.market_bars")[0]
    assert before == after
    assert bars[0] == bars[1]


def test_sentiment_is_incremental(built):
    assert tasks.build_features(built, "again")["newly_scored"] == 0


def test_quality_gate_passes_on_clean_build(built):
    with connect(built.warehouse) as con:
        results = run_checks(con, "t")
    assert results["bytes_artifacts_removed"] and results["every_headline_scored"]
    assert results["provided_label_trustworthy"] is False  # non-blocking, documented


def test_quality_gate_blocks_on_bad_data(built):
    with connect(built.warehouse) as con:
        con.execute(
            "UPDATE silver.market_daily SET return_1d = 0.9 WHERE trade_date = "
            "(SELECT max(trade_date) FROM silver.market_daily)"
        )
        with pytest.raises(DataQualityError, match="returns_within_plausible_range"):
            run_checks(con, "t")


def test_rss_ingestion_dedupes_across_polls(built):
    from pathlib import Path

    xml = (Path(__file__).parent / "fixtures" / "feed.xml").read_bytes()

    def fetcher(url, name):
        frame = rss.parse_feed(xml, name)
        frame["_ingested_at"] = pd.Timestamp.now()
        return frame

    built.raw["rss"]["feeds"] = {"one": "x", "two": "y"}
    first = tasks.ingest_rss(built, "p1", fetcher=fetcher)
    second = tasks.ingest_rss(built, "p2", fetcher=fetcher)
    assert first == 6 and second == 0  # 3 items x 2 feeds, then nothing new
    marks = dict(
        _q(built, "SELECT source, high_watermark FROM ops.watermarks WHERE source LIKE 'rss:%'")
    )
    assert set(marks) == {"rss:one", "rss:two"}


def test_broken_feed_does_not_stop_others(built):
    def fetcher(url, name):
        if name == "bad":
            raise ConnectionError("boom")
        return pd.DataFrame(
            [
                {
                    "item_id": "1",
                    "feed": name,
                    "title": "t",
                    "link": "l",
                    "published_at": pd.Timestamp("2026-01-01"),
                    "_ingested_at": pd.Timestamp.now(),
                }
            ]
        )

    built.raw["rss"]["feeds"] = {"bad": "x", "good": "y"}
    assert tasks.ingest_rss(built, "p", fetcher=fetcher) == 1
    status = _q(built, "SELECT status FROM ops.pipeline_runs WHERE task = 'ingest_rss'")
    assert status[-1][0] == "success"
