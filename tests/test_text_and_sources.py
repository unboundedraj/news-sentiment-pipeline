import json
from datetime import date
from pathlib import Path

import pandas as pd

from newspipe.sources import historical, market, rss
from newspipe.text import clean_headline, headline_hash, normalize

FIXTURES = Path(__file__).parent / "fixtures"


def test_clean_headline_strips_byte_literals_and_entities():
    assert clean_headline("b\"Georgia 'downs two Russian warplanes'\"") == (
        "Georgia 'downs two Russian warplanes'"
    )
    assert clean_headline("b'Why won\\'t America help?'") == "Why won't America help?"
    assert clean_headline("b'Caf\\xc3\\xa9 owners protest'") == "Café owners protest"
    assert clean_headline("Tom &amp; Jerry   return") == "Tom & Jerry return"
    assert clean_headline("bold move by bank") == "bold move by bank"  # not a literal
    assert clean_headline(None) == ""


def test_normalize_and_hash_are_stable_across_artifacts():
    assert normalize("Stocks FALL 3%!") == "stocks fall 3"
    assert headline_hash("b'Stocks fall'") != headline_hash("Stocks fall")
    assert headline_hash(clean_headline("b'Stocks fall'")) == headline_hash("stocks  FALL")


def test_historical_wide_to_long():
    csv = b"Date,Label,Top1,Top2,Top3\n2000-01-03,0,a,b,\n2000-01-04,1,c,d,e\n"
    long = historical.to_long(csv)
    assert len(long) == 5  # missing Top3 on day 1 is dropped
    assert list(long.columns) == ["trade_date", "headline_rank", "headline_raw", "provided_label"]
    assert long.groupby("trade_date").provided_label.first().tolist() == [0, 1]


def test_market_parse_chart_uses_exchange_dates_and_drops_null_bars():
    payload = json.loads((FIXTURES / "yahoo_chart.json").read_text())
    bars = market.parse_chart(payload, "^DJI")
    assert bars.trade_date.astype(str).tolist() == ["2016-06-29", "2016-06-30"]
    assert bars.close.notna().all()


def test_incremental_window_refetches_overlap():
    start = date(2000, 1, 1)
    assert market.incremental_window(None, start, 5, today=date(2020, 1, 1)) == (
        start,
        date(2020, 1, 1),
    )
    assert market.incremental_window(date(2020, 1, 10), start, 5, today=date(2020, 1, 12)) == (
        date(2020, 1, 5),
        date(2020, 1, 12),
    )


def test_rss_parse_dedupes_and_tolerates_bad_dates():
    items = rss.parse_feed((FIXTURES / "feed.xml").read_bytes(), "test")
    assert len(items) == 3  # one exact duplicate guid removed, one empty title skipped
    assert items.item_id.is_unique
    assert items.published_at.isna().sum() == 1
    assert pd.Timestamp("2026-10-01 09:30:00") in set(items.published_at)
