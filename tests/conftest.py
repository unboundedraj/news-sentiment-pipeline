"""Synthetic, network-free fixtures that reproduce the archive's real quirks:

* two segments split at 2008-08-08, with byte-literal artifacts in the second one
* labels that equal the *previous* session's direction before the split and the
  *same* session's direction after it (what the label audit found in the real file)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newspipe.config import load_settings
from newspipe.sources import market

WORDS_UP = ["rally", "gains", "record high", "optimism", "growth"]
WORDS_DOWN = ["slump", "fears", "crisis", "losses", "panic"]


def make_market(start="2007-01-01", end="2009-12-31", seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    close = 12000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    return pd.DataFrame(
        {
            "symbol": "^DJI",
            "trade_date": days.date,
            "open": close * (1 + rng.normal(0, 0.002, len(days))),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e8,
        }
    )


def make_archive(
    bars: pd.DataFrame, start="2007-06-01", end="2009-06-30", boundary="2008-08-08", seed=5
) -> bytes:
    rng = np.random.default_rng(seed)
    b = bars.copy()
    b["up"] = (b.close.diff() >= 0).astype(int)
    b["prev_up"] = b["up"].shift(1)
    b = b[(b.trade_date >= pd.Timestamp(start).date()) & (b.trade_date <= pd.Timestamp(end).date())]
    rows = []
    for r in b.itertuples():
        after = str(r.trade_date) >= boundary
        label = int(r.up if after else r.prev_up)
        words = WORDS_UP if r.up else WORDS_DOWN
        heads = []
        for i in range(25):
            h = f"Markets {words[rng.integers(len(words))]} as story {i} unfolds"
            heads.append(f"b'{h}'" if after and i % 3 == 0 else h)
        rows.append([str(r.trade_date), label, *heads])
    cols = ["Date", "Label", *[f"Top{i}" for i in range(1, 26)]]
    return pd.DataFrame(rows, columns=cols).to_csv(index=False).encode("ISO-8859-1")


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("NEWSPIPE_HOME", str(tmp_path))
    monkeypatch.delenv("NEWSPIPE_CONFIG", raising=False)
    s = load_settings(home=tmp_path)
    bars = make_market()
    raw = s.path("raw")
    raw.mkdir(parents=True)
    (raw / "historical_headlines.csv").write_bytes(make_archive(bars))
    s.raw["historical"]["sha256"] = None
    s.raw["training"]["first_test_year"] = 2009

    def fake_fetch(symbol, start, end, retries=3):
        return bars[(bars.trade_date >= start) & (bars.trade_date <= end)].reset_index(drop=True)

    monkeypatch.setattr(market, "fetch_yahoo", fake_fetch)
    return s


@pytest.fixture()
def built(settings):
    from newspipe import tasks

    tasks.ingest_historical(settings, "t")
    tasks.ingest_market(settings, "t", today=pd.Timestamp("2009-12-31").date())
    tasks.build_warehouse(settings, "t")
    tasks.build_features(settings, "t")
    return settings
