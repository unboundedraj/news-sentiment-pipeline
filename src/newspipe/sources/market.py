"""Daily index bars with incremental, watermark-driven fetching.

Each run fetches from (last stored trade date - overlap) to today and upserts by
(symbol, trade_date), so late corrections from the provider replace earlier values
and re-runs are idempotent.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

log = logging.getLogger("newspipe.market")

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def fetch_yahoo(symbol: str, start: date, end: date, retries: int = 3) -> pd.DataFrame:
    params = {
        "period1": int(
            datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp()
        ),
        "period2": int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp())
        + 86400,
        "interval": "1d",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                YAHOO_CHART.format(symbol=requests.utils.quote(symbol)),
                params=params,
                headers={"User-Agent": "Mozilla/5.0 (newspipe)"},
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
            break
        except (requests.RequestException, ValueError):
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    return parse_chart(payload, symbol)


def parse_chart(payload: dict, symbol: str) -> pd.DataFrame:
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    tz_offset = result["meta"].get("gmtoffset", 0)
    # Bars are stamped at the session open in UTC; shift to exchange-local time for the date.
    ts = pd.to_datetime(result.get("timestamp", []), unit="s") + pd.to_timedelta(tz_offset, "s")
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": ts.date,
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        }
    )
    # Yahoo returns null bars for some holidays/partial days - those are not trading days.
    return frame.dropna(subset=["close"]).drop_duplicates(["symbol", "trade_date"], keep="last")


def incremental_window(
    last_stored: date | None, start: date, overlap_days: int, today: date | None = None
) -> tuple[date, date]:
    today = today or datetime.now(timezone.utc).date()
    begin = start if last_stored is None else max(start, last_stored - timedelta(days=overlap_days))
    return begin, today
