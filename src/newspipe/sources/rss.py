"""RSS 2.0 feed ingestion (stdlib XML parser - no extra dependency).

Items are keyed by sha1(feed + guid-or-link), so the same article seen on every poll
is stored once. A per-feed watermark (newest `published_at`) is kept in ops.watermarks
for monitoring feed lag.
"""

from __future__ import annotations

import hashlib
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

log = logging.getLogger("newspipe.rss")


def parse_feed(xml_bytes: bytes, feed: str) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)
    rows = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        guid = (item.findtext("guid") or link).strip()
        pub = item.findtext("pubDate")
        if not title:
            continue
        try:
            published = parsedate_to_datetime(pub).astimezone(timezone.utc).replace(tzinfo=None)
        except (TypeError, ValueError):
            published = None
        rows.append(
            {
                "item_id": hashlib.sha1(f"{feed}|{guid}".encode()).hexdigest(),
                "feed": feed,
                "title": title,
                "link": link,
                "published_at": published,
            }
        )
    frame = pd.DataFrame(rows, columns=["item_id", "feed", "title", "link", "published_at"])
    return frame.drop_duplicates("item_id")


def fetch_feed(url: str, feed: str, timeout: int = 20) -> pd.DataFrame:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "newspipe/0.1"})
    resp.raise_for_status()
    frame = parse_feed(resp.content, feed)
    frame["_ingested_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    return frame
