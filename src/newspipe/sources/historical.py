"""Historical headline archive: one wide CSV (Date, Label, Top1..Top25).

Ingestion reshapes it to a long, keyed table (one row per date x rank) so it can be
upserted, validated and joined like any other feed. The provided `Label` is kept as-is
in bronze; whether it can be trusted is decided later by the label audit.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("newspipe.historical")


def fetch(url: str, raw_dir: Path, expected_sha256: str | None = None, retries: int = 3) -> Path:
    """Download once into the raw zone; re-use the local copy if its checksum matches."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "historical_headlines.csv"
    if target.exists():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if expected_sha256 in (None, digest):
            return target
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    digest = hashlib.sha256(resp.content).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"checksum mismatch for {url}: {digest}")
    target.write_bytes(resp.content)
    log.info("downloaded %s (%d bytes, sha256=%s)", url, len(resp.content), digest[:12])
    return target


def to_long(csv_bytes: bytes, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """Wide (Top1..Top25 columns) -> long (trade_date, headline_rank, headline_raw)."""
    wide = pd.read_csv(io.BytesIO(csv_bytes), encoding=encoding, dtype=str)
    top_cols = [c for c in wide.columns if c.startswith("Top")]
    long = wide.melt(
        id_vars=["Date", "Label"], value_vars=top_cols, var_name="col", value_name="headline_raw"
    )
    long["headline_rank"] = long["col"].str.removeprefix("Top").astype(int)
    long["trade_date"] = pd.to_datetime(long["Date"]).dt.date
    long["provided_label"] = long["Label"].astype(int)
    long = long.dropna(subset=["headline_raw"])
    return long[["trade_date", "headline_rank", "headline_raw", "provided_label"]].sort_values(
        ["trade_date", "headline_rank"], ignore_index=True
    )


def ingest(path: Path, encoding: str) -> pd.DataFrame:
    content = path.read_bytes()
    frame = to_long(content, encoding)
    frame["_source_sha256"] = hashlib.sha256(content).hexdigest()
    frame["_ingested_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    return frame
