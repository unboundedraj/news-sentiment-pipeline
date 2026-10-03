"""Daily scoring with the registered champion model.

Scores every feature day that has not yet been scored by the current champion version
(incremental), and records the version in gold.signals so every prediction is traceable
back to the MLflow run that produced the model.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import pandas as pd

from newspipe.config import Settings
from newspipe.features import SERVING_COLUMNS
from newspipe.warehouse import connect

log = logging.getLogger("newspipe.score")

SIGNALS_DDL = """
CREATE TABLE IF NOT EXISTS gold.signals (
    news_date      DATE,
    model_name     VARCHAR,
    model_version  VARCHAR,
    prob_up        DOUBLE,
    predicted_up   INTEGER,
    sources        VARCHAR,
    scored_at      TIMESTAMP,
    PRIMARY KEY (news_date, model_name, model_version)
)
"""


def load_champion(s: Settings):
    """Return (sklearn model, version) for the `champion` alias, or (None, None)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(s.mlflow_uri)
    name = s["training"]["registered_model"]
    try:
        mv = MlflowClient().get_model_version_by_alias(name, "champion")
    except Exception:
        return None, None
    return mlflow.sklearn.load_model(f"models:/{name}@champion"), str(mv.version)


def score_new_days(s: Settings, only_live: bool = True) -> int:
    model, version = load_champion(s)
    if model is None:
        log.warning("no champion model registered - skipping scoring")
        return 0
    name = s["training"]["registered_model"]
    with connect(s.warehouse) as con:
        con.execute(SIGNALS_DDL)
        live = "AND f.sources NOT LIKE 'archive%'" if only_live else ""
        todo = con.execute(
            f"""
            SELECT f.* FROM gold.daily_features f
            WHERE NOT EXISTS (
                SELECT 1 FROM gold.signals g
                WHERE g.news_date = f.news_date AND g.model_name = ? AND g.model_version = ?)
            {live}
            ORDER BY news_date
            """,
            [name, version],
        ).df()
        if todo.empty:
            return 0
        X = todo[SERVING_COLUMNS].astype({"n_headlines": "float64"})
        proba = model.predict_proba(X)[:, 1]
        out = pd.DataFrame(
            {
                "news_date": todo.news_date,
                "model_name": name,
                "model_version": version,
                "prob_up": proba,
                "predicted_up": (proba >= 0.5).astype(int),
                "sources": todo.sources,
                "scored_at": datetime.now(timezone.utc).replace(tzinfo=None),
            }
        )
        con.register("_signals", out)
        con.execute("INSERT OR REPLACE INTO gold.signals SELECT * FROM _signals")
        con.unregister("_signals")
    log.info("scored %d day(s) with %s v%s", len(out), name, version)
    return len(out)
