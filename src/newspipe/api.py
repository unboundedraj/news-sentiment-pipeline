"""FastAPI service: headline sentiment, champion-model predictions and stored signals.

uvicorn newspipe.api:app --reload        # docs at http://localhost:8000/docs
"""

from __future__ import annotations

from functools import lru_cache

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from newspipe import __version__
from newspipe.config import load_settings
from newspipe.features import SERVING_COLUMNS, score_headlines
from newspipe.text import clean_headline, normalize

app = FastAPI(
    title="newspipe",
    version=__version__,
    description="Headline sentiment and market-direction signals.",
)


class Headlines(BaseModel):
    headlines: list[str] = Field(
        ..., min_length=1, max_length=200, examples=[["Markets rally as inflation cools"]]
    )


class HeadlineScore(BaseModel):
    headline: str
    compound: float
    label: str


class SentimentResponse(BaseModel):
    items: list[HeadlineScore]
    mean_compound: float


class PredictionResponse(BaseModel):
    prob_up: float
    predicted_up: bool
    model_version: str
    note: str


@lru_cache(maxsize=1)
def _settings():
    return load_settings()


@lru_cache(maxsize=1)
def _champion():
    from newspipe.score import load_champion

    return load_champion(_settings())


def _label(c: float) -> str:
    return "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"


def _day_features(headlines: list[str]) -> pd.DataFrame:
    cleaned = pd.Series([clean_headline(h) for h in headlines])
    scores = score_headlines(cleaned)
    c = scores["compound"]
    row = {
        "doc_text": " ".join(normalize(h) for h in cleaned),
        "n_headlines": float(len(cleaned)),
        "sent_mean": c.mean(),
        "sent_min": c.min(),
        "sent_max": c.max(),
        "sent_std": c.std(ddof=1) if len(c) > 1 else 0.0,
        "frac_negative": (c <= -0.05).mean(),
        "frac_positive": (c >= 0.05).mean(),
        "neg_mean": scores["neg"].mean(),
        "pos_mean": scores["pos"].mean(),
    }
    return pd.DataFrame([row])[SERVING_COLUMNS]


@app.get("/health")
def health() -> dict:
    model, version = _champion()
    return {"status": "ok", "version": __version__, "champion_model_version": version}


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment(body: Headlines) -> SentimentResponse:
    cleaned = pd.Series([clean_headline(h) for h in body.headlines])
    scores = score_headlines(cleaned)
    items = [
        HeadlineScore(headline=h, compound=round(float(c), 4), label=_label(c))
        for h, c in zip(cleaned, scores["compound"], strict=True)
    ]
    return SentimentResponse(items=items, mean_compound=round(float(scores["compound"].mean()), 4))


@app.post("/predict", response_model=PredictionResponse)
def predict(body: Headlines) -> PredictionResponse:
    model, version = _champion()
    if model is None:
        raise HTTPException(503, "no champion model registered yet - run `newspipe train`")
    proba = float(model.predict_proba(_day_features(body.headlines))[:, 1][0])
    return PredictionResponse(
        prob_up=round(proba, 4),
        predicted_up=proba >= 0.5,
        model_version=version,
        note="Walk-forward backtests show no edge over the majority baseline; "
        "treat as a demonstration of the serving path, not trading advice.",
    )


@app.get("/signals")
def signals(limit: int = Query(30, ge=1, le=1000)) -> list[dict]:
    try:
        with duckdb.connect(str(_settings().warehouse), read_only=True) as con:
            rows = con.execute(
                "SELECT * FROM gold.signals ORDER BY news_date DESC LIMIT ?", [limit]
            ).df()
    except duckdb.Error:
        return []
    return rows.astype({"news_date": str, "scored_at": str}).to_dict("records")
