"""Candidate models, all as sklearn Pipelines over the gold.daily_features frame.

Every candidate consumes the same DataFrame (doc_text + numeric sentiment features), so
the model registry can serve whichever wins without special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from newspipe.features import NUMERIC_FEATURES


class MajorityClass(BaseEstimator, ClassifierMixin):
    """Always predicts the most frequent training label."""

    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p_), np.full(n, self.p_)])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class Momentum(BaseEstimator, ClassifierMixin):
    """Predicts that the market repeats its most recent known direction."""

    def __init__(self, column: str = "last_known_up"):
        self.column = column

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = X[self.column].fillna(0.5).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass
class Candidate:
    name: str
    description: str
    estimator: Any
    servable: bool = True


def _text(vectorizer) -> ColumnTransformer:
    return ColumnTransformer([("text", vectorizer, "doc_text")])


def candidates(seed: int) -> list[Candidate]:
    return [
        Candidate("majority_class", "always the majority label (baseline)", MajorityClass(), False),
        Candidate(
            "momentum", "repeat the latest known market direction (baseline)", Momentum(), False
        ),
        Candidate(
            "bigram_random_forest",
            "original approach: bag of bigrams + RandomForest(200, entropy)",
            Pipeline(
                [
                    ("features", _text(CountVectorizer(ngram_range=(2, 2)))),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200, criterion="entropy", n_jobs=-1, random_state=seed
                        ),
                    ),
                ]
            ),
        ),
        Candidate(
            "tfidf_logreg",
            "TF-IDF uni+bi-grams (min_df=3) + L2 logistic regression",
            Pipeline(
                [
                    (
                        "features",
                        _text(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)),
                    ),
                    ("model", LogisticRegression(C=0.5, max_iter=2000)),
                ]
            ),
        ),
        Candidate(
            "vader_logreg",
            "daily VADER sentiment aggregates + logistic regression",
            Pipeline(
                [
                    ("features", ColumnTransformer([("num", StandardScaler(), NUMERIC_FEATURES)])),
                    ("model", LogisticRegression(C=1.0, max_iter=1000)),
                ]
            ),
        ),
    ]


def add_market_context(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """The latest market direction known when the prediction is made (for Momentum)."""
    out = frame.copy()
    out["last_known_up"] = out["same_day_up"] if target == "next_day_up" else out["prev_day_up"]
    return out
