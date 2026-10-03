"""Reproduce the original notebook protocol on the same data, for comparison.

Original protocol: provided labels, train < 2015, test >= 2015, punctuation stripped and
lower-cased bag of bigrams, RandomForest(200, entropy). The notebook only applied the
cleaning to the *training* text; both variants are reported.
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

from newspipe.config import Settings
from newspipe.warehouse import connect


def _clean(text: str) -> str:
    return re.sub("[^a-zA-Z]", " ", text).lower()


def reproduce_original(s: Settings, seed: int = 1) -> dict[str, float]:
    with connect(s.warehouse) as con:
        days = con.execute(
            """
            SELECT trade_date, provided_label,
                   string_agg(headline_raw, ' ' ORDER BY headline_rank) AS doc
            FROM bronze.historical_headlines
            GROUP BY 1, 2 ORDER BY 1
            """
        ).df()
    days["trade_date"] = pd.to_datetime(days.trade_date)
    train = days[days.trade_date < "2015-01-01"]
    test = days[days.trade_date >= "2015-01-01"]
    vec = CountVectorizer(ngram_range=(2, 2))
    X = vec.fit_transform(train.doc.map(_clean))
    rf = RandomForestClassifier(200, criterion="entropy", n_jobs=-1, random_state=seed)
    rf.fit(X, train.provided_label)
    majority = max(test.provided_label.mean(), 1 - test.provided_label.mean())
    return {
        "test_days": len(test),
        "accuracy_test_uncleaned (as in notebook)": accuracy_score(
            test.provided_label, rf.predict(vec.transform(test.doc))
        ),
        "accuracy_test_cleaned (bug fixed)": accuracy_score(
            test.provided_label, rf.predict(vec.transform(test.doc.map(_clean)))
        ),
        "majority_baseline": majority,
    }
