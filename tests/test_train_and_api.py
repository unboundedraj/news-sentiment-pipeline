import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from newspipe.models import MajorityClass, Momentum
from newspipe.train import run_backtest, summarise, walk_forward, wilson_interval


def _frame(n_years=4, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=250 * n_years)
    y = rng.integers(0, 2, len(dates))
    words = np.where(y == 1, "rally gains optimism", "slump fears panic") if signal else "news"
    noise = rng.normal(0, 1, len(dates))
    return pd.DataFrame(
        {
            "news_date": dates.date,
            "doc_text": [
                f"{w} story {i % 7}" for i, w in enumerate(np.broadcast_to(words, len(dates)))
            ],
            "n_headlines": 25.0,
            "sent_mean": noise,
            "sent_min": noise - 1,
            "sent_max": noise + 1,
            "sent_std": 0.3,
            "frac_negative": 0.4,
            "frac_positive": 0.4,
            "neg_mean": 0.1,
            "pos_mean": 0.1,
            "target": y,
            "last_known_up": rng.integers(0, 2, len(dates)),
        }
    )


def test_walk_forward_never_trains_on_the_future():
    frame = _frame()
    folds, preds = walk_forward(frame, 2011, seed=0, only=["majority_class"])
    years = pd.to_datetime(frame.news_date).dt.year
    for f in folds.itertuples():
        assert f.n_train == (years < f.test_year).sum()
        assert f.n_test == (years == f.test_year).sum()
    assert preds.news_date.min() >= pd.Timestamp("2011-01-01").date()


def test_text_model_finds_planted_signal_and_baselines_do_not():
    folds, preds = walk_forward(
        _frame(), 2011, seed=0, only=["majority_class", "momentum", "tfidf_logreg"]
    )
    s = summarise(preds).set_index("model")
    assert s.loc["tfidf_logreg", "accuracy"] > 0.95
    assert s.loc["majority_class", "accuracy"] < 0.6
    assert s.loc["tfidf_logreg", "p_value_vs_majority"] < 0.001


def test_wilson_interval_brackets_rate():
    lo, hi = wilson_interval(520, 1000)
    assert lo < 0.52 < hi and round(hi - lo, 3) == 0.062


def test_baselines():
    X = pd.DataFrame({"last_known_up": [1, 0, np.nan]})
    assert Momentum().fit(X, [0, 1, 1]).predict(X).tolist() == [1, 0, 1]
    assert MajorityClass().fit(X, [1, 1, 0]).predict(X).tolist() == [1, 1, 1]


def test_backtest_on_pipeline_output(built):
    result = run_backtest(built, "t", only=["majority_class", "vader_logreg"], track=False)
    assert set(result["summary"].model) == {"majority_class", "vader_logreg"}
    assert (built.home / "reports" / "backtest_next_day_up.md").exists()


def test_api_sentiment_and_predict_without_model(settings):
    from newspipe import api

    api._settings.cache_clear()
    api._champion.cache_clear()
    client = TestClient(api.app)

    r = client.post(
        "/sentiment", json={"headlines": ["Panic and fear as markets crash", "b'Markets rally'"]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["items"][1]["headline"] == "Markets rally"
    assert body["items"][0]["label"] == "negative"

    assert client.post("/predict", json={"headlines": ["x"]}).status_code == 503
    assert client.get("/signals").json() == []
    assert client.post("/sentiment", json={"headlines": []}).status_code == 422
