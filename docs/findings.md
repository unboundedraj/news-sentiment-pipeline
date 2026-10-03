# Findings

## 1. Headlines carry no exploitable signal for next-day DJIA direction

There are 13 walk-forward folds (test years 2004–2016, each trained only on prior years),
3,138 out-of-sample days in total, and the target is recomputed and causal (`next_day_up`).
Full table: [`reports/backtest_next_day_up.md`](../reports/backtest_next_day_up.md).

| Model | Accuracy | 95% CI | Balanced acc. | ROC-AUC |
|---|---:|---|---:|---:|
| Majority class (baseline) | 53.3% | 51.5–55.0 | 0.504 | 0.496 |
| Bigram RandomForest (original approach) | 52.9% | 51.2–54.7 | 0.510 | 0.502 |
| VADER sentiment + logistic regression | 52.8% | 51.1–54.5 | 0.504 | 0.488 |
| TF-IDF 1–2-grams + logistic regression | 52.3% | 50.6–54.1 | 0.504 | 0.502 |
| Momentum (repeat today's move) | 46.9% | 45.1–48.6 | 0.466 | 0.466 |

Every text model's confidence interval overlaps the majority baseline, and AUC is about 0.5.
That's what an efficient market predicts for public, same-day news: by the time a headline
is published, the price has already moved. The "52% accuracy" in the original project is
this baseline, not a model effect.

The pipeline can still detect signal when it exists. `tests/test_train_and_api.py`
plants a learnable pattern and asserts the same pipeline recovers it (more than 95%
accuracy, p < 0.001), so the null result isn't caused by a broken pipeline.

## 2. Daily DJIA moves mean-revert in this period

"Repeat today's direction" scores **46.9%** (CI 45.1–48.6), significantly *below* 50%.
Over 2004–2016, a down day was slightly more often followed by an up day and vice versa.
This mild negative lag-1 autocorrelation is the only statistically clear pattern here, and
it comes from prices, not headlines.

## 3. The label column mixes two definitions

See [`label_audit.md`](label_audit.md). Before 2008-08-08 the provided label is the
*previous* day's move (98.8% agreement). After that it's the same-day move (99.9%).

## 4. Source quality issues found by the checks

| Issue | Count | Handling |
|---|---:|---|
| Byte-literal scraping artifacts (`b'...'`) | 11,914 headlines | cleaned in silver, guarded by a check |
| Days with fewer than 25 headlines | 3 (2009-09-15, 2009-12-24, 2011-04-21) | non-blocking check |
| Duplicate headlines within a day | about 200 | de-duplicated by normalised hash |
| Market closures longer than 4 days | 9/11 (7 days), Ford mourning, Hurricane Sandy | encoded as known closures in the gap check |
