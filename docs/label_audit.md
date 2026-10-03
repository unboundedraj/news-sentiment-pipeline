# Label lineage audit: the archive's label is not one thing

The historical archive (4,101 trading days, 2000-01-03 to 2016-07-01, 25 headlines per day)
ships with a binary `Label`, usually described as "1 if the DJIA rose or stayed the same that
day". The pipeline never trusts that column. It recomputes every target from raw daily DJIA
bars and then measures how the provided label lines up (`gold.label_audit`):

| Segment | Dates | Days | Label = same-day DJIA move | Label = **previous**-day move | Label = next-day move |
|---|---|---:|---:|---:|---:|
| `archive_uk_press` | 2000-01-03 to 2008-08-07 | 2,112 | 46.7% | **98.8%** | 49.0% |
| `archive_reddit_worldnews` | 2008-08-08 to 2016-07-01 | 1,989 | **99.9%** | 47.7% | 47.7% |

## What this means

1. **The file stitches together two different sources.** Before 2008-08-08 the headlines are
   UK newspaper items (sport, culture and politics). From 2008-08-08 they are Reddit
   r/worldnews top posts, still wrapped in Python byte-literal artifacts such as
   `b"Georgia 'downs two Russian warplanes'..."`. 11,914 headlines carry that artifact. The
   pipeline removes it and the `bytes_artifacts_removed` check guards the result.
2. **In the first segment the label is shifted by one trading day.** It describes the
   *previous* session's move. A morning newspaper reports yesterday's events, so a model
   trained on this segment learns "do headlines describe yesterday's market", and then gets
   evaluated after 2008, where the label means something else entirely.
3. **Any model trained on the raw label mixes two targets.** The original notebook
   (train before 2015, test from 2015) does exactly that.

## How the pipeline handles it

- Targets live in `gold.targets` and are **recomputed from market bars** with explicit
  alignment:
  - `same_day_up`: the session on the news date (a nowcast)
  - `next_day_up`: the first session *after* the news date (a genuine forecast, and the
    default target)
- `provided_label_trustworthy` is a **non-blocking** check. The finding is recorded in
  `ops.dq_results` on every run instead of failing the pipeline, because the pipeline no
  longer depends on that column.
- `newspipe status` prints the audit, and `newspipe reproduce` re-runs the original protocol
  for comparison.

## Reproducing the original result

`newspipe reproduce` (bag of bigrams, RandomForest(200, entropy), train before 2015,
test from 2015, provided labels):

| Variant | Test accuracy | Majority baseline |
|---|---:|---:|
| Test text left uncleaned (as in the notebook) | 50.5% | 50.8% |
| Test text cleaned the same way as training | 50.3% | 50.8% |

Both sit inside the noise of the majority-class baseline on 378 test days. That matches the
roughly 52% reported for the original project, and the walk-forward backtest gives the
reason (see [`findings.md`](findings.md)).
