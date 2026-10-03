# Design decisions

### 1. Never trust a label you didn't compute
The archive's label was the riskiest input, so targets are rebuilt from raw market bars with
an explicit alignment (same session or next session). That's how the label shift described
in [`label_audit.md`](label_audit.md) came to light.

### 2. Walk-forward, not a single split
One train/test split gives one noisy number. Thirteen yearly folds (train on the past, test
on the next year) give 3,138 out-of-sample days, which is enough for confidence intervals
and a significance test against the baseline. Shuffled cross-validation would leak the
future into training.

### 3. Report baselines and uncertainty next to every model
A 52.9% accuracy only means something next to a 53.3% majority class and a ±1.7-point
interval. The report always includes both, plus a binomial test.

### 4. Long-format bronze
The archive arrives wide (Top1..Top25). Reshaping to (date, rank, headline) makes it
upsertable, joinable with RSS items, and lets VADER scores be cached per headline instead
of per day.

### 5. Content-addressed sentiment cache
VADER is cheap, but a daily job shouldn't re-score 100k headlines. Keying scores by a hash
of the normalised headline makes rescoring incremental and de-duplicates repeated
headlines for free.

### 6. Airflow for orchestration, plain Python for logic
DAGs stay thin, and all logic lives in `newspipe.tasks`. That keeps the pipeline testable
without a scheduler and runnable from the CLI. The same tasks run under Airflow in Docker.

### 7. MLflow aliases instead of stages
Registry stages are deprecated. The `@champion` alias is the deployment contract, so both
the scorer and the API resolve it at load time, and promotion is one atomic alias move.

### 8. One volume, one UID
Airflow, MLflow and the API share a single volume (DuckDB + SQLite MLflow store) and run as
UID 50000. That's the simplest setup that works locally. In production, the warehouse and
tracking store would be network services (Postgres, S3 and a cloud warehouse).

### 9. The quality gate distinguishes blocking from known issues
Implausible returns, unscored headlines or leftover scraping artifacts stop the DAG.
Documented source quirks (three short days, the label shift) are recorded on every run but
don't block, so the warnings stay visible without paging anyone.
