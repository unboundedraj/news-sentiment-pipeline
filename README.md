# News Sentiment Pipeline

[![ci](https://github.com/unboundedraj/news-sentiment-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/unboundedraj/news-sentiment-pipeline/actions/workflows/ci.yml)
![stack](https://img.shields.io/badge/stack-Airflow%20·%20DuckDB%20·%20MLflow%20·%20FastAPI%20·%20pandera-informational)

A production-style pipeline that turns news headlines into market-direction signals. It
ingests a 16-year headline archive, live RSS feeds and DJIA market data incrementally,
audits label lineage, keeps a versioned feature store, runs walk-forward backtests tracked
in MLflow, promotes models through a champion/challenger registry, and serves predictions
over FastAPI. It's orchestrated by Airflow in Docker.

> Built on top of the original stock-sentiment analysis by
> [@tfkhushij](https://github.com/tfkhushij/Stock-Sentiment-Analysis) (bag-of-bigrams +
> Random Forest on 4,101 days of top-25 headlines). That model is reproduced here as one of
> the candidates.

## Headline result: a data-lineage bug, found by the pipeline

The archive's label was never used directly. The pipeline **recomputes targets from raw
DJIA bars** and audits the provided label against them:

| Archive segment | Provided label matches same-day DJIA move | matches **previous-day** move |
|---|---:|---:|
| 2000-01-03 to 2008-08-07 (UK newspaper headlines) | 46.7% | **98.8%** |
| 2008-08-08 to 2016-07-01 (Reddit r/worldnews) | **99.9%** | 47.7% |

The file stitches together two sources, and **for the first 8 years the label is shifted by
one trading day**. Any model trained on that column learns two different targets at once.
Details: [`docs/label_audit.md`](docs/label_audit.md).

With causal, recomputed targets and a 13-fold walk-forward backtest (3,138 out-of-sample
days), **no text model beats the majority baseline**: bigram RF 52.9% [51.2, 54.7] against
majority 53.3% [51.5, 55.0], with AUC around 0.50. The original project's "52%" is the
baseline, and the reason is efficient markets, not tuning
([`docs/findings.md`](docs/findings.md)). The test suite plants a learnable signal and
checks that the same pipeline recovers it, so the null result isn't a pipeline bug.

## Architecture

```
 headline archive ─┐                                  ┌─▶ gold.targets (recomputed, causal)
 DJIA bars (API) ──┼─▶ bronze (keyed upserts) ─▶ silver ┼─▶ gold.label_audit
 RSS feeds ────────┘   watermarks, checksums     │      └─▶ gold.daily_features (versioned)
                                                  │                 │
                     VADER sentiment cache ◀──────┘          pandera + SQL quality gate
                     (scores only new headlines)                    │
                                                    walk-forward backtest ─▶ MLflow registry (@champion)
                                                                                  │
                                                     gold.signals (model lineage) ◀┴▶ FastAPI
                 ▲──────────── Airflow: news_backfill · news_daily · model_retrain ────────────▲
```

More: [`docs/architecture.md`](docs/architecture.md) · [`docs/decisions.md`](docs/decisions.md)

## Data engineering features

- **Three ingestion patterns:** a checksum-pinned file archive (wide-to-long reshape),
  **watermarked API increments** with a refetch overlap for revised bars, and **RSS
  polling** with content-addressed ids, `ON CONFLICT DO NOTHING` and per-feed watermarks.
  One failing feed never blocks the others.
- **Idempotent everywhere:** keyed upserts, incremental sentiment (only unseen headline
  hashes are scored), and scoring only days the current champion hasn't scored yet.
- **Data quality:** pandera schemas on every ingested frame and a SQL quality gate with
  **blocking vs known-issue checks**. Real events like the 9/11 market closure are encoded,
  not ignored, and results are persisted to `ops.dq_results`.
- **Lineage:** `ops.pipeline_runs` keyed by the Airflow run id, MLflow runs tagged with the
  pipeline run and feature version, and every signal stamped with its model version.
- **ML engineering:** walk-forward folds, Wilson confidence intervals, binomial tests
  against the baseline, nested MLflow runs, a signature-typed model registry, and alias-based
  promotion.
- **Orchestration and serving:** three Airflow DAGs (TaskFlow API, retries with backoff,
  an `ExternalTaskSensor`) plus a FastAPI service. All of it runs from `docker compose`.
- **CI:** ruff, pytest on Python 3.10–3.12, and a **DagBag integrity check inside the real
  Airflow image**.

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[tracking,api,dev]"

newspipe backfill     # archive + 16y of DJIA bars -> warehouse -> features -> quality gate (~20 s)
newspipe train        # walk-forward backtest -> MLflow -> register + promote champion (~2.5 min)
newspipe daily        # RSS + market increment -> rebuild -> quality gate -> score new days
newspipe status       # row counts, watermarks, label audit, recent runs
newspipe reproduce    # the original notebook protocol on the same data

mlflow ui --backend-store-uri sqlite:///data/mlflow.db    # http://localhost:5000
uvicorn newspipe.api:app                                  # http://localhost:8000/docs
```

## Quickstart (Docker: Airflow + MLflow + API)

```bash
docker compose up -d --build              # API_PORT=8010 if 8000 is taken
docker compose exec airflow airflow dags trigger news_backfill
docker compose exec airflow airflow dags trigger model_retrain
docker compose exec airflow airflow dags unpause news_daily
```

| Service | URL |
|---|---|
| Airflow | http://localhost:8080 (user `admin`, password in `standalone_admin_password.txt` inside the container) |
| MLflow | http://localhost:5000 |
| API docs | http://localhost:8000/docs |

```bash
curl -X POST localhost:8000/sentiment -H 'content-type: application/json' \
     -d '{"headlines": ["Markets rally as inflation cools", "b\"Panic as banks fail\""]}'
```

## Project layout

```
src/newspipe/
  sources/        historical archive, market bars, RSS connectors
  warehouse.py    DuckDB schema, upserts, watermarks, run ledger
  text.py         headline cleaning (byte-literal artifacts, entities)
  transform.py    silver/gold builds, recomputed targets, label audit
  features.py     incremental VADER cache, versioned daily features
  quality.py      pandera schemas + SQL quality gate
  models.py       baselines, original bigram RF, TF-IDF and sentiment models
  train.py        walk-forward backtest, MLflow tracking, champion promotion
  score.py        incremental champion scoring -> gold.signals
  api.py          FastAPI service
  tasks.py        idempotent tasks shared by CLI and Airflow
dags/             news_backfill, news_daily, model_retrain
docker/           Airflow and app images
reports/          generated backtest reports
tests/            synthetic archive + market fixtures; no network needed
```

## Data sources

- **Headline archive:** combined daily top-25 news headlines with a market label,
  2000–2016 (via the original project's repository; checksum pinned on first download).
- **Market data:** DJIA (`^DJI`) daily bars from Yahoo Finance's public chart endpoint.
- **Live news:** BBC World, BBC Business and Guardian World RSS feeds.

Nothing here is investment advice. The point of the project is the pipeline, and its
honest finding is that this signal doesn't exist.
