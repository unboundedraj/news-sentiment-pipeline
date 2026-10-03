PY ?= python

.PHONY: install lint fmt test backfill daily train reproduce status api up down

install:            ## package with tracking, API and dev extras
	$(PY) -m pip install -e ".[tracking,api,dev]"

lint:
	ruff check src tests dags
	ruff format --check src tests dags

fmt:
	ruff format src tests dags
	ruff check --fix src tests dags

test:               ## unit + pipeline tests (synthetic data, no network)
	$(PY) -m pytest

backfill:           ## archive + market history -> warehouse -> features -> DQ gate
	$(PY) -m newspipe backfill

daily:              ## RSS + market increment -> rebuild -> DQ gate -> score
	$(PY) -m newspipe daily

train:              ## walk-forward backtest, MLflow logging, champion promotion
	$(PY) -m newspipe train

reproduce:          ## original notebook protocol for comparison
	$(PY) -m newspipe reproduce

status:
	$(PY) -m newspipe status

api:                ## FastAPI on http://localhost:8000/docs
	uvicorn newspipe.api:app --reload

up:                 ## Airflow :8080, MLflow :5000, API :8000 (override with API_PORT)
	docker compose up -d --build

down:
	docker compose down
