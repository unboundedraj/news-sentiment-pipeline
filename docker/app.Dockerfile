FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    NEWSPIPE_HOME=/opt/newspipe MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /opt/newspipe

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[tracking,api]"
COPY config ./config

EXPOSE 8000
CMD ["uvicorn", "newspipe.api:app", "--host", "0.0.0.0", "--port", "8000"]
