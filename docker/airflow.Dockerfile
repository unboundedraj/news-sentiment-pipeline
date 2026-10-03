FROM apache/airflow:2.10.5-python3.11

# Install the pipeline package into Airflow's environment (DAG tasks import it).
COPY --chown=airflow:root pyproject.toml README.md /opt/newspipe/
COPY --chown=airflow:root src /opt/newspipe/src
RUN pip install --no-cache-dir "/opt/newspipe[tracking]"

COPY --chown=airflow:root config /opt/newspipe/config
# The shared data volume is initialised from this directory, so it must be airflow-owned.
USER root
RUN mkdir -p /opt/newspipe/data && chown -R airflow:root /opt/newspipe/data
USER airflow
ENV NEWSPIPE_HOME=/opt/newspipe \
    NEWSPIPE_CONFIG=/opt/newspipe/config/settings.yaml \
    MLFLOW_DISABLE_AGENT_HINT=1
