"""Walk-forward backtest + MLflow tracking + model registration.

For each test year Y >= first_test_year, every candidate is trained on all days before
Y and evaluated on Y. This mimics how the model would actually have been used: no
future data, no shuffled splits. Results are pooled across years, with a 95% Wilson
interval and a binomial test against the majority-class baseline, because "52%" only
means something next to its uncertainty.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from newspipe.config import Settings
from newspipe.features import SERVING_COLUMNS, training_frame
from newspipe.models import add_market_context, candidates
from newspipe.warehouse import connect

log = logging.getLogger("newspipe.train")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def walk_forward(
    frame: pd.DataFrame, first_test_year: int, seed: int, only: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-fold metrics, out-of-sample predictions)."""
    frame = frame.assign(year=pd.to_datetime(frame.news_date).dt.year)
    years = sorted(y for y in frame.year.unique() if y >= first_test_year)
    folds, preds = [], []
    for cand in candidates(seed):
        if only and cand.name not in only:
            continue
        for year in years:
            train, test = frame[frame.year < year], frame[frame.year == year]
            if train.empty or test.empty:
                continue
            est = clone(cand.estimator).fit(train, train.target)
            proba = est.predict_proba(test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            folds.append(
                {
                    "model": cand.name,
                    "test_year": year,
                    "n_train": len(train),
                    "n_test": len(test),
                    "accuracy": accuracy_score(test.target, pred),
                    "balanced_accuracy": balanced_accuracy_score(test.target, pred),
                    "auc": roc_auc_score(test.target, proba)
                    if test.target.nunique() > 1
                    else np.nan,
                }
            )
            preds.append(
                pd.DataFrame(
                    {
                        "model": cand.name,
                        "news_date": test.news_date,
                        "y": test.target.to_numpy(),
                        "proba": proba,
                        "pred": pred,
                    }
                )
            )
        log.info("walk-forward done: %s", cand.name)
    return pd.DataFrame(folds), pd.concat(preds, ignore_index=True)


def summarise(preds: pd.DataFrame) -> pd.DataFrame:
    majority_rate = None
    rows = []
    for model, g in preds.groupby("model", sort=False):
        n, k = len(g), int((g.y == g.pred).sum())
        lo, hi = wilson_interval(k, n)
        rows.append(
            {
                "model": model,
                "n_days": n,
                "accuracy": k / n,
                "ci95_low": lo,
                "ci95_high": hi,
                "balanced_accuracy": balanced_accuracy_score(g.y, g.pred),
                "auc": roc_auc_score(g.y, g.proba) if g.proba.nunique() > 1 else 0.5,
            }
        )
        if model == "majority_class":
            majority_rate = k / n
    out = pd.DataFrame(rows)
    if majority_rate is not None:
        out["p_value_vs_majority"] = [
            binomtest(
                int(round(r.accuracy * r.n_days)), r.n_days, majority_rate, alternative="greater"
            ).pvalue
            for r in out.itertuples()
        ]
    return out.sort_values("balanced_accuracy", ascending=False, ignore_index=True)


def run_backtest(
    s: Settings,
    run_id: str,
    target: str | None = None,
    only: list[str] | None = None,
    track: bool = True,
) -> dict[str, Any]:
    cfg = s["training"]
    target = target or cfg["target"]
    with connect(s.warehouse) as con:
        frame = add_market_context(training_frame(con, target), target)
    folds, preds = walk_forward(frame, cfg["first_test_year"], cfg["random_seed"], only)
    summary = summarise(preds)
    log.info("\n%s", summary.to_string(index=False))

    servable = {c.name: c for c in candidates(cfg["random_seed"]) if c.servable}
    best = summary[summary.model.isin(servable)].iloc[0]
    result: dict[str, Any] = {
        "run_id": run_id,
        "target": target,
        "summary": summary,
        "folds": folds,
        "best_model": best.model,
        "rows": len(frame),
    }
    if track:
        result["mlflow"] = log_to_mlflow(
            s, run_id, target, frame, folds, summary, servable[best.model]
        )
    write_report(s, target, summary, folds)
    return result


def log_to_mlflow(
    s: Settings,
    run_id: str,
    target: str,
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    best,
) -> dict[str, Any]:
    import mlflow
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    cfg = s["training"]
    mlflow.set_tracking_uri(s.mlflow_uri)
    artifacts = s.path("mlflow_artifacts").resolve()
    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        mlflow.create_experiment(cfg["experiment"], artifact_location=artifacts.as_uri())
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name=f"walkforward-{target}-{run_id}") as parent:
        mlflow.set_tags(
            {
                "pipeline_run_id": run_id,
                "target": target,
                "feature_version": s["features"]["version"],
            }
        )
        mlflow.log_params(
            {
                "target": target,
                "first_test_year": cfg["first_test_year"],
                "rows": len(frame),
                "train_end": str(frame.news_date.max()),
            }
        )
        for r in summary.itertuples():
            with mlflow.start_run(run_name=r.model, nested=True):
                mlflow.log_metrics(
                    {
                        "accuracy": r.accuracy,
                        "balanced_accuracy": r.balanced_accuracy,
                        "auc": r.auc,
                        "ci95_low": r.ci95_low,
                        "ci95_high": r.ci95_high,
                    }
                )
                for f in folds[folds.model == r.model].itertuples():
                    mlflow.log_metric("fold_accuracy", f.accuracy, step=int(f.test_year))
        with tempfile.TemporaryDirectory() as tmp:
            summary.to_csv(Path(tmp) / "summary.csv", index=False)
            folds.to_csv(Path(tmp) / "folds.csv", index=False)
            mlflow.log_artifacts(tmp, artifact_path="backtest")

        # Final model: the best servable candidate refit on all available history.
        X = frame[SERVING_COLUMNS].astype({"n_headlines": "float64"})
        final = clone(best.estimator).fit(X, frame.target)
        sample = X.head(5)
        info = mlflow.sklearn.log_model(
            final,
            name="model",
            signature=infer_signature(sample, final.predict_proba(sample)[:, 1]),
            input_example=sample,
            registered_model_name=cfg["registered_model"],
            serialization_format="cloudpickle",
        )
        best_row = summary[summary.model == best.name].iloc[0]
        mlflow.log_metric("selected_balanced_accuracy", best_row.balanced_accuracy)

    client = MlflowClient()
    version = info.registered_model_version
    client.set_model_version_tag(cfg["registered_model"], version, "algorithm", best.name)
    client.set_model_version_tag(cfg["registered_model"], version, "target", target)
    promoted = _promote(
        client, cfg["registered_model"], version, best_row.balanced_accuracy, target
    )
    return {"run_id": parent.info.run_id, "model_version": version, "promoted": promoted}


def _promote(client, name: str, version: str, score: float, target: str) -> bool:
    """Champion/challenger: move the `champion` alias only if the challenger is better."""
    try:
        champ = client.get_model_version_by_alias(name, "champion")
        champ_run = client.get_run(champ.run_id)
        champ_score = champ_run.data.metrics.get("selected_balanced_accuracy", 0.0)
        same_target = champ.tags.get("target") == target
    except Exception:
        champ_score, same_target = -1.0, False
    if not same_target or score > champ_score:
        client.set_registered_model_alias(name, "champion", version)
        log.info("model v%s promoted to champion (%.4f > %.4f)", version, score, champ_score)
        return True
    log.info("model v%s kept as challenger (%.4f <= %.4f)", version, score, champ_score)
    return False


def write_report(s: Settings, target: str, summary: pd.DataFrame, folds: pd.DataFrame) -> Path:
    reports = s.home / "reports"
    reports.mkdir(exist_ok=True)
    path = reports / f"backtest_{target}.md"
    by_year = folds.pivot_table(index="test_year", columns="model", values="accuracy")
    order = [m for m in summary.model if m in by_year.columns]
    lines = [
        f"# Walk-forward backtest - target `{target}`",
        "",
        "_Auto-generated by `newspipe train`. Train on all years before Y, test on Y._",
        "",
        "| Model | Days | Accuracy | 95% CI | Balanced acc. | ROC-AUC | p (vs majority) |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        p = getattr(r, "p_value_vs_majority", float("nan"))
        lines.append(
            f"| {r.model} | {r.n_days:,} | {r.accuracy:.3f} | {r.ci95_low:.3f}-{r.ci95_high:.3f} "
            f"| {r.balanced_accuracy:.3f} | {r.auc:.3f} | {p:.3f} |"
        )
    lines += [
        "",
        "## Accuracy by test year",
        "",
        "| Year | " + " | ".join(order) + " |",
        "|---|" + "---:|" * len(order),
    ]
    for year, row in by_year[order].iterrows():
        lines.append(f"| {year} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (reports / f"backtest_{target}.json").write_text(
        json.dumps(summary.to_dict("records"), indent=2, default=float), encoding="utf-8"
    )
    return path
