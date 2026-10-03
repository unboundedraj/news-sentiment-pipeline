"""Settings loader. `NEWSPIPE_HOME` relocates every relative path (tests use tmp dirs)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "settings.yaml"


@dataclass
class Settings:
    raw: dict[str, Any]
    home: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, key: str) -> Path:
        value = self.raw["paths"][key]
        p = Path(value)
        return p if p.is_absolute() else (self.home / p)

    @cached_property
    def warehouse(self) -> Path:
        p = self.path("warehouse")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def mlflow_uri(self) -> str:
        uri = self.raw["paths"]["mlflow_tracking_uri"]
        if uri.startswith("sqlite:///") and not Path(uri[10:]).is_absolute():
            db = (self.home / uri[10:]).resolve()
            db.parent.mkdir(parents=True, exist_ok=True)
            return "sqlite:///" + db.as_posix()
        return uri


def load_settings(
    config: str | os.PathLike | None = None, home: str | os.PathLike | None = None
) -> Settings:
    path = Path(config or os.environ.get("NEWSPIPE_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    root = Path(home or os.environ.get("NEWSPIPE_HOME", PACKAGE_ROOT))
    return Settings(raw=raw, home=root)
