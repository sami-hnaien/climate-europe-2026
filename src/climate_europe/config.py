"""Load config.yaml and resolve paths relative to the project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = "config.yaml"


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: current directory) until config.yaml is found."""
    start = (start or Path.cwd()).resolve()
    for folder in [start, *start.parents]:
        if (folder / CONFIG_NAME).is_file():
            return folder
    raise FileNotFoundError(
        f"{CONFIG_NAME} not found in {start} or any parent folder. "
        "Run from inside the project or pass an explicit path to load_config()."
    )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the configuration as a dict.

    ``cfg["root"]`` is the project root and every entry of ``cfg["paths"]``
    is converted to an absolute ``Path``.
    """
    if path is None:
        root = find_project_root()
        path = root / CONFIG_NAME
    path = Path(path).resolve()
    root = path.parent

    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    cfg["root"] = root
    cfg["paths"] = {name: root / rel for name, rel in cfg["paths"].items()}
    return cfg
