"""Common skeleton of every data loader (template method pattern).

``load()`` always runs the same steps in the same order:

    _find_files -> _read -> _validate -> _standardize

Subclasses only implement what is specific to their source. Validation
failures raise ``ValueError`` and everything is logged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from climate_europe.utils.logging import get_logger


class BaseLoader(ABC):
    """Abstract loader. Subclasses must implement the four steps below."""

    name: str = "base"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(f"loader.{self.name}")

    # ------------------------------------------------------------- public API
    def load(self) -> Any:
        """Run the full pipeline and return clean data."""
        self.logger.info("Start loading %s", self.name)
        files = self._find_files()
        if not files:
            raise FileNotFoundError(f"[{self.name}] no input file found. Check config.yaml paths and file names.")
        self.logger.info("%d file(s) found", len(files))
        data = self._read(files)
        self._validate(data)
        result = self._standardize(data)
        self.logger.info("Loading finished")
        return result

    # ---------------------------------------------------- steps to implement
    @abstractmethod
    def _find_files(self) -> list[Path]:
        """Return the list of input files (sorted)."""

    @abstractmethod
    def _read(self, files: list[Path]) -> Any:
        """Read the files and return raw data (no cleaning yet)."""

    @abstractmethod
    def _validate(self, data: Any) -> None:
        """Check expected variables, dimensions and periods. Raise ValueError if wrong."""

    @abstractmethod
    def _standardize(self, data: Any) -> Any:
        """Rename, convert units and return data in the project standard format."""
