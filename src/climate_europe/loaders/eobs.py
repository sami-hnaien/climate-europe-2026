"""E-OBS loader: daily gridded temperature and precipitation (NetCDF).

Two kinds of files are combined along the time axis:
- ``history``: full, quality-controlled release (v33.0e, 1980-2025 here);
- ``2026``: monthly updates, provisional.

Checked on the real files: variables are ``tg, tn, tx, rr`` (int16 with a scale
factor, decoded by xarray), units ``Celsius`` and ``mm``, coordinates ``latitude``
and ``longitude`` (renamed ``lat``/``lon``). History and 2026 files share the same
0.25 deg grid (201 x 464 cells, 25.375N-75.375N, 40.375W-75.375E); this is checked
at every load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from climate_europe.loaders.base import BaseLoader


class EobsLoader(BaseLoader):
    name = "eobs"

    def _find_files(self) -> list[Path]:
        """Return all E-OBS files of the configured variables (history + 2026)."""
        eobs_cfg = self.config["eobs"]
        folders_patterns = [
            (self.config["paths"]["eobs_history_dir"], eobs_cfg["history_pattern"]),
            (self.config["paths"]["eobs_2026_dir"], eobs_cfg["monthly_pattern"]),
        ]
        files: list[Path] = []
        for folder, pattern in folders_patterns:
            for var in eobs_cfg["variables"]:
                files.extend(sorted(Path(folder).glob(pattern.format(var=var))))
        return files

    def files_by_variable(self, files: list[Path] | None = None) -> dict[str, list[Path]]:
        """Group files by variable code (tg, tn, tx, rr), using the prefix of the file name."""
        files = files if files is not None else self._find_files()
        grouped: dict[str, list[Path]] = {v: [] for v in self.config["eobs"]["variables"]}
        for path in files:
            prefix = path.name.split("_", 1)[0].lower()
            if prefix in grouped:
                grouped[prefix].append(path)
        return grouped

    def _read(self, files: list[Path]) -> Any:
        """Open each variable lazily (dask) and merge them into one xarray Dataset."""
        import xarray as xr  # imported here so that file discovery works without xarray

        datasets = []
        for var, var_files in self.files_by_variable(files).items():
            if not var_files:
                raise FileNotFoundError(f"[{self.name}] no file for variable '{var}'")
            self._check_same_grid(var_files)
            self.logger.info("Opening %s (%d files)", var, len(var_files))
            ds = xr.open_mfdataset(
                sorted(var_files), combine="by_coords", chunks={"time": 365}, join="exact", data_vars="minimal"
            )
            datasets.append(ds)
        return xr.merge(datasets, join="exact")  # fails loudly if grids differ

    def _check_same_grid(self, files: list[Path]) -> None:
        """Raise if the latitude/longitude axes differ between files (history vs 2026)."""
        import numpy as np
        import xarray as xr

        reference = None
        for path in files:
            with xr.open_dataset(path) as ds:
                axes = (ds["latitude"].values, ds["longitude"].values)
            if reference is None:
                reference = axes
            elif not all(np.array_equal(a, b) for a, b in zip(reference, axes)):
                raise ValueError(f"[{self.name}] grid of {path.name} differs from {files[0].name}")

    def _validate(self, data: Any) -> None:
        names = self.config["eobs"]["netcdf_variable_names"]
        missing = [names[v] for v in self.config["eobs"]["variables"] if names[v] not in data.data_vars]
        if missing:
            raise ValueError(f"[{self.name}] variables not found in the NetCDF files: {missing}")
        if int(data["time"].to_index().duplicated().sum()) > 0:
            raise ValueError(f"[{self.name}] duplicated dates after combining history and 2026")
        missing_coords = {"latitude", "longitude", "time"} - set(data.coords)
        if missing_coords:
            raise ValueError(f"[{self.name}] coordinates not found: {sorted(missing_coords)}")
        expected = self.config["eobs"].get("expected_units", {})
        for var, unit in expected.items():
            found = data[names[var]].attrs.get("units")
            if found != unit:
                raise ValueError(f"[{self.name}] '{var}' in {found!r}, expected {unit!r}")
        gaps = data["time"].to_index().to_series().diff().dt.days.dropna()
        if (gaps != 1).any():
            raise ValueError(f"[{self.name}] the daily time axis has gaps or is not sorted")

    def _standardize(self, data: Any) -> Any:
        """Rename to project names (tg, tn, tx, rr / lat, lon) and keep the configured years.

        Units are already deg C and mm (checked in ``_validate``), so no conversion is needed.
        """
        names = self.config["eobs"]["netcdf_variable_names"]
        rename = {names[v]: v for v in names if names[v] != v}
        rename.update(self.config["eobs"].get("coordinate_names", {}))
        data = data.rename(rename)
        years = self.config["analysis"]
        data = data.sel(time=slice(f"{years['start_year']}-01-01", f"{years['end_year']}-12-31"))
        start, end = data["time"].to_index()[[0, -1]]
        self.logger.info("E-OBS grid %s, %s to %s", dict(data.sizes), start.date(), end.date())
        return data
