"""Spatial aggregation: from the E-OBS grid to one value per NUTS 2 region."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from climate_europe.utils.logging import get_logger


class RegionalAggregator:
    """Average the grid cells falling inside each region, for every day and variable.

    Output (one row per region and date, one column per variable):
        region_id | date | tg | tn | tx | rr

    The wide layout is used instead of a long one (region/date/variable/value):
    it has 4x fewer rows, is what the indicators need, and maps directly to the
    daily Power BI table.

    Weighting rule (documented choice):
        weight(cell, region) = fraction of the cell inside the region x cos(latitude)
    - the fraction comes from ``regionmask.mask_3D_frac_approx`` (sub-sampling of
      each cell), so small regions without any cell centre still get the cells
      they overlap;
    - cos(latitude) corrects for the smaller area of cells further north;
    - cells without data on a given day (sea, missing stations) are ignored that
      day: value = sum(w * x over valid cells) / sum(w over valid cells);
    - regions whose valid overlap is below ``min_cells`` cell equivalents are
      dropped (no reliable E-OBS signal, e.g. remote islands).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger("processing.spatial")
        self.coverage_: pd.DataFrame | None = None

    batch_size = 30  # regions per regionmask call (the 10x10 sub-sampling is memory hungry)

    # ---------------------------------------------------------------- weights
    @staticmethod
    def crop(grid: Any, regions: Any, margin: float = 0.5) -> Any:
        """Keep only the part of the grid that covers the regions (plus a margin in degrees)."""
        west, south, east, north = regions.total_bounds
        return grid.sel(lon=slice(west - margin, east + margin), lat=slice(south - margin, north + margin))

    @staticmethod
    def _fractions(grid: Any, regions: Any) -> np.ndarray:
        """Fraction of each grid cell covered by each region (regionmask, approximate)."""
        import regionmask

        ids = regions["region_id"].tolist()
        masks = regionmask.Regions(list(regions.geometry), abbrevs=ids, names=ids, overlap=False)
        frac = masks.mask_3D_frac_approx(grid["lon"], grid["lat"])
        frac = frac.reindex(region=range(len(ids))).fillna(0.0)  # regions without overlap get zeros
        return frac.values.astype(np.float32)

    def build_weights(self, grid: Any, regions: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return (cell index, weight matrix cells x regions, region ids) for the land cells.

        Only cells that overlap a region and have data at least once in a sample
        year are kept, which keeps the matrix small.
        """
        ids = regions["region_id"].tolist()
        frac = np.concatenate(
            [self._fractions(grid, regions.iloc[i : i + self.batch_size]) for i in range(0, len(ids), self.batch_size)]
        )  # (regions, lat, lon): fraction of each cell inside each region
        cos_lat = np.cos(np.deg2rad(grid["lat"].values))[:, None]
        weights = (frac * cos_lat[None, :, :]).reshape(len(ids), -1).T  # (cells, regions)

        # land cells = cells with data during one sample year (last reference year if available)
        years = np.unique(grid["time"].dt.year.values)
        sample_year = min(self.config["analysis"]["reference_period"][1], years[-1])
        sample_year = sample_year if sample_year in years else years[0]
        has_data = grid["tg"].sel(time=str(sample_year)).notnull().any("time").values.ravel()
        cells = np.flatnonzero((weights.sum(axis=1) > 0) & has_data)
        weights = weights[cells].astype(np.float64)

        min_cells = self.config["aggregation"]["min_cells"]
        effective = frac.reshape(len(ids), -1).T[cells].sum(axis=0)
        self.coverage_ = pd.DataFrame({"region_id": ids, "effective_cells": effective})
        self.coverage_["kept"] = self.coverage_["effective_cells"] >= min_cells
        dropped = self.coverage_.loc[~self.coverage_["kept"], "region_id"].tolist()
        if dropped:
            self.logger.info("Regions dropped (< %.1f valid cells): %s", min_cells, dropped)
        keep = self.coverage_["kept"].values
        return cells, weights[:, keep], [r for r, k in zip(ids, keep) if k]

    # ------------------------------------------------------------ aggregation
    def aggregate(self, grid: Any, regions: Any) -> Any:
        """Return the regional daily table.

        The grid is processed one year at a time (about 140 MB per variable and year),
        so memory stays low whatever the length of the period.
        """
        grid = self.crop(grid, regions)
        cells, weights, ids = self.build_weights(grid, regions)
        variables = self.config["eobs"]["variables"]
        years = np.unique(grid["time"].dt.year.values)
        self.logger.info("Aggregating %d cells to %d regions, %d years", len(cells), len(ids), len(years))

        frames = []
        for year in years:
            block = grid.sel(time=str(year))
            dates = block["time"].to_index()
            columns = {}
            for var in variables:
                values = block[var].values.reshape(len(dates), -1)[:, cells].astype(np.float64)
                valid = ~np.isnan(values)
                numerator = np.where(valid, values, 0.0) @ weights
                denominator = valid.astype(np.float64) @ weights
                with np.errstate(invalid="ignore", divide="ignore"):
                    columns[var] = np.where(denominator > 0, numerator / denominator, np.nan).astype(np.float32).ravel()
            frame = pd.DataFrame(columns)
            frame.insert(0, "date", np.repeat(dates.values, len(ids)))
            frame.insert(0, "region_id", np.tile(ids, len(dates)))
            frames.append(frame)
            self.logger.info("Year %d done", year)

        table = pd.concat(frames, ignore_index=True)
        table["region_id"] = table["region_id"].astype("category")
        return table.sort_values(["region_id", "date"], ignore_index=True)

    def save(self, table: Any, name: str = "regional_daily.parquet") -> None:
        """Save to data/interim so the heavy computation is done only once."""
        path = self.config["paths"]["interim_dir"] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
        if self.coverage_ is not None:
            self.coverage_.to_csv(path.with_name("regional_coverage.csv"), index=False)
        self.logger.info("Saved %s (%d rows)", path, len(table))
