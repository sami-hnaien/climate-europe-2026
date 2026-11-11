"""Climate indicators per region and year, computed from the regional daily table."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from climate_europe.processing import drought
from climate_europe.utils.logging import get_logger


def _runs(flags: np.ndarray) -> np.ndarray:
    """Lengths of the runs of consecutive True values in a 1-D boolean array."""
    padded = np.concatenate([[False], flags, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return edges[1::2] - edges[::2]


class IndicatorCalculator:
    """Compute summer (June-August) indicators for each region and year.

    Input: the regional daily table ``region_id | date | tg | tn | tx | rr``.
    Output: one row per region and year with the columns

        summer_tg, summer_tx, summer_tn          summer means (deg C)
        summer_tg_anomaly, summer_tx_anomaly,    minus the 1991-2020 summer mean
        summer_tn_anomaly
        hot_days_30, hot_days_35                 days with tx > 30 / 35 deg C
        hot_days_p90                             days with tx > local p90 of 1991-2020 summers
        tropical_nights                          nights with tn > 20 deg C
        heatwave_count, heatwave_days,           runs of >= 3 days with tx > local p90
        heatwave_max_length, heatwave_intensity  (intensity = mean excess over p90, deg C)
        summer_rr, summer_rr_anomaly             total (mm) and anomaly in % of 1991-2020
                                                 (NaN where the reference mean is < 10 mm)
        max_consecutive_dry_days                 longest run with rr < 1 mm
        drought_index                            SPEI-3 of August (needs region latitudes)

    Thresholds, reference period and months come from config.yaml. Regional values are
    means over many cells, so absolute thresholds (35 deg C) are reached less often than
    at a station; the relative p90 indicators are comparable across Europe.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger("processing.indicators")
        analysis = config["analysis"]
        self.thresholds = analysis["thresholds"]
        self.reference = tuple(analysis["reference_period"])
        self.months = analysis["summer_months"]

    # --------------------------------------------------------------- helpers
    def _summer(self, daily: pd.DataFrame) -> pd.DataFrame:
        summer = daily[daily["date"].dt.month.isin(self.months)].copy()
        summer["year"] = summer["date"].dt.year
        summer["region_id"] = summer["region_id"].astype(str)
        return summer

    def _in_reference(self, years: pd.Series) -> pd.Series:
        return years.between(*self.reference)

    def _p90(self, summer: pd.DataFrame) -> pd.Series:
        """Local tx threshold: percentile of all reference-period summer days of the region."""
        ref = summer[self._in_reference(summer["year"])]
        return ref.groupby("region_id")["tx"].quantile(self.thresholds["hot_day_percentile"] / 100)

    # ------------------------------------------------------------ indicators
    def summer_anomaly(self, daily: Any, variable: str = "tg") -> Any:
        """Summer mean of ``variable`` minus the 1991-2020 summer mean, per region."""
        summer = self._summer(daily)
        means = summer.groupby(["region_id", "year"])[variable].mean().rename(f"summer_{variable}").reset_index()
        ref = means[self._in_reference(means["year"])].groupby("region_id")[f"summer_{variable}"].mean()
        means[f"summer_{variable}_anomaly"] = means[f"summer_{variable}"] - means["region_id"].map(ref)
        return means

    def hot_days(self, daily: Any, relative: bool = False, threshold: float | None = None) -> Any:
        """Number of days with tx above an absolute threshold, or above the local percentile.

        ``threshold`` overrides the absolute threshold of config.yaml (``hot_day_tx``).
        """
        summer = self._summer(daily)
        if relative:
            above = summer["tx"] > summer["region_id"].map(self._p90(summer))
            name = f"hot_days_p{self.thresholds['hot_day_percentile']}"
        else:
            threshold = self.thresholds["hot_day_tx"] if threshold is None else threshold
            above = summer["tx"] > threshold
            name = f"hot_days_{threshold:g}"
        return above.groupby([summer["region_id"], summer["year"]]).sum().rename(name).reset_index()

    def tropical_nights(self, daily: Any) -> Any:
        """Number of nights with tn above the configured threshold."""
        summer = self._summer(daily)
        above = summer["tn"] > self.thresholds["tropical_night_tn"]
        return above.groupby([summer["region_id"], summer["year"]]).sum().rename("tropical_nights").reset_index()

    def heatwaves(self, daily: Any) -> Any:
        """Count, longest length and intensity of runs of at least N consecutive hot days.

        Hot day = tx above the local p90 (relative threshold, so that a heatwave means
        "unusually hot for this region" both in Finland and in Andalusia). Runs are
        counted inside June-August only.
        """
        summer = self._summer(daily).sort_values(["region_id", "date"])
        summer["excess"] = summer["tx"] - summer["region_id"].map(self._p90(summer))
        min_days = self.thresholds["heatwave_min_days"]
        rows = []
        for (region, year), group in summer.groupby(["region_id", "year"], sort=False):
            hot = (group["excess"] > 0).to_numpy()
            lengths = _runs(hot)
            waves = lengths[lengths >= min_days]
            in_wave = np.zeros_like(hot)
            if len(waves):
                padded = np.concatenate([[False], hot, [False]]).astype(np.int8)
                edges = np.flatnonzero(np.diff(padded))
                for start, end in zip(edges[::2], edges[1::2]):
                    if end - start >= min_days:
                        in_wave[start:end] = True
            excess = group["excess"].to_numpy()[in_wave]
            rows.append(
                {
                    "region_id": region,
                    "year": year,
                    "heatwave_count": len(waves),
                    "heatwave_days": int(waves.sum()),
                    "heatwave_max_length": int(waves.max()) if len(waves) else 0,
                    "heatwave_intensity": float(excess.mean()) if len(excess) else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def dry_spells(self, daily: Any) -> Any:
        """Longest run of consecutive dry days (rr below the configured threshold)."""
        summer = self._summer(daily).sort_values(["region_id", "date"])
        summer["dry"] = summer["rr"] < self.thresholds["dry_day_mm"]
        longest = summer.groupby(["region_id", "year"], sort=False)["dry"].agg(
            lambda s: int(_runs(s.to_numpy()).max(initial=0))
        )
        return longest.rename("max_consecutive_dry_days").reset_index()

    def precipitation(self, daily: Any) -> Any:
        """Summer precipitation total (mm) and its anomaly in % of the 1991-2020 mean.

        In regions where summer is normally almost rainless (Attica, Aegean islands) a
        percentage of a few millimetres is meaningless, so the anomaly is left NaN when
        the reference mean is below ``thresholds.min_reference_rr_mm``.
        """
        summer = self._summer(daily)
        total = summer.groupby(["region_id", "year"])["rr"].sum(min_count=1).rename("summer_rr").reset_index()
        ref = total[self._in_reference(total["year"])].groupby("region_id")["summer_rr"].mean()
        ref = ref.where(ref >= self.thresholds["min_reference_rr_mm"])
        total["summer_rr_anomaly"] = 100 * (total["summer_rr"] / total["region_id"].map(ref) - 1)
        return total

    def monthly(self, daily: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Monthly mean tg and total rr, one column per region (input of the drought index)."""
        frame = daily.assign(region_id=daily["region_id"].astype(str))
        frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()
        grouped = frame.groupby(["month", "region_id"])
        tg = grouped["tg"].mean().unstack()
        rr = grouped["rr"].sum(min_count=1).unstack()
        return tg, rr

    def drought_index(self, daily: Any, latitudes: pd.Series) -> pd.DataFrame:
        """Monthly SPEI-like index (region columns), computed on the full record."""
        drought_cfg = self.config["analysis"]["drought"]
        tg, rr = self.monthly(daily)
        pet = drought.potential_evapotranspiration(tg, latitudes, reference=self.reference)
        balance = drought.water_balance(rr, pet)
        return drought.standardized_index(balance, drought_cfg["scale_months"], self.reference)

    # --------------------------------------------------------------- summary
    def coverage(self, daily: Any) -> pd.DataFrame:
        """Share of summer days with valid tg, tx, tn and rr, per region and year."""
        summer = self._summer(daily)
        valid = summer[["tg", "tn", "tx", "rr"]].notna().all(axis=1)
        expected = summer.groupby(["region_id", "year"]).size()
        return (valid.groupby([summer["region_id"], summer["year"]]).sum() / expected).rename("coverage").reset_index()

    def compute_all(self, daily: Any, latitudes: pd.Series | None = None) -> Any:
        """Return the final region x year indicator table.

        ``latitudes`` (region_id -> latitude of the centroid) is needed for the PET of the
        drought index; without it the ``drought_index`` column is left out. Region-years
        whose summer coverage is below ``aggregation.min_summer_coverage`` are set to NaN.
        """
        keys = ["region_id", "year"]
        parts = [
            self.summer_anomaly(daily, "tg"),
            self.summer_anomaly(daily, "tx"),
            self.summer_anomaly(daily, "tn"),
            self.hot_days(daily, relative=False),
            self.hot_days(daily, threshold=self.thresholds["hot_day_tx_moderate"]),
            self.hot_days(daily, relative=True),
            self.tropical_nights(daily),
            self.heatwaves(daily),
            self.precipitation(daily),
            self.dry_spells(daily),
            self.coverage(daily),
        ]

        table = parts[0]
        for part in parts[1:]:
            table = table.merge(part, on=keys, how="outer")

        if latitudes is not None:
            spei = self.drought_index(daily, latitudes)
            summer_month = self.config["analysis"]["drought"]["summer_month"]
            august = spei[spei.index.month == summer_month]
            august.index = august.index.year
            long = august.stack().rename("drought_index").rename_axis(["year", "region_id"]).reset_index()
            table = table.merge(long, on=keys, how="left")

        low = table["coverage"] < self.config["aggregation"]["min_summer_coverage"]
        if low.any():
            self.logger.info("%d region-years with low summer coverage set to NaN", int(low.sum()))
            value_cols = [c for c in table.columns if c not in keys + ["coverage"]]
            table.loc[low, value_cols] = np.nan

        order = keys + sorted(c for c in table.columns if c not in keys)
        return table[order].sort_values(keys, ignore_index=True)
