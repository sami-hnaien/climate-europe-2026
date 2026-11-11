"""Statistical tests used in the EDA (one function per research question)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps


def trend_mann_kendall(series: Any) -> dict[str, float]:
    """Mann-Kendall trend test and Sen's slope for one yearly series (Q1, Q4).

    ``series`` is indexed by year. Missing years are dropped. The slope is per year;
    Sen's slope and intercept are computed against the real years (scipy ``theilslopes``)
    so that a missing year does not shift the line. The original test assumes no serial
    correlation; this is checked on the detrended series in the notebook.
    """
    import pymannkendall as mk

    clean = pd.Series(series).dropna()
    if len(clean) < 10:
        return {"slope": np.nan, "intercept": np.nan, "tau": np.nan, "p_value": np.nan, "n": len(clean)}
    result = mk.original_test(clean.to_numpy())
    sen = sps.theilslopes(clean.to_numpy(), clean.index.to_numpy(dtype=float))
    return {
        "slope": sen.slope,
        "intercept": sen.intercept,
        "tau": result.Tau,
        "p_value": result.p,
        "n": len(clean),
    }


def trend_table(indicators: Any, variable: str, years: tuple[int, int] | None = None, alpha: float = 0.05) -> Any:
    """Apply the trend test to every region; return slope, p-value and significance.

    ``slope_decade`` is Sen's slope per 10 years. ``significant_fdr`` applies the
    Benjamini-Hochberg correction over all regions (several hundred tests at once).
    """
    from statsmodels.stats.multitest import multipletests

    data = indicators
    if years is not None:
        data = data[data["year"].between(*years)]
    rows = []
    for region, group in data.groupby("region_id", observed=True):
        result = trend_mann_kendall(group.set_index("year")[variable])
        rows.append({"region_id": region, **result})
    table = pd.DataFrame(rows)
    table["slope_decade"] = 10 * table["slope"]
    table["significant"] = table["p_value"] < alpha
    tested = table["p_value"].notna()
    table["significant_fdr"] = False
    table.loc[tested, "significant_fdr"] = multipletests(table.loc[tested, "p_value"], alpha=alpha, method="fdr_bh")[0]
    table.attrs["variable"] = variable
    return table


def anomaly_zscore(indicators: Any, variable: str, year: int = 2026, reference: tuple[int, int] = (1991, 2020)) -> Any:
    """Z-score of ``year`` against the reference period, per region (Q2).

    Also returns the rank of ``year`` in the full record (1 = highest) and whether it
    is a record.
    """
    rows = []
    for region, group in indicators.groupby("region_id", observed=True):
        series = group.set_index("year")[variable].dropna()
        if year not in series.index:
            continue
        ref = series.loc[reference[0] : reference[1]]
        value = series.loc[year]
        std = ref.std(ddof=1)
        rows.append(
            {
                "region_id": region,
                "value": value,
                "ref_mean": ref.mean(),
                "ref_std": std,
                "zscore": (value - ref.mean()) / std if std > 0 else np.nan,
                "rank": int((series > value).sum() + 1),
                "n_years": len(series),
            }
        )
    table = pd.DataFrame(rows)
    table["is_record"] = table["rank"] == 1
    table.attrs.update(variable=variable, year=year)
    return table


def departure_from_trend(indicators: Any, trends: Any, variable: str, year: int = 2026) -> Any:
    """Compare ``year`` with the value expected from the Sen trend line (Q3).

    ``residual_z`` is the departure divided by the standard deviation of the detrended
    series over the trend period: |residual_z| < 2 means that the year is compatible
    with "trend + usual year-to-year variability".
    """
    merged = indicators.merge(trends[["region_id", "slope", "intercept"]], on="region_id")
    merged["expected"] = merged["intercept"] + merged["slope"] * merged["year"]
    merged["residual"] = merged[variable] - merged["expected"]
    others = merged[merged["year"] != year]
    spread = others.groupby("region_id", observed=True)["residual"].std(ddof=2).rename("residual_std")
    target = merged[merged["year"] == year].set_index("region_id")
    out = target[["expected", variable, "residual"]].rename(columns={variable: "value"}).join(spread)
    out["residual_z"] = out["residual"] / out["residual_std"]
    return out.reset_index()


def north_south_comparison(trends: Any, regions: Any, bands: list[float] | None = None) -> Any:
    """Compare warming trends by latitude band (Q4). Beware of spatial autocorrelation.

    ``regions`` needs ``region_id`` and ``latitude`` (centroid). Returns one row per
    band (count, mean and median slope per decade, share of significant trends).
    ``attrs`` holds a Kruskal-Wallis test across bands and the Spearman correlation
    between slope and latitude. Neighbouring regions are not independent, so the
    p-values are optimistic (the effective sample size is smaller than n).
    """
    bands = bands or [30, 45, 55, 72]
    data = trends.merge(regions[["region_id", "latitude"]], on="region_id").dropna(subset=["slope_decade"])
    labels = [f"{lo}-{hi}N" for lo, hi in zip(bands[:-1], bands[1:])]
    data["band"] = pd.cut(data["latitude"], bands, labels=labels)
    summary = data.groupby("band", observed=False).agg(
        n_regions=("slope_decade", "size"),
        mean_slope_decade=("slope_decade", "mean"),
        median_slope_decade=("slope_decade", "median"),
        share_significant=("significant_fdr", "mean"),
    )
    groups = [g["slope_decade"].to_numpy() for _, g in data.groupby("band", observed=True)]
    kruskal = sps.kruskal(*groups) if len(groups) > 1 else None
    spearman = sps.spearmanr(data["latitude"], data["slope_decade"])
    summary.attrs["kruskal"] = {"H": kruskal.statistic, "p_value": kruskal.pvalue} if kruskal else None
    summary.attrs["spearman"] = {"rho": spearman.statistic, "p_value": spearman.pvalue, "n": len(data)}
    return summary.reset_index()
