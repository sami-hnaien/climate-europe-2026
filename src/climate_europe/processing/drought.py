"""Drought index computed from E-OBS (SPEI-like), because SPEIbase ends in 2024.

Idea: climatic water balance D = precipitation - potential evapotranspiration (PET),
accumulated over 3 months, then standardised over the reference period.

Choices (to document in the report):
- PET: Thornthwaite (1948), which only needs monthly mean temperature and latitude.
  It is the PET of the original SPEI paper but underestimates PET in hot, dry,
  windy conditions (no radiation, wind or humidity), so the index is "SPEI-like".
  Above 26.5 deg C the Willmott et al. (1985) high-temperature formula is used.
- Standardisation: three-parameter log-logistic distribution fitted per calendar
  month with probability-weighted moments (Vicente-Serrano et al., 2010), then
  converted to a standard normal value. A plain z-score is used as fallback if
  the fit fails. Values are clipped to +/-3: with 30 reference years, anything
  beyond (probability < 0.13 %) is outside what the fit can resolve.
- Validation against SPEIbase (1980-2024) is possible if it is downloaded; it
  was not part of this project.

All functions work on pandas objects with a monthly DatetimeIndex and one column
per region (a Series is treated as a single region).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import special, stats


def _as_frame(data: Any) -> tuple[pd.DataFrame, bool]:
    if isinstance(data, pd.Series):
        return data.to_frame(), True
    return data, False


def mean_day_length(month_starts: pd.DatetimeIndex, latitude: np.ndarray) -> np.ndarray:
    """Mean astronomical day length (hours) of each month (rows) at each latitude (columns)."""
    mid_month = (month_starts + pd.Timedelta(days=14)).dayofyear.values
    declination = 0.409 * np.sin(2 * np.pi * mid_month / 365 - 1.39)
    phi = np.deg2rad(np.asarray(latitude, dtype=float))
    cos_ws = np.clip(-np.tan(phi)[None, :] * np.tan(declination)[:, None], -1.0, 1.0)
    return 24.0 / np.pi * np.arccos(cos_ws)


def potential_evapotranspiration(
    tg_monthly: Any, latitude: Any, reference: tuple[int, int] = (1991, 2020)
) -> Any:
    """Estimate monthly PET (mm/month) from mean temperature and latitude (Thornthwaite).

    The heat index I uses the monthly climatology of the reference period, so that
    PET in a given year does not depend on that year's own heat index.
    """
    frame, was_series = _as_frame(tg_monthly)
    lat = np.atleast_1d(latitude if not isinstance(latitude, pd.Series) else latitude.reindex(frame.columns).values)
    lat = np.broadcast_to(lat, (frame.shape[1],))
    temp = frame.clip(lower=0.0)

    ref = temp[(temp.index.year >= reference[0]) & (temp.index.year <= reference[1])]
    if ref.empty:
        ref = temp
    climatology = ref.groupby(ref.index.month).mean()
    heat_index = ((climatology / 5.0) ** 1.514).sum()  # one value per region
    a = 6.75e-7 * heat_index**3 - 7.71e-5 * heat_index**2 + 1.792e-2 * heat_index + 0.49239

    with np.errstate(divide="ignore", invalid="ignore"):
        unadjusted = 16.0 * (10.0 * temp / heat_index) ** a
    hot = temp > 26.5
    unadjusted = unadjusted.where(~hot, -415.85 + 32.24 * temp - 0.43 * temp**2)
    unadjusted = unadjusted.where(temp > 0, 0.0)

    day_length = mean_day_length(frame.index, lat)
    days = frame.index.days_in_month.values[:, None]
    pet = unadjusted * (day_length / 12.0) * (days / 30.0)
    pet = pet.where(frame.notna())
    return pet.iloc[:, 0] if was_series else pet


def water_balance(rr_monthly: Any, pet_monthly: Any) -> Any:
    """Monthly climatic water balance: precipitation minus PET (mm/month)."""
    return rr_monthly - pet_monthly


def _loglogistic_pwm(sample: np.ndarray) -> tuple[float, float, float] | None:
    """Fit a 3-parameter log-logistic (alpha, beta, gamma) with probability-weighted moments."""
    x = np.sort(sample[np.isfinite(sample)])
    n = len(x)
    if n < 10:
        return None
    f = (np.arange(1, n + 1) - 0.35) / n
    w0, w1, w2 = (np.mean((1 - f) ** s * x) for s in range(3))
    beta = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
    if not np.isfinite(beta) or beta <= 1.0:
        return None
    g = special.gamma(1 + 1 / beta) * special.gamma(1 - 1 / beta)
    alpha = (w0 - 2 * w1) * beta / g
    gamma = w0 - alpha * g
    if not np.isfinite(alpha) or alpha <= 0:
        return None
    return alpha, beta, gamma


def _standardize_values(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    params = _loglogistic_pwm(reference)
    if params is None:
        mean, std = np.nanmean(reference), np.nanstd(reference, ddof=1)
        return (values - mean) / std
    alpha, beta, gamma = params
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = 1.0 / (1.0 + (alpha / (values - gamma)) ** beta)
    prob = np.where(values <= gamma, 0.0, prob)
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return np.where(np.isfinite(values), stats.norm.ppf(prob), np.nan)


INDEX_LIMIT = 3.0


def standardized_index(balance: Any, scale_months: int = 3, reference: tuple[int, int] = (1991, 2020)) -> Any:
    """Accumulate over ``scale_months`` and standardise against the reference period.

    Each calendar month is standardised separately (an August SPEI-3 is compared
    with the June-August balances of 1991-2020 only).
    """
    frame, was_series = _as_frame(balance)
    accumulated = frame.rolling(scale_months, min_periods=scale_months).sum()
    in_ref = (accumulated.index.year >= reference[0]) & (accumulated.index.year <= reference[1])
    values = accumulated.to_numpy(dtype=float)
    result = np.full(values.shape, np.nan)
    for month in range(1, 13):
        rows = accumulated.index.month == month
        for j in range(values.shape[1]):
            result[rows, j] = _standardize_values(values[rows, j], values[rows & in_ref, j])
    result = pd.DataFrame(np.clip(result, -INDEX_LIMIT, INDEX_LIMIT), index=accumulated.index, columns=accumulated.columns)
    return result.iloc[:, 0] if was_series else result
