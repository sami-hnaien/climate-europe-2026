"""Plotting helpers shared by the notebook (maps and style)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

MAP_CRS = 3035  # ETRS89 / LAEA Europe: equal-area projection, standard for European maps


def set_style() -> None:
    """Consistent, sober style for every figure of the project."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "font.size": 9,
        }
    )


def map_layer(regions: Any, simplify_m: float = 2000) -> Any:
    """Project regions to LAEA Europe and simplify them (display only)."""
    shapes = regions.to_crs(MAP_CRS)
    shapes["geometry"] = shapes.geometry.simplify(simplify_m)
    return shapes


def plot_map(
    shapes: Any,
    column: str,
    ax: Any = None,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    title: str = "",
    label: str = "",
    categorical: bool = False,
    hatch_mask: Any = None,
    legend: bool = True,
    legend_kwds: dict[str, Any] | None = None,
) -> Any:
    """Choropleth of ``column``. ``center`` makes a symmetric diverging scale around it.

    ``hatch_mask`` (boolean Series aligned on ``shapes``) hatches the selected regions,
    e.g. non-significant trends. Missing values are drawn in light grey.
    ``legend_kwds`` overrides the legend options (e.g. its position).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    if center is not None and vmin is None and vmax is None:
        span = np.nanpercentile(np.abs(shapes[column] - center), 98)
        vmin, vmax = center - span, center + span
    kwargs: dict[str, Any] = {"legend": legend, "missing_kwds": {"color": "#dddddd", "label": "no data"}}
    if categorical:
        kwargs.update(categorical=True, legend_kwds={"loc": "upper left", "title": label, "fontsize": 8})
    else:
        kwargs.update(vmin=vmin, vmax=vmax, legend_kwds={"label": label, "shrink": 0.6})
    kwargs["legend_kwds"].update(legend_kwds or {})
    shapes.plot(column=column, cmap=cmap, ax=ax, linewidth=0.15, edgecolor="white", **kwargs)
    if hatch_mask is not None and hatch_mask.any():
        shapes[hatch_mask.values].plot(ax=ax, facecolor="none", edgecolor="#555555", hatch="////", linewidth=0)
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def save(fig: Any, folder: Path, name: str) -> None:
    """Save a figure as PNG in the report figure folder."""
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png")
