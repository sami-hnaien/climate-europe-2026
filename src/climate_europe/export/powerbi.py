"""Export clean tables for Power BI. Column names must stay stable between runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from climate_europe.utils.logging import get_logger

SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


class PowerBIExporter:
    """Write the CSV tables used by the dashboard into data/processed/.

    Tables written by the notebook:
        regions.csv      region_id, name, country, latitude, longitude, cluster, ...
        indicators.csv   region_id, year, <indicators>
        trends.csv       region_id, variable, slope_decade, p_value, significant_fdr
        daily_2026.csv   region_id, date, tg, tn, tx, rr, tg_clim, tg_anomaly   (for the date slicer)
        pca.csv          region_id, pc1, pc2, cluster
        calendar.csv     date table for Power BI
    The simplified geometry for the map is written separately by ``export_map``
    (GeoJSON, and TopoJSON for the Power BI Shape Map visual).

    Format: UTF-8, comma separator, dot decimal (6 significant digits, so small p-values
    survive), ISO dates (yyyy-mm-dd), so that
    Power BI types the columns correctly whatever the Windows locale.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.folder = Path(config["paths"]["processed_dir"])
        self.logger = get_logger("export.powerbi")

    def export_all(self, tables: dict[str, Any]) -> None:
        """Write every table of ``tables`` as CSV (UTF-8) in the processed folder."""
        self.folder.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            path = self.folder / (name if name.endswith(".csv") else f"{name}.csv")
            frame = table.reset_index() if not isinstance(table.index, pd.RangeIndex) else table
            frame.to_csv(path, index=False, encoding="utf-8", date_format="%Y-%m-%d", float_format="%.6g")
            self.logger.info("%s: %d rows x %d columns", path.name, len(frame), frame.shape[1])

    @staticmethod
    def build_calendar(start: str, end: str) -> Any:
        """Date table (date, year, month, season) to relate to the daily data in Power BI."""
        dates = pd.date_range(start, end, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "year": dates.year,
                "month": dates.month,
                "month_name": dates.strftime("%b"),
                "day": dates.day,
                "day_of_year": dates.dayofyear,
                "week": dates.isocalendar().week.to_numpy(),
                "season": dates.month.map(SEASONS),
                "is_summer": dates.month.isin([6, 7, 8]),
            }
        )

    def export_map(self, regions: Any, name: str = "regions_map") -> None:
        """Write simplified region boundaries as GeoJSON and TopoJSON (EPSG:4326).

        Simplification (Douglas-Peucker, ``export.map_simplify_deg``) keeps the file small;
        it is only for display, never for computation.
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        tolerance = self.config["export"]["map_simplify_deg"]
        shapes = regions[["region_id", "name", "country", "geometry"]].copy()
        shapes["geometry"] = shapes.geometry.simplify(tolerance, preserve_topology=True)
        shapes.to_file(self.folder / f"{name}.geojson", driver="GeoJSON")
        with open(self.folder / f"{name}.topojson", "w", encoding="utf-8") as handle:
            json.dump(self._to_topojson(shapes), handle, separators=(",", ":"))
        self.logger.info("%s.geojson and %s.topojson written (%d regions)", name, name, len(shapes))

    @staticmethod
    def _to_topojson(shapes: Any) -> dict[str, Any]:
        """Minimal TopoJSON: one arc per ring, absolute coordinates (no shared arcs)."""
        arcs: list[list[list[float]]] = []

        def ring_arc(ring: Any) -> list[int]:
            arcs.append([[round(x, 4), round(y, 4)] for x, y in ring.coords])
            return [len(arcs) - 1]

        def polygon_arcs(polygon: Any) -> list[list[int]]:
            return [ring_arc(polygon.exterior)] + [ring_arc(r) for r in polygon.interiors]

        geometries = []
        for row in shapes.itertuples(index=False):
            parts = list(row.geometry.geoms) if row.geometry.geom_type == "MultiPolygon" else [row.geometry]
            geometries.append(
                {
                    "type": "MultiPolygon",
                    "arcs": [polygon_arcs(p) for p in parts],
                    "properties": {"region_id": row.region_id, "name": row.name, "country": row.country},
                }
            )
        return {
            "type": "Topology",
            "objects": {"regions": {"type": "GeometryCollection", "geometries": geometries}},
            "arcs": arcs,
        }
