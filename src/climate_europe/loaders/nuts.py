"""NUTS loader: administrative regions from the Eurostat GISCO GeoPackage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from climate_europe.loaders.base import BaseLoader


class NutsLoader(BaseLoader):
    name = "nuts"

    def _find_files(self) -> list[Path]:
        path = self.config["paths"]["nuts_dir"] / self.config["nuts"]["file"]
        return [path] if path.is_file() else []

    def _read(self, files: list[Path]) -> Any:
        import geopandas as gpd

        return gpd.read_file(files[0])

    def _validate(self, data: Any) -> None:
        cfg = self.config["nuts"]
        needed = {cfg["level_column"], cfg["id_column"], cfg["country_column"], cfg["name_column"], "geometry"}
        missing = needed - set(data.columns)
        if missing:
            raise ValueError(f"[{self.name}] missing columns in the GeoPackage: {sorted(missing)}")
        if data.crs is None or data.crs.to_epsg() != 4326:
            raise ValueError(f"[{self.name}] expected EPSG:4326, got {data.crs}")

    def _standardize(self, data: Any) -> Any:
        """Keep NUTS 2 regions of the configured countries, without outermost regions."""
        cfg = self.config["nuts"]
        regions = data[data[cfg["level_column"]] == cfg["level"]]
        regions = regions[regions[cfg["country_column"]].isin(cfg["countries"])]
        regions = regions[~regions[cfg["id_column"]].isin(cfg["exclude_regions"])]
        regions = regions.rename(
            columns={cfg["id_column"]: "region_id", cfg["country_column"]: "country", cfg["name_column"]: "name"}
        )
        regions = regions[["region_id", "name", "country", "geometry"]]
        self.logger.info("%d NUTS %d regions kept", len(regions), cfg["level"])
        return regions.reset_index(drop=True)
