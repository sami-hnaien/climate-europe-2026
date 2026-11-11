# Summer 2026 in Europe: heat, drought and regional climate profiles

Scientific data visualisation project. Question: **is summer 2026 an exception or the continuation of a long-term trend, and which European NUTS 2 regions share comparable heat/drought profiles?**

Pipeline: E-OBS gridded data -> aggregation by NUTS 2 region -> climate indicators -> EDA and statistical tests -> clustering -> Power BI dashboard.

## Scope

| Item | Choice |
|---|---|
| Area | Extended Europe, NUTS 2 regions (NUTS 2024) |
| Period | 1980-2026 (reference period 1991-2020, summer = June-August) |
| Climate data | E-OBS v33.0e (1980-2025) + monthly updates (2026, provisional), 0.25 deg, ensemble mean |
| Drought | Index computed from E-OBS precipitation and temperature |
| Tools | Python (analysis, clustering), Power BI (final dashboard) |

## Data (not in the repository)

Download and place the files as follows:

```
data/raw/
├── nuts/               NUTS_RG_01M_2024_4326.gpkg   (Eurostat GISCO, NUTS 2024, 1:1M, EPSG:4326, regions)
└── eobs/
    ├── history/        tg|tn|tx|rr _ens_mean_0.25deg_reg_{1980-1994,1995-2010,2011-2025}_v33.0e.nc
    └── 2026/           tg|tn|tx|rr _0.25deg_day_2026_{01..09}_grid_ensmean.nc
```

Sources:
- E-OBS: https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php
- NUTS boundaries: https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics

Acknowledgement required by E-OBS: *"We acknowledge the E-OBS dataset from the Copernicus Climate Change Service, and the data providers in the ECA&D project (https://www.ecad.eu)"*; Cornes et al. (2018), J. Geophys. Res. Atmos., 123, doi:10.1029/2017JD028200.

## Installation

```
conda env create -f environment.yml
conda activate climat-europe
python -m ipykernel install --user --name climat-europe
```

This also installs the project in editable mode (`pip install -e .`) so that notebooks can `import climate_europe`.
On Windows, install Miniforge/Miniconda in a folder **without spaces** (e.g. `C:\Users\Public\miniforge3`); if the
conda solver runs out of memory, use `mamba env create -f environment.yml` (shipped with Miniforge).

## Running

Open `notebooks/00_full_project.ipynb` with the `climat-europe` kernel and run all cells. The grid -> region
aggregation (about 3 minutes, low memory: one year at a time) is cached in `data/interim/regional_daily.parquet`;
set `RECOMPUTE = True` to redo it. Figures go to `reports/figures/`, Power BI tables to `data/processed/`
(see `docs/powerbi_guide.md`).

## Repository structure

```
config.yaml            all settings (paths, thresholds, periods, country list)
src/climate_europe/
  loaders/             BaseLoader + EobsLoader + NutsLoader
  processing/          regional aggregation, indicators, drought index
  analysis/            statistical tests, clustering
  export/              Power BI export (CSV, calendar, GeoJSON/TopoJSON map)
  utils/               logging
  viz.py               map and style helpers
notebooks/             00_full_project.ipynb: introduction, EDA, clustering, final dataviz, export, conclusion
dashboard/             Power BI file and screenshots
reports/, docs/        report, figures, methodological notes
```

Reusable code lives in `src/`; notebooks only call it and tell the story.

## Methodological notes

- 0.25 deg grid: the unit of analysis is the NUTS 2 region, which contains several cells; station density, not grid fineness, limits E-OBS quality.
- Ensemble mean is the best estimate; the ensemble spread (uncertainty) is not used in the main analysis.
- 2026 monthly updates are provisional and may be revised in the next full E-OBS release.

- Region value = mean of the cells weighted by (fraction of the cell inside the region) x cos(latitude); Malta, Ceuta and Melilla have no E-OBS data and are dropped (281 regions kept).
- Drought index: SPEI-like, Thornthwaite PET, 3-month water balance, log-logistic fit (probability-weighted moments) on 1991-2020, clipped to +/-3.
- Trends: Mann-Kendall + Sen's slope on 1980-2025 with Benjamini-Hochberg correction; 2026 is then compared with the trend.

## Status

Complete pipeline: loaders, aggregation, indicators, drought index, statistics, clustering, Power BI export and the full notebook.
