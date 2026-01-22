# Small-Angle Scatter Retention in PET
This repository contains Monte Carlo (MC) simulation code and analysis workflows used to study small-angle Compton scatter retention for PET-like back-to-back 511 keV photons in a water-cylinder phantom. The main outputs are  

1. photon and LOR-level event tables and 
2. figures quantifying the trade-off between recovered coincidences (sensitivity gain) and spatial displacement (FWHM of LOR distance-to-source).

## Repository layout
```
├── data/
│ └── xcomH2O.tsv # NIST XCOM attenuation data for water
├── src/MC/
│ └── water_cylinder.py # MC transport model
├── notebooks/ # Notebooks for MC simulation and figures
├── results/
│ ├── figures/ # The resulting figures
│ ├── MC_data/ # Example dataset
│ └── MC_data_1e7/ # Large dataset (~10^7 annihilations) (not tracked in git)
├── pyproject.toml # Python project + dependencies (uv compatible)
└── uv.lock # For reproducability
```

## Data
Two main table types are produced and stored as Apache Feather files:
- `photons_*.feather`: per-photon records (e.g., exit energy, scatter counts, etc.)
- `lors_*.feather`: per-coincidence/LOR records (e.g., LOR distance-to-source)
Phantom sizes are encoded in filenames via `R5`, `R10`, and `R15` (cm).

### Large dataset (10^7 annihilations)
The directory `results/MC_data_1e7/` contains chunked outputs corresponding to
approximately `10^7` annihilation events. This dataset is **not committed to git**
due to size.

**How to obtain it:** TODO

## Installation

### Using uv (recommended)
This repo is compatible with `uv`:

```bash
uv sync
```
### Using pip
```
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Reproducing the figures
The finalized figures are stored in `results/figures/` (pdf and svg). The notebooks in `notebooks/` reproduce these figures from the Feather tables.

1. Ensure data exist in `results/MC_data_1e7/` for exact figures in publication or your own MC-data produced via the `run_MC.ipynb` notebook (explaineed below).
2. Run the relevant notebooks with the path set to your data in the `MC_DATA_FOLDER` paramter:
    - `notebooks/d_distribution.ipynb`
    - `notebooks/energy_spectra.ipynb`
    - `notebooks/gain_vs_fwhm.ipynb`
    - `notebooks/table_of_simulation_results.ipynb`

## Running the Monte Carlo
The MC main methods is implemented in `src/MC/water_cylinder.py`. A ready to run notebook is provided in `notebooks/run_MC.ipynb` and will produce MC-data to the folder `data/MC_data`.
