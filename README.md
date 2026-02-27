# Quantifying Sensitivity Gains from Selective Inclusion of Small-Angle Scattered Coincidences in PET

This repository contains the Monte Carlo simulation framework and analysis code used to investigate the trade-off between sensitivity and spatial resolution when selectively retaining small-angle Compton-scattered coincidences in PET.

The work quantifies how lowering the energy threshold increases the number of accepted lines of response (LORs) while introducing controlled spatial broadening.

### Repository structure
```
data/
    xcomH2O.tsv # NIST XCOM attenuation data for water
notebooks/
    01_run_simulation.ipynb
    02_fig_energy_exit_spectra.ipynb
    03_fig_lor_displacement_distributions.ipynb
    04_fig_gain_vs_fwhm.ipynb
    05_table_simulation_summary.ipynb
src/sas_pet/
    simulation, transport, geometry, physics models
    analysis modules (fwhm_gain, fates, etc.)
results/
    figures/ # All manuscript figures
    MC_data/ # Example datasets
    MC_data_1e7/ # Large dataset (not tracked in git)
```
### Scope

The simulation models:
- Back-to-back 511 keV annihilation photons
- Compton scattering in a cylindrical water phantom
- Ideal cylindrical detector ring
- Energy-based coincidence selection
- Spatial displacement distributions (perpendicular and axial)
- FWHM vs sensitivity gain trade-off curves

### Installation
__Using pip__
```
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Minimal example
```python
from sas_pet.simulate import SimulationConfig, run_annihilations

cfg = SimulationConfig(
    cylinder_radius_cm=10.0,
    cylinder_half_len_cm=15.0,
    rng_seed=42,
)

photons, lors = run_annihilations(
    10000,
    cfg=cfg,
    xcom_path="data/xcomH2O.tsv",
)
```

### Reproducing figures
Each manuscript figure corresponds to one notebook:
| Notebook                                    | Figure                      | Figure number |
| ------------------------------------------- | --------------------------- | ------------- |
| 02_fig_energy_exit_spectra.ipynb            | Exit energy spectra         | 1 |
| 03_fig_lor_displacement_distributions.ipynb | LOR displacement histograms | 2 |
| 04_fig_gain_vs_fwhm.ipynb                   | FWHM vs sensitivity gain    | 3 |
| 05_table_simulation_summary.ipynb           | Simulation summary tables   | 4 |

### Large dataset (10^7 annihilations per radius)
The full Monte Carlo outputs used to generate the manuscript figures are archived separately:
- Dataset (Monte Carlo outputs): https://doi.org/10.5281/zenodo.18788484  
- Software (source code): https://doi.org/10.5281/zenodo.18787917  
Each radius (R = 5, 10, 15 cm) contains 10^7 annihilation events (2 × 10^7 photons). To reproduce the manuscript figures exactly, download the dataset and extract it to:
```
results/MC_data_1e7/
    R5/
    R10/
    R15/
```
The archived dataset was generated using sas_pet v1.0.0 with deterministic pseudo-random number generation (seed = 42).

### Reproducibility
- Python ≥ 3.12
- Dependencies specified in `pyproject.toml`
- Locked dependency versions in `uv.lock`
- Deterministic random seed (seed = 42)
- All figures generated programmatically from archived data

### Citation
If you use this software, please cite:

Lundhammar, P. (2026).
*Quantifying Sensitivity Gains from Selective Inclusion of Small-Angle Scattered Coincidences in PET*.
Zenodo. https://doi.org/10.5281/zenodo.18787917

The exact version used in the manuscript corresponds to:
v1.0.0 — https://doi.org/10.5281/zenodo.18787918

### Licence
This project is licensed under the MIT License (see LICENCE).

