# XAS Analysis Script Documentation

**Script**: `analysis_publication.py`  
**Version**: 1.0  
**Author**: Charles Le Losq  
**Date**: November 2025

## Overview

This script performs complete X-ray Absorption Spectroscopy (XAS) analysis for Mayotte volcanic samples, including:
- Fe K-edge XANES analysis (iron redox state)
- S K-edge XANES analysis (sulfur speciation)
- Publication-quality figure generation
- Beam damage assessment

## Objectives

- Calibrate Fe pre-edge XANES centroid - Fe redox relationship using reference glass materials
- Determine Fe³⁺/Feᵀᴼᵀ ratios
- Quantify S⁶⁺/Sᵀᴼᵀ speciation from S K-edge XANES spectra using multiple methods:
  - Lerner et al. 2021 (based on peak areas)
  - Jugo 2010 (intensity-based)
  - Jugo 2010 revised by us (LL2023, not convincing?)

## Input Data

```
Mayotte_publication/
├── tables/liste.xlsx                  # Sample metadata (Excel)
│   ├── Fe_glasses_standards    # Fe reference materials
│   ├── Fe_glasses             # Fe sample analyses
│   ├── Fe_glasses_standards_forfigure  # For publication figures
│   ├── Glasses_forfigure      # Sample spectra for figures
│   ├── S_glasses              # S sample analyses
│   ├── Fe_damage              # Beam damage tests
│   └── Centroid_Data          # Literature calibrations
│
└── xas/
    ├── iron/*.dat             # Fe K-edge spectra (ASCII)
    └── sulfur/*.dat           # S K-edge spectra (ASCII)
```

## Output Files

### Figures

**Individual Spectra** (`figures/Iron/`, `figures/Sulfur/`):
- One PDF per sample showing:
  - Raw and normalized spectra
  - Pre-edge and post-edge fits
  - Peak deconvolution

**Publication Figures** (`figures/`):
- `calibration.pdf` - Fe centroid vs. Fe³⁺/Feᵀᴼᵀ calibration
- `Spectra_refs.pdf` - Reference spectra (standards)
- `Spectra_samples.pdf` - Fe sample spectra overlay
- `Spectra_S_samples.pdf` - S sample spectra overlay
- `Fe_beam_damage.pdf` - Temporal evolution under beam
- `S_damage.pdf` - S beam damage with logarithmic fits

### Results (CSV)

**`results/Fe_.csv`**:
- Sample name
- Fe²⁺ peak position (eV)
- Fe³⁺ peak position (eV)
- Fe³⁺ area ratio
- Centroid (eV)
- Fe³⁺/Feᵀᴼᵀ (multiple methods)

**`results/S_res.csv`**:
- Sample name
- S⁶⁺ area ratio
- S⁶⁺ intensity ratio
- S⁶⁺/Sᵀᴼᵀ (L2021, J2010, LL2023)

**`results/colors.csv`**:
- Consistent color scheme for all samples

**Important Note**

Results from the Fe_res and S_res files were manually copied to construct the file `/results/Results_synthese.xlsx`.

## Usage

### Docker (Recommended)
```bash
./docker-run.sh run
```

### Local Execution
```bash
cd src
python analysis_publication.py
```

### Execution Time
a few minutes for the complete dataset (around 100 spectra)

## Script Structure

```python
# Section 1: Fe Standards Analysis
# - Load Fe_glasses_standards sheet
# - Process each standard
# - Generate calibration curve

# Section 2: Fe Sample Analysis  
# - Load Fe_glasses sheet
# - Apply calibration to samples
# - Calculate statistics per sample

# Section 3: Publication Figures
# - Generate multi-panel figures
# - Consistent formatting and colors

# Section 4: S Analysis
# - Load S_glasses sheet
# - Gaussian deconvolution
# - Calculate S6/STOT ratios

# Section 5: Beam Damage
# - Time-series analysis
# - Logarithmic fitting for extrapolation
```

## Dependencies

```python
numpy>=1.20
scipy>=1.7
pandas>=1.3
matplotlib>=3.4
rampy>=0.4       # Raman/XAS baseline correction
xraylarch>=0.9       # X-ray spectroscopy tools
tqdm>=4.62       # Progress bars
scikit-learn>=1.0
uncertainties>=3.1
```

## Key Functions (from `functions.py`)

### `get_Fe_spectrum(filename, ...)`
Load and pre-process Fe K-edge spectrum
- **Returns**: Larch group with normalized spectrum

### `glass_treatment(filename, ...)`
Complete Fe analysis including peak fitting
- **Returns**: Energy, intensity, peak positions, centroid, Fe³⁺ ratio

### `get_S_spectrum(filename, path, ...)`
Load and pre-process S K-edge spectrum
- **Returns**: Larch group with normalized spectrum

### `S_treatment(filename, spectrum_type, ...)`
Complete S analysis including Gaussian fitting
- **Returns**: S⁶⁺ area ratio, intensity ratio, redox values

### `calculate_centroid(centroid, method)`
Convert centroid position to Fe³⁺/Feᵀᴼᵀ
- **Methods**: "F2017", "W2005", "Z2018"

---

**Last updated**: November 2025
