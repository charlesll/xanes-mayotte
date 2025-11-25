# Thermodynamic Modelling Documentation

**Script**: `modelling.py`  
**Version**: 1.0  
**Author**: Charles Le Losq  
**Date**: Last Updated November 2025

## Overview

This script performs thermodynamic modelling of Fe and S redox states in basanite-to-phonolite glasses using the Ionic Polymeric Approach (IPA), implemented in the **ctsfg6 Fortran code**.

## Methodology

### Models Implemented

**For iron**
- IPA
- B2018
- KC1991

**For sulfur**
- J2010
- BW2023 (in Results_synthese.xlsx spreadsheet)

### Optimization Strategy

The script performs inverse modelling to find optimal fO₂ values by minimizing the difference between calculated and measured redox ratios.

#### Approach 1: Fit Fe³⁺/Feᵀᴼᵀ
```python
minimize RMSE(Fe³⁺/Feᵀᴼᵀ_calculated - Fe³⁺/Feᵀᴼᵀ_measured)
```
- Method: Powell (scipy.optimize)
- Initial guess: KC1991 solution
- Optimizes one fO₂ value per sample
- Uses KC1991, B2018 and IPA models

#### Approach 2: Fit S⁶⁺/Sᵀᴼᵀ
```python
minimize RMSE(S⁶⁺/Sᵀᴼᵀ_calculated - S⁶⁺/Sᵀᴼᵀ_measured)
```
- Method: Powell (scipy.optimize)
- Initial guess: log₁₀(fO₂) = -11.0
- Optimizes one fO₂ value per sample
- Uses IPA & J2010 models 

## Input Data

**`results/Results_synthese.xlsx`** 

**Main sheet: "synthese"**

Contain the glass compositions and some additional columns copied from either XANES peak fitting results or other calculations (BW2023 model for instance)

Columns:
```
sample          # Sample name
T_start         # Temperature (°C)
P_start         # Pressure (bar, typically 1)
H2O            # H₂O content (wt%)
SiO2           # Oxide composition (wt%)
TiO2
Al2O3
FeO_tot        # Total Fe as FeO
MnO
MgO
CaO
Na2O
K2O
S_ppm          # Total S (ppm)
Fe3            # Measured Fe³⁺/Feᵀᴼᵀ
S6             # Measured S⁶⁺/Sᵀᴼᵀ
dFMQ_Boulliung2023  # ΔFMQ from S redox (BW2023, see the corresponding Excel sheet for calculation)
C, M, Y, K     # CMYK color codes for plotting
```

**Additional sheet: "Bell_2025"**

Contains MELTS-OSaS thermodynamic calculations from B2025

**Additional sheet: "Boulliung_2023"**

Contains the calculation of sulfur redox state using the BW2023 model.

**Note**: The synthese sheet was compiled manually from the results of the XAS analysis script (`analysis_publication.py`). The Bell_2025 sheet contains independent thermodynamic model results.

## Workflow

```
[1] Load Data
    ↓
[2] B2018 + KC1991 Optimization (initial guess)
    ↓
[3] IPA Optimization - Fe³⁺ adjustment
    FOR EACH SAMPLE:
      - Prepare composition input
      - Write INPUT.txt and COMPO.txt
      - Run ./ctsfg6 Fortran code
      - Read ctsfg6.jet output
      - Calculate RMSE vs. measured Fe³⁺/Feᵀᴼᵀ
      - Optimize fO₂ to minimize RMSE
    ↓
[4] IPA Optimization - S⁶⁺ adjustment (if data available)
    Similar to step 3, but fitting S⁶⁺/Sᵀᴼᵀ
    ↓
[5] Empirical Model Calculations
    - J2010 for S⁶⁺/Sᵀᴼᵀ
    - BW2023 for S⁶⁺/Sᵀᴼᵀ
    ↓
[6] Calculate ΔQFM values
    ↓
[7] Generate Comparison Figures (5 PDFs)
    - Temperature_SiO2.pdf
    - models.pdf (6 panels)
    - dQFM_Fe3.pdf
    - dQFM_S6.pdf
    - dFMQ_allmethods.pdf (All methods comparison, including MELTS-OSaS)
    ↓
[8] Save Results (CSV)
```

## Output Files

### Figures (`figures/Modelling/`)

**`Temperature_SiO2.pdf`**
- Temperature vs. SiO₂ content
- Shows compositional range of samples

**`models.pdf`** (6 panels)
- Compare results obtained when fitting iron or sulfur data

**`dQFM_Fe3.pdf`**
- ΔQFM comparison for Fe³⁺-based models
- IPA vs. B2018 vs. KC1991

**`dQFM_S6.pdf`**
- ΔQFM comparison for S⁶⁺-based models  
- IPA vs. J2010 vs. BW2023

**`dFMQ_allmethods.pdf`**
- Comparison of all fO₂ determination methods
- X-axis: SiO₂ content (wt%)
- Y-axis: ΔFMQ (deviation from Fayalite-Magnetite-Quartz buffer)
- Shows three methods:
  - MELTS-OSaS (B2025): filled circles
  - Fe redox state (IPA, shifted -0.5): open diamonds
  - S redox state (BW2023): left triangles
- Color-coded by sample using CMYK values
- Uses `dFMQ` data from Bell_2025 sheet in Results_synthese.xlsx

### Results (`results/`)

**`results/dQFM_models.csv`**

Columns:
```
sample                       # Sample name
dQFM_Fe_M2005                # ΔQFM Fe data + IPA model
dQFM_Fe_KC1991               # ΔQFM Fe data + KC1991
dQFM_Fe_B2018                # ΔQFM Fe data + B2018
dQFM_S_M2005                 # ΔQFM S data + IPA model
dQFM_S_J2010                 # ΔQFM S data + J2010
dQFM_S_BW2023                # ΔQFM S data + BW2023
```

**`results/modelling/dQFM_Moretti2005_on_Fe3_adjustment.csv`**

Legacy format file containing:
```
sample                       # Sample name
dFMQ                         # ΔQFM from IPA Fe³⁺ optimization only
```

## Key Functions

### `forward(theta, data_, theta_mode="none")`
Forward model calling ctsfg6 Fortran code
- **Inputs**: fO₂ values (theta), sample data
- **Outputs**: Fe³⁺/Feᵀᴼᵀ, S⁶⁺/Sᵀᴼᵀ, S_total arrays
- **Process**:
  1. Write INPUT.txt (T, P, fO₂, fS₂)
  2. Write COMPO.txt (oxide composition)
  3. Call `./ctsfg6` subprocess
  4. Read ctsfg6.jet output
  5. Parse results

### `forward_simu(theta, data_, T_simu)`
Forward model for temperature simulations
- Similar to `forward()` but with custom temperatures

### `objective_function_M2005(theta, mode, data_, measured)`
Objective function for scipy.optimize
- **Inputs**: fO₂ guess, fitting mode ("Fe" or "S"), data, measurements
- **Output**: RMSE to minimize
- **Used by**: `scipy.optimize.minimize()`

### Helper Functions

**From `opt_functions.py`**:
- `QFM(T, P)`: Calculate QFM buffer
- `wt_mol(composition)`: Convert wt% to mol%
- `chimie_control(composition)`: Normalize composition
- `redox_B2018(composition, T, P, fO2)`: Borisov model
- `fo2_B2018(composition, T, P, Fe3)`: Inverse Borisov
- `redox_KC1991(composition, T, P, fO2)`: Kress-Carmichael model

## Usage

### Docker (Recommended)

```bash
# Modelling only (requires Results_synthese.xlsx)
./docker-run.sh model

# Full pipeline (XAS + Modelling)
./docker-run.sh all
```

Note that the Fortran code is automatically compiled during Docker build:
```dockerfile
RUN cd src && \
    gfortran ctsfg6.for -o ctsfg6 && \
    chmod +x ctsfg6
```

### Local Execution

```bash
cd src
python modelling.py
```

**Prerequisites**:
- Fortran compiler (gfortran)
- ctsfg6 compiled: `gfortran ctsfg6.for -o ctsfg6`
- Input file: `../results/Results_synthese.xlsx`

### Execution Time

- ~1 minutes for 13 samples
- Dominated by optimization iterations

## Troubleshooting

### "FileNotFoundError: ctsfg6"
```bash
# Compile Fortran code
cd src
gfortran ctsfg6.for -o ctsfg6
chmod +x ctsfg6
```

### "FileNotFoundError: Results_synthese.xlsx"
```bash
# Run XAS analysis first
./docker-run.sh run
# Or use full pipeline
./docker-run.sh all
```

### "ctsfg6.jet parsing error"
- Check INPUT.txt and COMPO.txt are correctly formatted
- Ensure composition is normalized
- Check for NaN values in input data

### Optimization doesn't converge
- Try different initial guess
- Check measured Fe³⁺/Feᵀᴼᵀ is physically reasonable (0-1)
- Verify temperature and composition are realistic

## Support

For questions or issues:
- Check `USAGE.md` for general Docker usage
- Refer to `README.md` for project overview
- Consult Moretti & Ottonello (2005) for model details

---

**Last updated**: November 2025
