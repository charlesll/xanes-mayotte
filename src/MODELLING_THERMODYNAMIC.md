# Thermodynamic Modelling Documentation

**Script**: `modelling.py`  
**Version**: 1.0  
**Author**: Charles Le Losq  
**Date**: Last Updated November 2025

## Overview

This script performs thermodynamic modelling of Fe and S redox states in basaltic melts using the **Moretti & Ottonello (2005)** Ionic Polymeric Approach (IPA), implemented in the **ctsfg6 Fortran code**.

## Methodology

### Models Implemented

**Ionic Polymeric Approach (Moretti, 2005)**
- Thermochemical model
- Inputs: Composition, T, P, fO₂, fS₂
- Outputs: Fe³⁺/Feᵀᴼᵀ, S⁶⁺/Sᵀᴼᵀ, S_total (ppm)
- Implementation: ctsfg6 Fortran code

**Borisov et al. (2018)**
- Empirical parameterization for Fe³⁺/Feᵀᴼᵀ
- Inputs: Composition, T, P, fO₂
- Forward: fO₂ → Fe³⁺/Feᵀᴼᵀ
- Inverse: Fe³⁺/Feᵀᴼᵀ → fO₂

**Kress & Carmichael (1991)**
- Classic empirical model
- Widely used reference
- Temperature and composition dependent

**Jugo et al. (2010)**
- Empirical model for S⁶⁺/Sᵀᴼᵀ
- Function of ΔFMQ:
```python
S6/STOT = 1 / (1 + 10^(2.1 - 2*ΔFMQ))
```

**Nash et al. (2019)**
- S speciation based on Fe³⁺/Fe²⁺ and T
- From XANES measurements

### Optimization Strategy

The script performs inverse modelling to find optimal fO₂ values by minimizing the difference between calculated and measured redox ratios.

#### Approach 1: Fit Fe³⁺/Feᵀᴼᵀ (Primary)
```python
minimize RMSE(Fe³⁺/Feᵀᴼᵀ_calculated - Fe³⁺/Feᵀᴼᵀ_measured)
```
- Method: Powell (scipy.optimize)
- Initial guess: Kress-Carmichael (1991) solution
- Optimizes one fO₂ value per sample

#### Approach 2: Fit S⁶⁺/Sᵀᴼᵀ (Secondary)
```python
minimize RMSE(S⁶⁺/Sᵀᴼᵀ_calculated - S⁶⁺/Sᵀᴼᵀ_measured)
```
- Independent optimization per sample
- Initial guess: log₁₀(fO₂) = -11.0
- Requires S redox measurements

#### Approach 3: Empirical Models
- Jugo (2010): Direct calculation from ΔFMQ
- Nash (2019): Function of Fe redox and T

## Input Data

**`results/Results_synthese.xlsx`** 

**Main sheet: "synthese"**

Columns:
```
sample          # Sample name
T_start         # Temperature (°C)
P_start         # Pressure (bar, typically 1)
h2o            # H₂O content (wt%)
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
S6             # Measured S⁶⁺/Sᵀᴼᵀ (optional)
dFMQ_Boulliung2023  # ΔFMQ from S redox (Boulliung et al. 2023, see the corresponding Excel sheet for calculation)
C, M, Y, K     # CMYK color codes for plotting
```

**Additional sheet: "Bell_2025"**

Contains MELTS-OSaS thermodynamic calculations from Bell (2025)

**Additional sheet: "Boulliung_2023"**

Contains the calculation of sulfur redox state using the Boulliung and Wood (2023) model.

**Note**: The synthese sheet was compiled manually from the results of the XAS analysis script (`analysis_publication.py`). The Bell_2025 sheet contains independent thermodynamic model results.

## Workflow

```
[1] Load Data
    ↓
[2] Kress-Carmichael Optimization (initial guess)
    ↓
[3] Moretti (2005) Optimization - Fe³⁺ adjustment
    FOR EACH SAMPLE:
      - Prepare composition input
      - Write INPUT.txt and COMPO.txt
      - Run ./ctsfg6 Fortran code
      - Read ctsfg6.jet output
      - Calculate RMSE vs. measured Fe³⁺/Feᵀᴼᵀ
      - Optimize fO₂ to minimize RMSE
    ↓
[4] Moretti (2005) Optimization - S⁶⁺ adjustment (if data available)
    Similar to step 3, but fitting S⁶⁺/Sᵀᴼᵀ
    ↓
[5] Empirical Model Calculations
    - Jugo (2010) for S⁶⁺/Sᵀᴼᵀ
    - Nash (2019) for S⁶⁺/Sᵀᴼᵀ
    - Borisov (2018) for comparison
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
- Moretti (2005) vs. Borisov (2018) vs. Kress-Carmichael (1991)

**`dQFM_S6.pdf`**
- ΔQFM comparison for S⁶⁺-based models  
- Moretti (2005) vs. Jugo (2010) vs. Nash (2019)

**`dFMQ_allmethods.pdf`**
- Comparison of all fO₂ determination methods
- X-axis: SiO₂ content (wt%)
- Y-axis: ΔFMQ (deviation from Fayalite-Magnetite-Quartz buffer)
- Shows three methods:
  - MELTS-OSaS (Bell 2025): filled circles
  - Fe redox state (Moretti 2005, shifted -0.6): open diamonds
  - S redox state (Boulliung 2023): left triangles
- Color-coded by sample using CMYK values
- Uses `dFMQ` data from Bell_2025 sheet in Results_synthese.xlsx

### Results (`results/modelling/`)

**`dQFM_Moretti2005_on_Fe3_adjustment.csv`**

Columns:
```
sample                          # Sample name
T_C                            # Temperature (°C)
log_fO2_M2005_Fe3             # Optimized log₁₀(fO₂) from Fe fit
dQFM_M2005_Fe3                # ΔQFM from Fe fit
log_fO2_M2005_S6              # Optimized log₁₀(fO₂) from S fit
dQFM_M2005_S6                 # ΔQFM from S fit
dQFM_B2018                    # ΔQFM from Borisov (2018)
dQFM_KC1991                   # ΔQFM from Kress-Carmichael (1991)
dQFM_Ju2010                   # ΔQFM from Jugo (2010)
Fe3_measured                  # Measured Fe³⁺/Feᵀᴼᵀ
Fe3_M2005                     # Calculated Fe³⁺/Feᵀᴼᵀ (Moretti)
Fe3_B2018                     # Calculated Fe³⁺/Feᵀᴼᵀ (Borisov)
S6_measured                   # Measured S⁶⁺/Sᵀᴼᵀ
S6_M2005                      # Calculated S⁶⁺/Sᵀᴼᵀ (Moretti)
S6_Ju2010                     # Calculated S⁶⁺/Sᵀᴼᵀ (Jugo)
S6_Na2019                     # Calculated S⁶⁺/Sᵀᴼᵀ (Nash)
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

## References

### Primary Model
- **Moretti & Ottonello (2005)** *Geochimica et Cosmochimica Acta*, 69, 801-823

### Comparison Models
- **Borisov et al. (2018)** *Contributions to Mineralogy and Petrology*, 173:98
- **Kress & Carmichael (1991)** - *Contributions to Mineralogy and Petrology*, 108, 82-92
- **Jugo et al. (2010)** - *Geochimica et Cosmochimica Acta*, 74, 5926-5938
- **Nash et al. (2019)** - *Earth and Planetary Science Letters*, 507, 187-198
- **Boulliung and Wood (2023)** - *Contri Mineral Petrol*, 178, 56

## Support

For questions or issues:
- Check `USAGE.md` for general Docker usage
- Refer to `README.md` for project overview
- Consult Moretti & Ottonello (2005) for model details

---

**Last updated**: November 2025
