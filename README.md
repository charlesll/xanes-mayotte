# XAS Analysis & Thermodynamic Modelling - Mayotte Volcanic Samples

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-cc-by.svg)](LICENSE.md)

X-ray Absorption Spectroscopy (XAS) analysis and thermodynamic modelling pipeline for Mayotte volcanic samples, including Fe K-edge and S K-edge XANES analysis with redox state determination.

## 🚀 Quick Start

**Recommended method** - Docker ensures reproducibility across all systems. After installing Docker (see the instructions online), run in a terminal: 

```bash
# 1. Build Docker image (once)
./docker-run.sh build

# 2. Run complete pipeline (XAS + Thermodynamic Modelling)
./docker-run.sh all

# Or run components separately:
./docker-run.sh run     # XAS analysis only
./docker-run.sh model   # Thermodynamic modelling only
./docker-run.sh help    # Show all commands
```

**That's it!** Results will be in `figures/` and `results/` directories.

📖 **In-depth Docker guide**: See [`DOCKER.md`](DOCKER.md)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Output Files](#-output-files)
- [Documentation](#-documentation)
- [Citation](#-citation)
- [License](#-license)

---

## 🔬 Overview

This repository provides the code to replicate the results from the publication

*Charles Le Losq, Roberto Moretti, Etienne Médard, Carole Berthod, Federica Schiavi, Nicolas Trcera, Elodie Lebas (2025) Oxidation state of Mayotte magmatic series: insights from Fe and S
K-edge XANES spectroscopy. Submitted to Journal Volcanica.*

It leverages two scripts to perform the analysis, as described below.

### XAS Analysis (`src/analysis_publication.py`)

This scripts uses [xraylarch](https://xraypy.github.io/xraylarch/) and [rampy](https://charlesll.github.io/rampy/) to 

- treat the Fe K-edge XANES spectra and provide Fe redox state determination (Fe³⁺/Feᵀᴼᵀ)
- treat the S K-edge XANES spectra and provide sulfur speciation (S²⁻, S⁶⁺)
- perform beam-damage analysis
- generate figures

### Thermodynamic Modelling (`src/modelling.py`)

This scripts leverages the compositional data (including volatile contents and S/Fe redox states) available in `results/Results_synthese.xlsx` to calculate magmatic fO₂ by inverse modelling using the [Moretti and Ottonello (2005) IPA model](https://github.com/charlesll/sulfur-magma). Results are compared with various models.

---

## 💻 Requirements

### Docker (Recommended)
- **Docker Desktop** (Windows, macOS, Linux)
  - Download: https://www.docker.com/products/docker-desktop
- **Disk space**: ~2 GB for Docker image
- **RAM**: 4 GB minimum, 8 GB recommended

### Local Installation (Alternative)
- **Python**: 3.10 or higher
- **Fortran compiler**: gfortran (Linux/macOS only - not easily available on Windows)
- **Python packages**: See `requirements.txt`
- **OS**: Linux or macOS (Windows requires WSL for Fortran)

**⚠️ Windows users**: Docker is strongly recommended due to Fortran compiler limitations.

---

## 🔧 Installation

### Method 1: Docker (⭐ Recommended)

```bash
# Prerequisites: Docker Desktop installed and running

# Clone or download this repository
cd Mayotte_publication

# Build the Docker image (includes Fortran compilation)
./docker-run.sh build

# Ready to use!
```

**That's it!** Docker handles all dependencies, including:
- Python 3.10 and all packages
- Fortran compiler (gfortran)
- ctsfg6 compilation
- System libraries (HDF5, BLAS, LAPACK)

### Method 2: Local Installation (Development only)

```bash
# Prerequisites: Python 3.10+, gfortran

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Install Python dependencies
pip install -r requirements.txt

# Compile Fortran code
cd src
gfortran ctsfg6.for -o ctsfg6
chmod +x ctsfg6
cd ..
```

**⚠️ Note**: Local installation does not work on Windows without WSL due to Fortran requirements.

---

## 🎯 Usage

### Docker (Recommended)

```bash
# Complete pipeline (XAS + Modelling)
./docker-run.sh all

# Individual components
./docker-run.sh run     # XAS analysis only (~15-20 min)
./docker-run.sh model   # Thermodynamic modelling only (~2-5 min)

# Utilities
./docker-run.sh exec    # Open interactive shell in container
./docker-run.sh logs    # View execution logs
./docker-run.sh clean   # Remove container and image
./docker-run.sh help    # Show all available commands
```

### Local Execution

```bash
cd src

# XAS analysis
python analysis_publication.py

# Thermodynamic modelling (requires Results_synthese.xlsx from XAS)
python modelling.py
```

### Typical Workflow

```bash
# First time: Build image
./docker-run.sh build

# Every time: Run analysis
./docker-run.sh all

# Check results
ls figures/Iron/ figures/Sulfur/ figures/Modelling/
ls results/*.csv results/modelling/*.csv
```

---

## 📁 Project Structure

```
Mayotte_publication/
│
├── src/                              # Source code
│   ├── analysis_publication.py       # XAS analysis script (839 lines)
│   ├── modelling.py                  # Thermodynamic modelling (665 lines)
│   ├── functions.py                  # XAS processing functions (455 lines)
│   ├── opt_functions.py              # Optimization functions (653 lines)
│   ├── ctsfg6.for                    # Fortran code (Moretti 2005)
│   ├── ctsfg6                        # Compiled executable (generated)
│   ├── ANALYSIS_XAS.md               # XAS script documentation
│   └── MODELLING_THERMODYNAMIC.md    # Modelling script documentation
│
├── xas/                              # Input: Raw XAS spectra (read-only)
│   ├── iron/*.dat                    # Fe K-edge spectra
│   └── sulfur/*.dat                  # S K-edge spectra
│
├── tables/
(read-only)
│   └── liste.xlsx                        # Input: Sample metadata
│   └── Microsonde_Mayotte_Lucia.xlsx # analysis results of the various samples
│
├── results/                          # Output: CSV results (read-write)
│   ├── Fe_res.csv                    # Fe redox results
│   ├── S_res.csv                     # S speciation results
│   ├── colors.csv                    # Color scheme
│   ├── dQFM_models.csv               # dFMQ modelling results
(read-only)
│   ├── Results_synthese.xlsx         # Combined results (input for modelling, modify manually using the Fe_ and S_res files)
│   └── MELTS-OSaS/                   # MELTS-OSaS FOLDER
│        ├── OSaS_MAY_Input.xlsx      # input file
│        ├── Olivine_Spinel_OxygenBarometry_Mayotte.ipynb   # code, run with ThermoEngine docker
│        └── MELTSOSaS_OSaS_MAY_FO2_CALC.xlsx           # MELTS-OSaS results
│
├── figures/                          # Output: Publication figures (read-write)
│   ├── Iron/                         # Individual Fe spectra PDFs
│   ├── Sulfur/                       # Individual S spectra PDFs
│   ├── Modelling/                    # Thermodynamic figures
│   │   ├── Temperature_SiO2.pdf
│   │   ├── models.pdf
│   │   ├── dQFM_Fe3.pdf
│   │   ├── dQFM_S6.pdf
│   │   └── dFMQ_allmethods.pdf
│   ├── calibration.pdf               # Fe calibration curve
│   ├── Spectra_refs.pdf              # Reference spectra
│   ├── Spectra_samples.pdf           # Fe sample spectra
│   ├── Spectra_S_samples.pdf         # S sample spectra
│   ├── Fe_beam_damage.pdf            # Beam damage analysis
│   └── S_damage.pdf                  # S beam damage
│
├── Dockerfile                        # Docker configuration
├── docker-run.sh                     # Docker helper script
├── requirements.txt                  # Python dependencies
│
├── README.md                         # This file
└── DOCKER.md                         # Docker technical details
```

---

## 📊 Output Files

### XAS Analysis Results

**Figures** (`figures/`):
- `calibration.pdf` - Fe centroid vs. Fe³⁺/Feᵀᴼᵀ calibration curve
- `Spectra_refs.pdf` - Fe reference spectra (multi-panel)
- `Spectra_samples.pdf` - Fe sample spectra overlay
- `Spectra_S_samples.pdf` - S sample spectra overlay
- `Fe_beam_damage.pdf` - Time-series analysis (7112, 7114 eV)
- `S_damage.pdf` - S beam damage with logarithmic fits
- `Iron/*.pdf` - Individual Fe spectra with peak fits
- `Sulfur/*.pdf` - Individual S spectra with Gaussian deconvolution

**Results** (`results/`):
- `Fe_res.csv` - Fe redox results (centroid, Fe³⁺/Feᵀᴼᵀ, multiple methods)
- `S_res.csv` - S speciation (S⁶⁺/Sᵀᴼᵀ by L2021, J2010, LL2023)
- `colors.csv` - Consistent color scheme for all samples
- `Results_synthese.xlsx` - Combined dataset used as input for modelling, manually created from `Fe_res.csv` and `S_res.csv` as well as other analysis (Boulliung model, Bell MELTS-OSaS, etc.).

### Thermodynamic Modelling Results

**Figures** (`figures/Modelling/`):
- `Temperature_SiO2.pdf` - Temperature vs. SiO₂ content
- `models.pdf` - 6-panel comparison (Fe, S, ΔQFM)
- `dQFM_Fe3.pdf` - ΔQFM comparison for Fe-based models
- `dQFM_S6.pdf` - ΔQFM comparison for S-based models
- `dFMQ_allmethods.pdf` - Comprehensive comparison (MELTS-OSaS, Fe redox, S redox)

**Results** (`results/`):
- `dQFM_models.csv` - calculated deviations to the Fayalite-Magnetite-Quartz buffer (ΔQFM):
  - Sample names
  - ΔQFM from Fe data: IPA (Moretti 2005), KC1991, B2018
  - ΔQFM from S data: IPA (Moretti 2005), J2010, BW2023
- `modelling/dQFM_Moretti2005_on_Fe3_adjustment.csv` - Legacy format (IPA Fe³⁺ optimization only)

---

## 📚 Documentation

- **[src/ANALYSIS_XAS.md](src/ANALYSIS_XAS.md)** - XAS analysis script documentation
- **[src/MODELLING_THERMODYNAMIC.md](src/MODELLING_THERMODYNAMIC.md)** - Thermodynamic modelling documentation
- **[DOCKER.md](DOCKER.md)** - Docker technical details and troubleshooting

---

## 🔬 Scientific Methods

### XAS Analysis
- **Fe K-edge**: Pre-edge removal (Larch), pseudo-Voigt fitting, centroid method
- **S K-edge**: Background removal (Rampy), multi-Gaussian deconvolution
- **Calibrations**: Fe => F2017, W2005, Z2018; S => L2021, J2010, LL2023 (this study, not used)

### Thermodynamic Modelling
- **Primary**: Moretti & Ottonello (2005) - Ionic Polymeric Approach (IPA)
- **Implementation**: [ctsfg6 Fortran code](https://github.com/charlesll/sulfur-magma) (2632 lines, Fortran 77)
- **Method**: Inverse modelling with Powell optimization
- **Comparison**: B2018, KC1991, J2010, BW2023 (Excel calculation)

- **MELTS-OSaS**: We provide the notebook and input we used in the publication to determine the fO2 given the composition of melt, olivine and spinel. The code is a direct minor modification of the Waters et al. (2025) software, see [https://zenodo.org/records/13988167](https://zenodo.org/records/13988167). The best way to run it is through the [ThermoEngine plateform](https://thermoenginelite.readthedocs.io). See the `results/MELTS-OSaS/` folder for code and input files. You can directly copied the results in the `results/Results_synthese.xlsx`spreadsheet prior to running the modelling section of the code.

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{mayotte_xas_2025,
  author = {Charles Le Losq, Roberto Moretti, Étienne Médard, Carole Berthod, Federica Schiavi, Nicolas Trcera, Élodie Lebas},
  title = {XAS Analysis and Thermodynamic Modelling for Mayotte Volcanic Samples},
  year = {2025},
  publisher = {Zenodo},
  url = {https://github.com/[YOUR_REPO]}
}
```

**Related Publications**:
- Charles Le Losq, Roberto Moretti, Etienne Médard, Carole Berthod, Federica Schiavi, Nicolas Trcera, Elodie Lebas (2025) Oxidation state of Mayotte magmatic series: insights from Fe and S
K-edge XANES spectroscopy. Submitted to Journal Volcanica.

---

## 📝 License

This project is licensed under the CC BY 4.0 License - see [LICENSE.md](LICENSE.md) for details.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

---

## 💬 Support

### Questions or Issues?

1. **Check documentation**: Start with this [README.md](README.md)
2. **Docker problems**: See [DOCKER.md](DOCKER.md)
3. **Scientific questions**: Consult the paper (see above)
4. **Bug reports**: Open an issue on GitHub

### Contact

- **Author**: Charles Le Losq
- **Email**: lelosq@ipgp.fr
- **Institution**: Université Paris Cité, Institut de physique du globe de Paris, CNRS, Institut Universitaire de France

---

## 🙏 Acknowledgments

- **[Larch](https://xraypy.github.io/xraylarch/)**: X-ray spectroscopy tools (Matt Newville)
- **[Rampy](https://charlesll.github.io/rampy/)**: Raman/XAS baseline correction (Charles Le Losq)
- **Moretti & Ottonello**: [CTSFG](https://github.com/charlesll/sulfur-magma) thermodynamic model
- **Synchrotron facilities**: [LUCIA beamline on SOLEIL synchrotron](https://www.synchrotron-soleil.fr/en/beamlines/lucia)

---

**Version**: 1.0  
**Last updated**: November 2025  
**Status**: Production-ready ✅

**Ready to start?** Run `./docker-run.sh build` then `./docker-run.sh all` 🚀
