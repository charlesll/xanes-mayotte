#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modelling Script for Mayotte Volcanic Samples - Publication Version
====================================================================

This script performs thermodynamic modelling of Fe and S redox states
using the Moretti (2005) model and the ctsfg6 Fortran code.

Data Input:
-----------
- results/Results_synthese.xlsx (synthese sheet):  composition, temperature, measured Fe3+/FeTOT and S6+/STOT

Processing:
-----------
1. Load sample compositions and measured redox states
2. Optimize fO2 values to fit Fe3+/FeTOT ratios using IPA model
3. Optimize fO2 values to fit S6+/STOT ratios using IPA model
4. Compare with other models (KC1991, B2018, J2010, BW2023)
5. Calculate dQFM values (deviation from Quartz-Fayalite-Magnetite buffer)
6. Generate comparison figures

Outputs:
--------
- figures/Modelling/Temperature_SiO2.pdf: Temperature vs SiO2
- figures/Modelling/models.pdf: Model comparison (Fe3+, S6+, dQFM)
- figures/Modelling/dQFM_Fe3.pdf: dQFM comparison for Fe3+ models
- figures/Modelling/dQFM_S6.pdf: dQFM comparison for S6+ models
- figures/Modelling/dFMQ_allmethods.pdf: All methods comparison (MELTS-OSaS, Fe redox, S redox)
- results/modelling/dQFM_Moretti2005_on_Fe3_adjustment.csv: Optimized dQFM values

Methods:
--------
Uses the ctsfg6 Fortran code (Moretti & Ottonello 2005) which implements
the ionic-polymeric approach for redox and sulfur speciation in silicate melts.

Author: Charles Le Losq
Date: Last Update November 2025
"""

import os
import csv
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Import custom functions
from opt_functions import QFM, wt_mol, chimie_control, redox_B2018, fo2_B2018, redox_KC1991

# ==============================================================================
# SECTION 1: FORWARD MODEL FUNCTIONS
# ==============================================================================

def forward(theta, data_, theta_mode="none"):
    """
    Forward model using ctsfg6 Fortran code.
    
    Parameters
    ----------
    theta : array-like
        Parameters for optimization:
        - theta[0:13]: log10(fO2) values for each sample
        - theta[13:26]: optional parameters (T, H2O, or fS2) depending on theta_mode
    data_ : pandas.DataFrame
        Sample compositions and conditions
    theta_mode : str
        "none": use data values for T, H2O, fS2
        "Temperature": use theta[13:26] as temperature values
        "Water": use theta[13:26] as H2O values
        "Sulfur": use theta[13:26] as fS2 values
    
    Returns
    -------
    FE3_FETOT : ndarray
        Calculated Fe3+/(Fe2+ + Fe3+) ratios
    S6_STOT : ndarray
        Calculated S6+/STOT ratios
    S_TOT : ndarray
        Calculated total sulfur content (ppm)
    """
    # Get the number of samples
    nb_val = len(data_)

    # Default values from data
    H2O = data_.loc[:, "h2o"]
    temperatures = data_.loc[:, "T_start"]
    f_S2 = -1.0*np.ones(nb_val)

    # Specific cases based on theta_mode
    if theta_mode == "Temperature":
        temperatures = theta[13:26]
    elif theta_mode == "Water":
        H2O = theta[13:26]
    elif theta_mode == "Sulfur":
        f_S2 = theta[13:26]

    # Construct input DataFrame for ctsfg6
    db = pd.DataFrame({
        "SiO2": data_.sio2,
        "TiO2": data_.tio2,
        "Al2O3": data_.al2o3,
        "Fe2O3": np.zeros(nb_val),
        "Cr2O3": np.zeros(nb_val),
        "FeO": data_.feo,
        "MnO": data_.mno,
        "MgO": data_.mgo,
        "CaO": data_.cao,
        "Na2O": data_.na2o,
        "K2O": data_.k2o,
        "P2O5": data_.p2o5,
        "H2O": H2O,
        "S_tot_ppm": data_.S_,
        "so2": np.zeros(nb_val),
        "T(C)": temperatures,
        "Pbar": 350.0*np.ones(nb_val),
        "xossi_fo2": theta[0:13],
        "fs2": f_S2,
        "index_author": np.zeros(nb_val),
        "kflag": np.zeros(nb_val),
        "wmol": 0.01*np.ones(nb_val),
        "kir": np.zeros(nb_val)
    })

    # Export the data to a text file for input to the Ctsfg6 code
    db.to_csv("COMPO.txt", index=False, float_format="%.5g")

    # Add the appropriate header
    with open('INPUT.txt', 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow([str(nb_val), "", "", "", "", "", "", "", "", "", "", "", "", 
                        "", "", "", "", "", "", "", "", "", ""])

        with open('COMPO.txt', 'r', newline='') as incsv:
            reader = csv.reader(incsv)
            writer.writerows(row + [0.0] for row in reader)

    # Run the Fortran code
    subprocess.run(["./ctsfg6"], check=True)

    # Retrieve results
    results = pd.read_csv("./ctsfg6.jet", skiprows=1, 
                         names=["T(K)", "P(bar)", "logfO2(in)", "logfO2(calc)", "logfS2", "aossi",
                                "TotAni", "TotCat", "nO=", "nO-", "nO0", "NBO/T", "Kpol", 
                                "Stot(obs wt%)", "Stot(calc wt%)", "S_as_S2-(wt%)", "S_as_S6+(wt%)",
                                "S6+/tot", "log(KSO4/KS2)", "Redox", "Redoz", "actFe2+", 
                                "cost_FeO", "kflag"])

    # Get the ratio of Fe2/Fe3, log10 unit
    FE2_FE3 = results["Redox"].values

    # Convert to Fe3/(Fe2 + Fe3) ratio
    FE3_FETOT = 1 - (10**FE2_FE3) / (1 + (10**FE2_FE3))

    # Get the sulfur redox
    S6_STOT = results.loc[:, "S6+/tot"].values

    # Get the sulfur concentration in ppm (convert from wt%)
    S_TOT = results.loc[:, "Stot(calc wt%)"].values * 10000

    return FE3_FETOT, S6_STOT, S_TOT


def forward_simu(data_):
    """
    Simplified forward model for simulations (no optimization parameters).
    
    Parameters
    ----------
    data_ : pandas.DataFrame
        Sample compositions including h2o and log10_fo2 columns
    
    Returns
    -------
    FE3_FETOT : ndarray
        Calculated Fe3+/(Fe2+ + Fe3+) ratios
    S6_STOT : ndarray
        Calculated S6+/STOT ratios
    S_TOT : ndarray
        Calculated total sulfur content (ppm)
    """
    nb_val = len(data_)
    temperatures = data_.loc[:, "T_start"]
    f_S2 = -1.0 * np.ones(nb_val)

    db = pd.DataFrame({
        "SiO2": data_.sio2,
        "TiO2": data_.tio2,
        "Al2O3": data_.al2o3,
        "Fe2O3": np.zeros(nb_val),
        "Cr2O3": np.zeros(nb_val),
        "FeO": data_.feo,
        "MnO": data_.mno,
        "MgO": data_.mgo,
        "CaO": data_.cao,
        "Na2O": data_.na2o,
        "K2O": data_.k2o,
        "P2O5": data_.p2o5,
        "H2O": data_.loc[:, "h2o"],
        "S_tot_ppm": data_.S_,
        "so2": np.zeros(nb_val),
        "T(C)": temperatures,
        "Pbar": 500.0 * np.ones(nb_val),
        "xossi_fo2": data_.loc[:, "log10_fo2"],
        "fs2": f_S2,
        "index_author": np.zeros(nb_val),
        "kflag": np.zeros(nb_val),
        "wmol": 0.01 * np.ones(nb_val),
        "kir": np.zeros(nb_val)
    })

    db.to_csv("COMPO.txt", index=False, float_format="%.5g")

    with open('INPUT.txt', 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow([str(nb_val), "", "", "", "", "", "", "", "", "", "", "", "",
                        "", "", "", "", "", "", "", "", "", ""])
        with open('COMPO.txt', 'r', newline='') as incsv:
            reader = csv.reader(incsv)
            writer.writerows(row + [0.0] for row in reader)

    subprocess.run(["./ctsfg6"], check=True)

    results = pd.read_csv("./ctsfg6.jet", skiprows=1,
                         names=["T(K)", "P(bar)", "logfO2(in)", "logfO2(calc)", "logfS2", "aossi",
                                "TotAni", "TotCat", "nO=", "nO-", "nO0", "NBO/T", "Kpol",
                                "Stot(obs wt%)", "Stot(calc wt%)", "S_as_S2-(wt%)", "S_as_S6+(wt%)",
                                "S6+/tot", "log(KSO4/KS2)", "Redox", "Redoz", "actFe2+",
                                "cost_FeO", "kflag"])

    FE2_FE3 = results["Redox"].values
    FE3_FETOT = 1 - (10**FE2_FE3) / (1 + (10**FE2_FE3))
    S6_STOT = results.loc[:, "S6+/tot"].values
    S_TOT = results.loc[:, "Stot(calc wt%)"].values * 10000

    return FE3_FETOT, S6_STOT, S_TOT


# ==============================================================================
# SECTION 2: OBJECTIVE FUNCTIONS
# ==============================================================================

def objective_function_M2005(theta, mode="Fe3", theta_mode="none"):
    """
    Objective function for optimization using IPA model.
    
    Parameters
    ----------
    theta : array-like
        Parameters to optimize
    mode : str
        "Fe3": fit Fe3+/FeTOT only
        "S6": fit S6+/STOT only
        "both": fit both Fe3+ and S6+
        "all": fit Fe3+, S6+, and S content
    theta_mode : str
        Passed to forward() function
    
    Returns
    -------
    residuals : float
        Root mean squared error(s) to minimize
    """
    fe3_pred, s6_pred, s_pred = forward(theta, data_, theta_mode=theta_mode)
    
    # Calculate the residuals
    residuals_1 = np.sqrt(np.mean((fe3_pred - data_["Fe3"])**2))
    residuals_2 = np.sqrt(np.mean((s6_pred - data_["S6"])**2))
    residuals_3 = np.sqrt(np.mean((s_pred - data_["S_"])**2)) / 1000  # scaling factor

    if mode == "Fe3":
        residuals = residuals_1
    elif mode == "S6":
        residuals = residuals_2
    elif mode == "both":
        residuals = residuals_1 + residuals_2
    elif mode == "all":
        residuals = residuals_1 + residuals_2 + residuals_3
    else:
        raise ValueError("Invalid mode. Choose from 'Fe3', 'S6', 'both', or 'all'.")
    
    # Check that residuals is a finite number
    if not np.isfinite(residuals):
        return np.nan

    return residuals


# ==============================================================================
# SECTION 3: PLOTTING FUNCTIONS
# ==============================================================================

def plot_model(ax, x, y, error, colors, lim, xlabel, ylabel, title=None, annotation=None):
    """
    Plot model predictions vs measurements with 1:1 line.
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    x : array-like
        Measured values
    y : array-like
        Predicted values
    error : float
        Error bar size
    colors : array-like
        Colors for each point
    lim : float
        Axis limits (0 to lim)
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str, optional
        Plot title
    annotation : str, optional
        Annotation text (e.g., "(a)")
    """
    if title:
        ax.set_title(title)
    if annotation:
        ax.annotate(annotation, xy=(0.5, 0.97), xycoords="axes fraction", 
                   ha="center", va="top")
    for i in range(len(x)):
        ax.errorbar(x[i], y[i], xerr=error, marker="s", linestyle="none", color=colors[i])
    ax.plot([0, lim], [0, lim], "--", color="grey")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_dqfm(ax, x, y, error, colors, marker='s', markersize=5, annotation=None, 
              ylim=None, xlabel=r"Measured $Fe^{3+}/Fe^\mathregular{TOT}$", 
              ylabel='$\\Delta$FMQ'):
    """
    Plot dQFM values vs measured Fe3+/FeTOT.
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    x : array-like
        Measured Fe3+/FeTOT values
    y : array-like
        dQFM values
    error : float
        Error bar size
    colors : array-like
        Colors for each point
    marker : str
        Marker style
    markersize : float
        Marker size
    annotation : str, optional
        Annotation text
    ylim : tuple, optional
        Y-axis limits
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    """
    if annotation:
        ax.annotate(annotation, xy=(0.5, 0.97), xycoords="axes fraction", 
                   ha="center", va="top")
    for i in range(len(x)):
        ax.errorbar(x[i], y[i], xerr=error, marker=marker, linestyle="none", 
                   markersize=markersize, color=colors[i])
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def Ju2010(dFMQ):
    """
    Calculate S6+/STOT from dFMQ values according to J2010
    
    Parameters
    ----------
    dFMQ : array-like
        Deviation from FMQ buffer
    
    Returns
    -------
    S6_STOT : array-like
        S6+/STOT ratio
    """
    return 1 / (1.0 + 10**(2.1 - 2 * dFMQ))


def Nash2019(Fe3_Fe2, T):
    """
    Calculate S6+/STOT from Fe3+/Fe2+ and T according to N2019.
    
    Parameters
    ----------
    Fe3_Fe2 : array-like
        Fe3+/Fe2+ ratio
    T : array-like
        Temperature in K
    
    Returns
    -------
    S6_STOT : array-like
        S6+/STOT ratio
    """
    log_S6_S2 = 8.0*np.log10(Fe3_Fe2) + 8.7436e6/T**2 - 27703/T + 20.273
    return 10**log_S6_S2 / (1 + 10**log_S6_S2)


def simulate_Fe3_vs_H2O(row, h2o_start=3.0, fo2_start=-0.5):
    """
    Simulate Fe3+/FeTOT as a function of H2O for a given composition.
    
    Parameters
    ----------
    row : pandas.Series
        Sample composition from data_ DataFrame
    h2o_start : float
        Starting H2O content (wt%)
    fo2_start : float
        Starting fO2 deviation from QFM
    
    Returns
    -------
    data_simu : DataFrame
        Simulation input data
    fe3_simu : ndarray
        Simulated Fe3+/FeTOT values
    s6_simu : ndarray
        Simulated S6+/STOT values
    s_simu : ndarray
        Simulated S content (ppm)
    """
    h2o_range = np.arange(h2o_start, row.h2o, -0.1)
    data_simu = pd.DataFrame([row] * len(h2o_range)).reset_index(drop=True)
    data_simu["h2o"] = h2o_range
    data_simu['log10_fo2'] = QFM(data_simu.T_start + 273.15, P=350.0) + \
                             np.linspace(fo2_start, row.dQFM_M2005, len(h2o_range))
    fe3_simu, s6_simu, s_simu = forward_simu(data_simu)
    return data_simu, fe3_simu, s6_simu, s_simu


# ==============================================================================
# SECTION 4: MAIN ANALYSIS
# ==============================================================================

def main():
    """
    Main analysis routine for thermodynamic modelling.
    """
    print("="*80)
    print("Mayotte Volcanic Samples - Thermodynamic Modelling")
    print("Using IPA model and ctsfg6 Fortran code")
    print("="*80)
    
    # Create output directories
    os.makedirs("../figures/Modelling", exist_ok=True)
    os.makedirs("../results/modelling", exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    print("\n[1/8] Loading data from Results_synthese.xlsx...")
    global data_  # Make accessible to objective function
    data_ = pd.read_excel("../results/Results_synthese.xlsx", 
                          sheet_name="synthese")
    nb_points = len(data_)
    print(f"      Loaded {nb_points} samples")
    
    # Extract measured values and errors
    fe3_ = data_.loc[:, "Fe3"].values
    s6_ = data_.loc[:, "S6"].values
    stot_ = data_.loc[:, "S_"].values
    fe3_err_ = 0.02 * fe3_
    fe3_err_[fe3_err_ == 0.] = 1.
    s6_err_ = 0.05 * s6_
    s6_err_[s6_err_ == 0.] = 1.
    stot_err_ = data_.loc[:, "ese_S_"].values
    stot_err_[stot_err_ == 0.] = 1.
    
    # Convert to molar composition
    data_mol = wt_mol(chimie_control(data_))
    
    # QFM buffer values at sample temperatures
    QFM_values = QFM(data_.loc[:, "T_start"] + 273.15, P=350.0).values
    
    # =========================================================================
    # STEP 2: Calculate fO2 using B2018
    # =========================================================================
    print("\n[2/8] Calculating fO2 using B2018 model...")
    log10_fo2 = fo2_B2018(data_mol, data_["Fe3"], data_["T_start"] + 273.15)
    dQFM_B2018 = log10_fo2 - QFM_values
    print(f"      dQFM (B2018): Median {np.median(dQFM_B2018):.1f} | "
          f"Min {np.min(dQFM_B2018):.1f} | Max {np.max(dQFM_B2018):.1f}")
    
    # =========================================================================
    # STEP 3: Optimize using KC1991
    # =========================================================================
    print("\n[3/8] Optimizing fO2 using KC1991 model...")
    objective_function_KC = lambda theta: np.sum(
        (redox_KC1991(data_mol, 10**theta, data_["T_start"] + 273.15) - data_["Fe3"])**2
    )
    theta_start_KC = [-9.03, -8.26, -8.50, -8.275506, -9.664400, -10.512570, 
                     -10.444127, -9.152720, -8.145305, -8.087077, -7.796709, 
                     -6.145855, -10.483786]
    res_KC = minimize(objective_function_KC, theta_start_KC, method="Nelder-Mead")
    dQFM_KC = res_KC.x - QFM_values
    print(f"      dQFM (KC1991): Median {np.median(dQFM_KC):.1f} | "
          f"Min {np.min(dQFM_KC):.1f} | Max {np.max(dQFM_KC):.1f}")
    
    # =========================================================================
    # STEP 4: Optimize using IPA model - Fe3+ adjustment
    # =========================================================================
    print("\n[4/8] Optimizing fO2 using IPA model (Fe3+ adjustment)...")
    res_Fe3 = minimize(objective_function_M2005, x0=theta_start_KC, 
                      method="Powell", args=("Fe3", False))
    preds_Fe3 = forward(res_Fe3.x, data_, theta_mode=False)
    dQFM_Fe3 = res_Fe3.x - QFM_values
    print(f"      dQFM (IPA-Fe3): Median {np.median(dQFM_Fe3):.1f} | "
          f"Min {np.min(dQFM_Fe3):.1f} | Max {np.max(dQFM_Fe3):.1f}")
    
    # =========================================================================
    # STEP 5: Optimize using IPA model - S6+ adjustment
    # =========================================================================
    print("\n[5/8] Optimizing fO2 using IPA model (S6+ adjustment)...")
    res_S6 = minimize(objective_function_M2005, x0=[-11.0] * nb_points, 
                     method="Powell", args=("S6", False))
    preds_S6 = forward(res_S6.x, data_, theta_mode=False)
    dQFM_S6 = res_S6.x - QFM_values
    print(f"      dQFM (IPA-S6): Median {np.median(dQFM_S6):.1f} | "
          f"Min {np.min(dQFM_S6):.1f} | Max {np.max(dQFM_S6):.1f}")
    
    # =========================================================================
    # STEP 6: Compare with J2010 model
    # =========================================================================
    print("\n[6/8] Comparing with J2010 model...")
    res_J2010 = minimize(lambda x: np.sum((Ju2010(x) - data_.S6)**2),
                        x0=dQFM_S6, method="Powell")
    dFMQ_J2010 = res_J2010.x
    
    # =========================================================================
    # STEP 7: Generate figures
    # =========================================================================
    print("\n[7/8] Generating figures...")
    
    # Figure 1: Temperature vs SiO2
    print("      Creating Temperature_SiO2.pdf...")
    plt.figure(figsize=(3.22, 3.22))
    for i in range(len(data_)):
        plt.errorbar(data_.sio2[i], data_.T_start[i], xerr=0.02, yerr=44.0, 
                    fmt='s', color=data_.loc[i, ["C", "M", "Y", "K"]].values[:], 
                    ecolor=data_.loc[i, ["C", "M", "Y", "K"]].values[:])
    plt.ylabel('Calculated T, °C')
    plt.xlabel(r"SiO$_2$, wt%")
    plt.tight_layout()
    plt.savefig("../figures/Modelling/Temperature_SiO2.pdf")
    plt.close()
    
    # Figure 2: Model comparison (6 panels)
    print("      Creating models.pdf...")
    fig, axes = plt.subplots(3, 2, figsize=(6.44, 7))
    colors = data_.loc[:, ["C", "M", "Y", "K"]].values
    
    plot_model(ax=axes[0, 0], x=data_.Fe3, y=preds_Fe3[0], error=0.02, colors=colors,
              lim=0.55, xlabel=r"Measured $Fe^{3+}/Fe^\mathregular{TOT}$",
              ylabel=r"Calculated $Fe^{3+}/Fe^\mathregular{TOT}$",
              title="MODEL 1:\nadjust $fO_2$ to fit $Fe^{3+}/Fe^\mathregular{TOT}$",
              annotation="(a)")
    
    plot_model(ax=axes[0, 1], x=data_.Fe3, y=preds_S6[0], error=0.02, colors=colors,
              lim=0.55, xlabel=r"Measured $Fe^{3+}/Fe^\mathregular{TOT}$",
              ylabel=r"Calculated $Fe^{3+}/Fe^\mathregular{TOT}$",
              title="MODEL 2:\nadjust $fO_2$ to fit $S^{6+}/S^\mathregular{TOT}$",
              annotation="(b)")
    
    plot_model(ax=axes[1, 0], x=data_.S6, y=preds_Fe3[1], error=0.02, colors=colors,
              lim=1.0, xlabel=r"Measured $S^{6+}/S^\mathregular{TOT}$",
              ylabel=r"Calculated $S^{6+}/S^\mathregular{TOT}$", annotation="(c)")
    
    plot_model(ax=axes[1, 1], x=data_.S6, y=preds_S6[1], error=0.02, colors=colors,
              lim=1.0, xlabel=r"Measured $S^{6+}/S^\mathregular{TOT}$",
              ylabel=r"Calculated $S^{6+}/S^\mathregular{TOT}$", annotation="(d)")
    
    plot_dqfm(ax=axes[2, 0], x=data_.Fe3, y=dQFM_Fe3, error=0.02, colors=colors,
             annotation="(e)", ylim=(-1.0, 3.5))
    
    plot_dqfm(ax=axes[2, 1], x=data_.Fe3, y=dQFM_S6, error=0.02, colors=colors,
             annotation="(f)", ylim=(-1.0, 3.5))
    
    fig.tight_layout()
    fig.savefig('../figures/Modelling/models.pdf')
    plt.close()
    
    # Figure 3: dQFM comparison for Fe3+
    print("      Creating dQFM_Fe3.pdf...")
    fig, ax = plt.subplots(1, 1, figsize=(3.22, 3.22))
    plot_dqfm(ax=ax, x=data_.Fe3, y=dQFM_Fe3, error=0.02, colors=colors, 
             marker='s', markersize=7)
    plot_dqfm(ax=ax, x=data_.Fe3, y=dQFM_B2018, error=0.02, colors=colors, 
             marker='o', markersize=7)
    plot_dqfm(ax=ax, x=data_.Fe3, y=dQFM_KC, error=0.02, colors=colors, 
             marker='d', markersize=7)
    plt.plot([], [], 's', mec='black', mfc="none", label='IPA')
    plt.plot([], [], 'o', mec='black', mfc="none", label='B2018')
    plt.plot([], [], 'd', mec='black', mfc="none", label='KC1991')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("../figures/Modelling/dQFM_Fe3.pdf")
    plt.close()
    
    # Figure 4: dQFM comparison for S6+
    print("      Creating dQFM_S6.pdf...")
    fig, ax = plt.subplots(1, 1, figsize=(3.22, 3.22))
    mask_S6_meas = data_.S6_meas == True
    plot_dqfm(ax=ax, x=data_.S6[mask_S6_meas], y=dQFM_S6[mask_S6_meas], 
             error=0.02, colors=colors, marker='s', markersize=7,
             xlabel=r"Measured $S^{6+}/S^\mathregular{TOT}$")
    plot_dqfm(ax=ax, x=data_.S6[mask_S6_meas], y=dFMQ_J2010[mask_S6_meas], 
             error=0.02, colors=colors, marker='p', markersize=7,
             xlabel=r"Measured $S^{6+}/S^\mathregular{TOT}$")
    plot_dqfm(ax=ax, x=data_.S6[mask_S6_meas], 
             y=data_.dFMQ_Boulliung2023[mask_S6_meas], 
             error=0.02, colors=colors, marker='<', markersize=7,
             xlabel=r"Measured $S^{6+}/S^\mathregular{TOT}$")
    plt.plot([], [], 's', mec='black', mfc="none", label='IPA')
    plt.plot([], [], 'p', mec='black', mfc="none", label='J2010')
    plt.plot([], [], '<', mec='black', mfc="none", label='BW2023')
    plt.legend(loc='lower right')
    plt.xlim(0, 0.2)
    plt.ylim(-1.0, 1.0)
    plt.tight_layout()
    plt.savefig("../figures/Modelling/dQFM_S6.pdf")
    plt.close()
    
    # Figure 5: dFMQ comparison - all methods
    print("      Creating dFMQ_allmethods.pdf...")
    
    # Load B2025 MELTS-OSaS data
    data_Bell2025 = pd.read_excel("../results/Results_synthese.xlsx", 
                                  sheet_name="Bell_2025")
    
    # Merge B2025 data with main data based on Sample name
    data_merged = data_.merge(data_Bell2025, left_on='abbrev', right_on='Sample', how='left')
    
    # Calculate dFMQ for B2025 (MELTS-OSaS)
    dFMQ_Bell2025 = data_merged['dFMQ'].values
    print("      Merged MELTS-OSaS data from B2025")
    print("Values are:")
    print(f"  dFMQ (MELTS-OSaS): Median {np.nanmedian(dFMQ_Bell2025):.1f} | "
          f"Min {np.nanmin(dFMQ_Bell2025):.1f} | Max {np.nanmax(dFMQ_Bell2025):.1f}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.22))
    X_axis_ = data_.sio2
    
    # Plot legend entries (empty plots for legend)
    plt.plot([], [], "o", mec="k", mfc="k", label="MELTS-OSaS")
    plt.plot([], [], "D", mfc="none", mec="k", label="Fe redox (IPA, -0.5)")
    plt.plot([], [], "<", color="k", label="S redox (BW2023)")
    
    # Plot each point individually with CMYK colors
    for i in range(len(data_)):
        point_color = data_.loc[i, ["C", "M", "Y", "K"]].values[:]
        
        # MELTS-OSaS (B2025)
        if not np.isnan(dFMQ_Bell2025[i]):
            plt.plot(X_axis_[i], dFMQ_Bell2025[i], "o", mec="k", color=point_color)
        
        # Fe redox state (IPA model) - shifted by -0.5
        plt.plot(X_axis_[i], dQFM_Fe3[i] - 0.5, "D", mfc="none", color=point_color)
        
        # S redox state (BW2023) - only for samples with S6 measurements
        if data_.S6_meas[i] == True and not np.isnan(data_.dFMQ_Boulliung2023[i]):
            plt.plot(X_axis_[i], data_.dFMQ_Boulliung2023[i], "<", color=point_color)
    
    plt.xlabel("SiO$_2$, wt%")
    plt.ylabel("ΔFMQ")
    plt.ylim(-1.0, 3.0)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("../figures/Modelling/dFMQ_allmethods.pdf")
    plt.close()
    
    # =========================================================================
    # STEP 8: Save results
    # =========================================================================
    print("\n[8/8] Saving results...")
    dic_out = {"sample": data_.loc[:, "Produit"], 
               "dQFM_Fe_M2005": dQFM_Fe3,
               "dQFM_Fe_KC1991": dQFM_KC,
               "dQFM_Fe_B2018": dQFM_B2018,
               "dQFM_S_M2005": dQFM_S6,
               "dQFM_S_J2010": dFMQ_J2010,
               "dQFM_S_BW2023": data_.dFMQ_Boulliung2023}
    df_out = pd.DataFrame(dic_out)
    df_out.to_csv("../results/dQFM_models.csv", 
                 index=False)
    print("      Saved dQFM_models.csv")
    
    # Save legacy format for compatibility
    df_legacy = pd.DataFrame({"sample": data_.loc[:, "Produit"], "dFMQ": dQFM_Fe3})
    df_legacy.to_csv("../results/modelling/dQFM_Moretti2005_on_Fe3_adjustment.csv", 
                    index=False)
    print("      Saved dQFM_Moretti2005_on_Fe3_adjustment.csv")
    
    print("\n" + "="*80)
    print("Modelling complete!")
    print("="*80)
    print("\nOutputs generated:")
    print("  - figures/Modelling/Temperature_SiO2.pdf")
    print("  - figures/Modelling/models.pdf")
    print("  - figures/Modelling/dQFM_Fe3.pdf")
    print("  - figures/Modelling/dQFM_S6.pdf")
    print("  - figures/Modelling/dFMQ_allmethods.pdf")
    print("  - results/modelling/dQFM_Moretti2005_on_Fe3_adjustment.csv")
    print("="*80)


if __name__ == "__main__":
    main()
