#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive XAS Analysis for Mayotte Publication
===================================================

This script combines all analyses from four Jupyter notebooks into a single
publication-ready Python script. It performs:

1. Iron (Fe) K-edge XANES analysis of glass standards and samples
2. Sulfur (S) K-edge XANES analysis of glasses
3. Generation of spectral figures for publication
4. Iron beam damage analysis

All figures are saved in the ./figures directory with subdirectories for
organization (./figures/Iron/, ./figures/Sulfur/).

Author: Charles Le Losq
Date: Last Update November 2025
Python version: 3.10
Required packages: numpy, scipy, pandas, matplotlib, rampy, larch, tqdm, 
                   sklearn, uncertainties
   
"""

import numpy as np
from scipy.optimize import curve_fit

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import r2_score
import uncertainties
from uncertainties import unumpy

# Import custom functions from src module
from functions import (
    get_Fe_spectrum, get_S_spectrum, glass_treatment, S_treatment,
    C_to_Fe3, calculate_centroid, calculate_centroid_noFeadjust, provide_redox
)

# Set matplotlib backend for non-interactive plotting
matplotlib.use('Agg')
plt.ioff()

print("="*70)
print("XAS Analysis for Mayotte Publication".center(70))
print("="*70)
print()


# =============================================================================
# SECTION 1: IRON STANDARDS ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("SECTION 1: Iron Standards Analysis".center(70))
print("="*70)

print("\n[1.1] Loading Fe standard glass data...")
dataliste_std = pd.read_excel("../tables/liste.xlsx", sheet_name="Fe_glasses_standards")
print(f"Loaded {len(dataliste_std)} standard samples")

print("\n[1.2] Processing Fe standards...")
for count, i in enumerate(dataliste_std.loc[:, "filename"]):
    print(f"  Processing {i}...")
    x, y, f1, f2, Fe3_area_ratio, centroid, centroid2 = glass_treatment(i)
    
    # Save results to dataframe
    dataliste_std.loc[count, "Fe2_position"] = round(f1, 3)
    dataliste_std.loc[count, "Fe3_position"] = round(f2, 3)
    dataliste_std.loc[count, "Fe3_area_ratio"] = round(Fe3_area_ratio, 3)
    dataliste_std.loc[count, "centroid_fit"] = round(centroid, 3)
    dataliste_std.loc[count, "centroid_trapz"] = centroid2
    dataliste_std.loc[count, "redox_F2017"] = round(
        C_to_Fe3(dataliste_std.centroid_fit[count], 7112.31, method="F2017")
    )

print("\n[1.3] Generating Fe standards calibration figure...")

# Import literature calibration data
data_literature = pd.read_excel("../tables/liste.xlsx", sheet_name="Centroid_Data")

# Normalize centroids to a common reference frame
data_literature["B2018_centroid_norm"] = (
    (data_literature.loc[:, "Centroid_MORB"] - 7112.958) / 
    (7114.566 - 7112.958) * 1.8
)
data_literature["G2011_centroid_norm"] = (
    (data_literature.loc[:, "Centroid_phono"] - 7112.9) / 
    (7114.5 - 7112.9) * 1.8
)
data_literature["W2005_centroid_norm"] = (
    (data_literature.loc[:, "Centroid_W2005"] - 7112.0) / 
    (7113.5 - 7112.0) * 1.8
)
data_literature["MRC2015_centroid_norm"] = (
    (data_literature.loc[:, "Centroid_MRC2015"] - 7112.9) / 
    (7114.5 - 7112.9) * 1.8
)
dataliste_std["Centroid_norm"] = (
    (dataliste_std.loc[:, "centroid_fit"] - 7112.32) / 
    (7114.13 - 7112.32) * 1.8
)

# Prepare combined literature data
Litterature_centroids = np.concatenate((
    data_literature.MRC2015_centroid_norm.dropna().copy().values,
    data_literature.G2011_centroid_norm.dropna().copy().values,
    data_literature.W2005_centroid_norm.dropna().copy().values,
    data_literature.B2018_centroid_norm.dropna().copy().values
))

Litterature_redox = np.concatenate((
    data_literature.Redox_MRC2015.dropna().copy().values,
    data_literature.Redox_phono.dropna().copy().values,
    data_literature.Redox_W2005.dropna().copy().values,
    data_literature.Redox_MORB.dropna().copy().values
))

# Fit calibration curve
popt, pcov = curve_fit(
    calculate_centroid_noFeadjust, 
    dataliste_std.redox, 
    dataliste_std.Centroid_norm
)
print(f"  Best calibration coefficient: {popt[0]:.4f}")

# Generate calibration plot
fig, ax = plt.subplots(figsize=(3.22, 3.22))

# Plot standards
ax.errorbar(
    dataliste_std.Centroid_norm, dataliste_std.redox, yerr=0.02,
    marker="^", markersize=10, linestyle="none", color="purple",
    label="PyNa standards", zorder=-1
)

# Plot calibration curve
x_redox = np.arange(0, 1, 0.01)
evolution = calculate_centroid(x_redox, f2=0.0, f3=7114.13 - 7112.32)
ax.plot(evolution, x_redox, "--", color="purple", linewidth=2,
        label="eq. 1, this study", zorder=1000)

# Plot literature data
ax.errorbar(
    data_literature.MRC2015_centroid_norm, data_literature.Redox_MRC2015,
    yerr=0.02, marker="D", linestyle="none", label="Phonolites C2015"
)
ax.errorbar(
    data_literature.G2011_centroid_norm, data_literature.Redox_phono,
    yerr=0.02, marker="^", linestyle="none", label="Phonolites G2011"
)
ax.errorbar(
    data_literature.W2005_centroid_norm, data_literature.Redox_W2005,
    yerr=0.04, marker=">", linestyle="none", label="Basalts W2005"
)

ax.legend(fontsize=8, loc=2)
ax.set_xlabel("Pre-edge centroid, eV")
ax.set_ylabel(r"Fe$^{3+}$/Fe$^\mathregular{TOT}$")
plt.tight_layout()
plt.savefig("../figures/calibration.pdf")
plt.close()

# Calculate RMSE for validation
Litterature_redox_calc = np.array([
    provide_redox(Litterature_centroids[i] + 7112.32)[0]
    for i in range(len(Litterature_centroids))
])
print(f"  RMSE for literature data: {rmse(Litterature_redox_calc, Litterature_redox):.4f}")

PyNa_redox = np.array([
    provide_redox(dataliste_std.centroid_fit[i])[0]
    for i in range(len(dataliste_std.redox))
])
print(f"  RMSE for PyNa standards: {rmse(PyNa_redox, dataliste_std.redox):.4f}")
print(f"  R² score: {r2_score(Litterature_redox_calc, Litterature_redox):.4f}")


# =============================================================================
# SECTION 2: IRON GLASS SAMPLES ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("SECTION 2: Iron Glass Samples Analysis".center(70))
print("="*70)

print("\n[2.1] Loading Fe glass sample data...")
dataliste = pd.read_excel("../tables/liste.xlsx", sheet_name="Fe_glasses")
print(f"Loaded {len(dataliste)} samples")

print("\n[2.2] Processing Fe glass samples...")
for count, i in enumerate(dataliste.loc[:, "filename"]):
    print(f"  Processing sample {count+1}/{len(dataliste)}: {i}")
    _, _, f1, f2, Fe3_area_ratio, centroid, centroid2 = glass_treatment(i)
    
    dataliste.loc[count, "f1"] = round(f1, 3)
    dataliste.loc[count, "f2"] = round(f2, 3)
    dataliste.loc[count, "Fe3_area_ratio"] = round(Fe3_area_ratio, 3)
    dataliste.loc[count, "centroid_fit"] = round(centroid, 3)
    dataliste.loc[count, "centroid_trapz"] = centroid2
    dataliste.loc[count, "redox_F2017"] = round(
        C_to_Fe3(dataliste.loc[count, "centroid_fit"], f1, method="F2017")
    )
    dataliste.loc[count, "redox_m"] = provide_redox(
        dataliste.loc[count, "centroid_fit"]
    )[0]

print("\n[2.3] Saving Fe glass results...")
dataliste_selected = dataliste[dataliste.check_good != 0]
dataliste_selected.to_csv("../results/Fe_res.csv")
print(f"  Saved {len(dataliste_selected)} good quality samples to ../results/Fe_res.csv")# Calculate statistics by sample
print("\n[2.4] Sample statistics (mean ± std):")
dataliste_stats = dataliste_selected.drop("filename", axis=1).copy()
sample_means = dataliste_stats.groupby("sample").mean()
sample_stds = dataliste_stats.groupby("sample").std()
for sample in sample_means.index:
    redox_mean = sample_means.loc[sample, "redox_m"]
    redox_std = sample_stds.loc[sample, "redox_m"]
    print(f"  {sample}: Fe³⁺/Fetot = {redox_mean:.3f} ± {redox_std:.3f}")


# =============================================================================
# SECTION 3: SPECTRAL FIGURES FOR PUBLICATION
# =============================================================================
print("\n" + "="*70)
print("SECTION 3: Spectral Figures for Publication".center(70))
print("="*70)

# --- Figure 3.1: Fe Reference Spectra ---
print("\n[3.1] Generating Fe reference spectra figure...")
dataliste_std_fig = pd.read_excel("../tables/liste.xlsx", sheet_name="Fe_glasses_standards_forfigure")
colors_std = plt.cm.rainbow(np.linspace(0, 1, len(dataliste_std_fig)))

fig = plt.figure(figsize=(3.22, 4.5))
ax = plt.subplot()

inset_ax = ax.inset_axes(
    [0.3, 0.1, 0.6, 0.4],
    xlim=[7110, 7116], ylim=[0.0, 0.32],
    yticklabels=[]
)

ax.set_xlim(7100, 7310)

shift_ = 0.0
for count, i in enumerate(dataliste_std_fig.loc[:, "filename"]):
    x_all, y_all, x, y = get_Fe_spectrum(i)
    
    ax.plot(x_all, y_all + shift_, color=colors_std[count])
    inset_ax.plot(x_all, y_all + shift_, color=colors_std[count])
    
    ax.annotate(
        dataliste_std_fig.loc[count, "sample"],
        xy=(7250, shift_ + 1.0), xycoords="data",
        va="center", fontsize=8
    )
    shift_ += 0.05

ax.set_xlabel("Energy, eV")
ax.set_ylabel("Normalized absorption")

position_fe2 = 7112.3
inset_ax.annotate("Fe$^{2+}$", xy=(position_fe2, 0.28), fontsize=10, ha="center")
inset_ax.annotate("Fe$^{3+}$", xy=(position_fe2 + 1.65, 0.26), fontsize=10, ha="center")
inset_ax.plot([position_fe2, position_fe2], [0.0, 0.27], "--", c="grey")
inset_ax.plot([position_fe2 + 1.85, position_fe2 + 1.85], [0.0, 0.25], "--", c="grey")

ax.indicate_inset_zoom(inset_ax, edgecolor="black")
plt.tight_layout()
plt.savefig("../figures/Spectra_refs.pdf")
plt.close()
print("  Saved: ../figures/Spectra_refs.pdf")# --- Figure 3.2: Fe Sample Spectra ---
print("\n[3.2] Generating Fe sample spectra figure...")
dataliste_fig = pd.read_excel("../tables/liste.xlsx", sheet_name="Glasses_forfigure")
colors = plt.cm.rainbow(np.linspace(0, 1, len(dataliste_fig)))

fig = plt.figure(figsize=(3.22, 5))
ax1 = plt.subplot()
ax2 = ax1.inset_axes(
    [0.3, 0.06, 0.6, 0.30],
    xlim=[7110, 7116], ylim=[0.0, 1.35],
    yticklabels=[]
)

shift_1 = 0.0
shift_2 = 0.0

for count, i in enumerate(dataliste_fig.loc[:, "Fe_filename"]):
    x_all, y_all, x, y = get_Fe_spectrum(i)
    
    ax1.plot(x_all, y_all + shift_1, color=colors[count])
    ax1.annotate(
        dataliste_fig.loc[count, "abbrev"],
        xy=(7251, shift_1 + 1), xycoords="data",
        color="k", va="center", fontsize=8
    )
    
    ax2.plot(x_all, y_all + shift_2, color=colors[count])
    
    shift_1 += 0.1
    shift_2 += 0.01

ax1.set_xlabel("Energy, eV")
ax1.set_ylabel("Normalized absorption")

ax2.annotate("Fe$^{2+}$", xy=(position_fe2, 0.202), fontsize=10, ha="center")
ax2.annotate("Fe$^{3+}$", xy=(position_fe2 + 1.9, 0.212), fontsize=10, ha="center")

ax1.set_xlim(7100, 7290)
ax1.set_ylim(-0.05, 2.55)
ax1.indicate_inset_zoom(ax2, edgecolor="black")
ax2.set_ylim(0, 0.25)

plt.tight_layout()
plt.savefig("../figures/Spectra_samples.pdf")
plt.close()
print("  Saved: ../figures/Spectra_samples.pdf")# --- Figure 3.3: S Sample Spectra ---
print("\n[3.3] Generating S sample spectra figure...")

shift_ = 0.
check_type = dataliste_fig.loc[:, 'S_filename'].isnull()

fig = plt.figure(figsize=(3.22, 5))
for count, i in enumerate(dataliste_fig.loc[:, "S_filename"]):
    if check_type[count] == False:
        x_all, y_all = get_S_spectrum(i)
        
        plt.annotate(
            dataliste_fig.loc[count, "abbrev"],
            xy=(2507, shift_ + 1.2), xycoords="data", fontsize=8
        )
        
        plt.plot(x_all, y_all + shift_, color=colors[count])
        shift_ += 0.7

plt.xlim(2455, 2520)

plt.annotate("sulfides", xy=(2465, 6.9), color="k", ha="center")
plt.plot([2460., 2471.0], [6.8, 6.8], "k--")

plt.annotate("2-", xy=(2475.9, 7.35), color="grey", ha="center")
plt.plot([2476, 2476], [1.2, 7.3], "--", color="grey")

plt.plot([2477, 2477], [0.5, 7.0], "--", color="grey")
plt.annotate("4+", xy=(2477., 0.3), color="grey", ha="center")

plt.text(2481.7, 0.8, "6+", fontsize=10, ha="center", va="center", color="grey")
plt.plot([2481.7, 2481.7], [1.0, 7.5], "--", color="grey")

plt.xlabel("Energy, eV")
plt.ylabel("Normalized absorption")

plt.tight_layout()
plt.savefig("../figures/Spectra_S_samples.pdf")
plt.close()
print("  Saved: ../figures/Spectra_S_samples.pdf")# Save color scheme for consistency
print("\n[3.4] Saving color scheme...")
colors_df = pd.DataFrame(colors, columns=["C", "M", "Y", "K"])
colors_df["name"] = dataliste_fig["sample"]
colors_df.to_csv("../results/colors.csv")
print("  Saved: ../results/colors.csv")

# =============================================================================
# SECTION 4: SULFUR ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("SECTION 4: Sulfur Analysis".center(70))
print("="*70)

print("\n[4.1] Loading S glass data...")
dataliste_S = pd.read_excel("../tables/liste.xlsx", sheet_name="S_glasses")
dataliste_S["area_S6"] = 0.
dataliste_S["I_S6"] = 0.
dataliste_S["redox_S6"] = 0.
print(f"Loaded {len(dataliste_S)} S glass samples")

print("\n[4.2] Processing S glass samples...")
data_ech_ = []

for count in tqdm(range(len(dataliste_S.filename)), desc="Processing S spectra"):
    if dataliste_S.loc[count, "type"] != -1:
        try:
            res_ = S_treatment(
                nom=dataliste_S.loc[count, "filename"],
                spectrum_type=dataliste_S.loc[count, "type"],
                output_flat_dat=True
            )
            
            dataliste_S.loc[count, "area_S6"] = res_[0]
            dataliste_S.loc[count, "I_S6"] = res_[1]
            dataliste_S.loc[count, "redox_S6_L2021"] = res_[2]
            dataliste_S.loc[count, "redox_S6_J2010"] = res_[3]
            dataliste_S.loc[count, "redox_S6_LL2023"] = res_[4]
            data_ech_.append(res_[5])
        except Exception as e:
            print(f"  Warning: Failed to process {dataliste_S.loc[count, 'filename']}: {e}")
            dataliste_S.loc[count, "area_S6"] = np.nan
            dataliste_S.loc[count, "I_S6"] = np.nan
            dataliste_S.loc[count, "redox_S6_L2021"] = np.nan
            dataliste_S.loc[count, "redox_S6_J2010"] = np.nan
            dataliste_S.loc[count, "redox_S6_LL2023"] = np.nan
            data_ech_.append(np.nan)

print("\n[4.3] Saving S results...")
dataliste_S.to_csv("../results/S_res.csv")
print("  Saved: ../results/S_res.csv")# Print statistics if Results_synthese.xlsx exists

try:
    dataliste_S_stats = pd.read_excel("../results/Results_synthese.xlsx", sheet_name="S_")
    products = dataliste_S_stats.loc[:, "sample"].unique()
    
    print("\n[4.4] Sample statistics (median ± std):")
    for i in products:
        mean_ = np.median(
            dataliste_S_stats.loc[
                (dataliste_S_stats["sample"] == i) & 
                (dataliste_S_stats["redox_good"] == 1),
                "redox_S6_J2010"
            ].dropna()
        )
        std_ = np.std(
            dataliste_S_stats.loc[
                (dataliste_S_stats["sample"] == i) & 
                (dataliste_S_stats["redox_good"] == 1),
                "redox_S6_J2010"
            ].dropna()
        )
        print(f"  {i}: S⁶⁺/Stot = {mean_:.2f} ± {std_:.2f}")
except FileNotFoundError:
    print("\n  Note: Results_synthese.xlsx not found, skipping statistics")


# =============================================================================
# SECTION 5: BEAM DAMAGE ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("SECTION 5: Beam Damage Analysis".center(70))
print("="*70)

# --- Fe Beam Damage ---
print("\n[5.1] Analyzing Fe beam damage...")
dataliste_damage = pd.read_excel("../tables/liste.xlsx", sheet_name="Fe_damage")

def measure_Fe_damage(liste):
    """Function to measure Fe beam damage in spectral time series"""
    I_7112 = np.zeros(len(liste))
    I_7114 = np.zeros(len(liste))
    
    for count, i in enumerate(liste):
        sp = np.genfromtxt("../xas/iron/" + i, encoding='cp1252')
        x = sp[:, 0]
        y = sp[:, 4] / sp[:, 1]
        
        I_7112[count] = y[(x > 7111.5) & (x < 7112.5)]
        I_7114[count] = y[(x > 7113.5) & (x < 7114.5)]
    
    return I_7112, I_7114

# Extract intensity time series
I_7112_50, I_7114_50 = measure_Fe_damage(
    dataliste_damage.loc[dataliste_damage.loc[:, "test"] == "Test_50", "filename"]
)
I_7112_70, I_7114_70 = measure_Fe_damage(
    dataliste_damage.loc[dataliste_damage.loc[:, "test"] == "Test_70", "filename"]
)
I_7112_100_1, I_7114_100_1 = measure_Fe_damage(
    dataliste_damage.loc[dataliste_damage.loc[:, "test"] == "Test_100-1", "filename"]
)
I_7112_100_2, I_7114_100_2 = measure_Fe_damage(
    dataliste_damage.loc[dataliste_damage.loc[:, "test"] == "Test_100-2", "filename"]
)

# Plot damage evolution
fig, axes = plt.subplots(nrows=2, ncols=1,
                           sharex=True, sharey=False,
                           figsize=(3.22,5))

axes[0].plot(np.arange(1,len(I_7112_50)+1,1)*30, I_7112_50, "s", label="50 %")
axes[0].plot(np.arange(1,len(I_7112_70)+1,1)*30,I_7112_70, "o", label="70 %")
axes[0].plot(np.arange(1,len(I_7112_100_1)+1,1)*30,I_7112_100_1, "d", label="100 %, series 1")
axes[0].plot(np.arange(1,len(I_7112_100_2)+1,1)*30,I_7112_100_2, "^", label="100 %, series 2")
axes[0].legend(loc="best",fontsize=7)
axes[0].annotate("(a) 7112.0 eV", xy=(0.02,0.97), xycoords="axes fraction", ha="left", va="top")
axes[0].set_ylim(1.25,1.55)

axes[1].plot(np.arange(1,len(I_7112_50)+1,1)*30, I_7114_50, "s", label="50 %")
axes[1].plot(np.arange(1,len(I_7112_70)+1,1)*30,I_7114_70, "o", label="70 %")
axes[1].plot(np.arange(1,len(I_7112_100_1)+1,1)*30,I_7114_100_1, "d", label="100 %, series 1")
axes[1].plot(np.arange(1,len(I_7112_100_2)+1,1)*30,I_7114_100_2, "^", label="100 %, series 2")
axes[1].annotate("(b) 7114.0 eV", xy=(0.02,0.97), xycoords="axes fraction", ha="left", va="top")
axes[1].set_ylim(1.3,1.7)

plt.xlabel("Seconds after exposure")
fig.text(0.04, 0.5, 'Normalized absorbance', va='center', rotation='vertical')
plt.tight_layout(pad=2.0)
plt.savefig('../figures/Fe_beam_damage.pdf')
plt.close()
print("  Saved: ../figures/Fe_beam_damage.pdf")

# --- S Beam Damage ---
print("\n[5.2] Analyzing S beam damage...")

try:
    dataliste_S_damage = pd.read_excel("../results/Results_synthese.xlsx", 
                                       sheet_name="S_")
    
    # Logarithmic model for damage evolution
    def log_model(t, a, b, c):
        return a * np.log(b * t + 1.0) + c
    
    def log_model_unumpy(t, a, b, c):
        return a * unumpy.log(b * t + 1.0) + c
    
    def fit_time_profile(t, y):
        """Fit logarithmic model to time series data"""
        params, pcov = curve_fit(
            log_model, t, y,
            p0=[0.01, 0.1, 0.001],
            bounds=[(0, 0., 0.), (np.inf, np.inf, np.inf)],
            maxfev=2000
        )
        
        print(f"    Fitted parameters: a={params[0]:.4f}, b={params[1]:.6f}, c={params[2]:.4f}")
        
        # Calculate uncertainty at t=0
        pars_with_std = uncertainties.correlated_values(params, pcov)
        y_at_0 = log_model_unumpy(0.0, *pars_with_std)
        print(f"    Value at time = 0: y(0) = {y_at_0:.4f}")
        
        return params, pcov
    
    # DR17 time series
    DR17_time_series_x = np.array([55, 55*2, 55*2 + (26*60+20), 
                                     55*2 + 2*(26*60+20), 55*2 + 3*(26*60+20)])
    DR17_time_series_y = np.array([0.17, 0.18, 0.20, 0.21, 0.21])
    
    # DR04 time series
    DR04_time_series_x = np.array([55, 55*2, 55*3, 55*3+26*60+20])
    DR04_time_series_y = np.array([0.02, 0.03, 0.04, 0.05])
    
    print("  Fitting DR17 phonolite:")
    params_DR17, pcov_DR17 = fit_time_profile(DR17_time_series_x, DR17_time_series_y)
    
    print("  Fitting DR04 basanite:")
    params_DR04, pcov_DR04 = fit_time_profile(DR04_time_series_x, DR04_time_series_y)
    
    # Generate fits
    t_fit = np.linspace(0, 5000, 200)
    y_fit_DR17 = log_model(t_fit, *params_DR17)
    y_fit_DR04 = log_model(t_fit, *params_DR04)
    
    # Plot damage evolution
    fig = plt.figure(figsize=(3.22, 3.22))
    
    plt.errorbar(
        DR17_time_series_x, DR17_time_series_y,
        yerr=0.01, marker="o", linestyle="none",
        color=(1.000000, 1.224647e-16, 6.123234e-17),
        label="DR17 phonolite"
    )
    plt.errorbar(
        DR04_time_series_x, DR04_time_series_y,
        yerr=0.01, marker="s", linestyle="none",
        color=(0.668627, 9.651241e-01, 6.075389e-01),
        label="DR04 basanite"
    )
    
    plt.plot(t_fit, y_fit_DR17, '--', color=(1.000000, 1.224647e-16, 6.123234e-17))
    plt.plot(t_fit, y_fit_DR04, '-.', color=(0.668627, 9.651241e-01, 6.075389e-01))
    
    plt.legend(loc=5)
    plt.grid(True)
    plt.xlabel("Seconds after exposure")
    plt.ylabel(r"S$^{6+}$/S$^\mathregular{TOT}$")
    plt.tight_layout()
    plt.savefig("../figures/S_damage.pdf")
    plt.close()
    print("  Saved: ../figures/S_damage.pdf")
    
except FileNotFoundError:
    print("  Note: Results_synthese.xlsx not found, skipping S damage analysis")


# =============================================================================
# COMPLETION
# =============================================================================
print("\n" + "="*70)
print("Analysis Complete!".center(70))
print("="*70)
print("\nAll figures have been saved to the ./figures directory:")
print("  - ./figures/Iron/calibration.pdf")
print("  - ./figures/Iron/[individual sample spectra].pdf")
print("  - ./figures/Sulfur/[individual sample spectra].pdf")
print("  - ./figures/Spectra_refs.pdf")
print("  - ./figures/Spectra_samples.pdf")
print("  - ./figures/Spectra_S_samples.pdf")
print("  - ./figures/Fe_beam_damage.pdf")
print("  - ./figures/S_damage.pdf (if data available)")
print("\nResults have been saved to:")
print("  - ./results/Fe_res.csv")
print("  - ./results/S_res.csv")
print("  - ./results/colors.csv")
print("\n" + "="*70)
