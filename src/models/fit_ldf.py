"""
LDF Parameter Extraction and Validation Script.
Fits the Linear Driving Force (LDF) model to COMSOL micro-scale diffusion data
and compares the extracted mass transfer coefficient with Glueckauf theory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# File paths
DATA_PATH = os.path.join("data", "ldf_data.csv")
OUTPUT_PLOT = os.path.join("data", "ldf_fit_validation.png")

# Particle and operational parameters (COMSOL conditions)
c_bulk = 40.0         # Bulk/surface concentration [mol/m^3]
R_p = 1.0e-3          # Particle radius: 1 mm [m]
D_eff = 2.0e-8        # Effective diffusivity [m^2/s]

# Theoretical mass transfer coefficient from Glueckauf approximation:
# k_LDF = 15 * D_eff / R_p^2
k_ldf_theory = 15.0 * D_eff / (R_p ** 2)

print("Loading COMSOL data from:", DATA_PATH)

# Robust CSV parser capable of handling COMSOL comment headers (%)
try:
    df = pd.read_csv(DATA_PATH, comment="%", sep=None, engine="python", header=None)
    t_data = df.iloc[:, 0].to_numpy(dtype=float)
    c_avg_data = df.iloc[:, 1].to_numpy(dtype=float)
except Exception:
    df = pd.read_csv(DATA_PATH)
    t_data = df.iloc[:, 0].to_numpy(dtype=float)
    c_avg_data = df.iloc[:, 1].to_numpy(dtype=float)

# LDF analytical solution for constant boundary condition:
# c_avg(t) = c_bulk * (1 - exp(-k * t))
def ldf_model(t, k):
    return c_bulk * (1.0 - np.exp(-k * t))

# Non-linear curve fitting using Levenberg-Marquardt algorithm
popt, pcov = curve_fit(ldf_model, t_data, c_avg_data, p0=[k_ldf_theory])
k_ldf_fitted = float(popt[0])

# Statistical metrics: R-squared and absolute deviation
ss_res = np.sum((c_avg_data - ldf_model(t_data, k_ldf_fitted)) ** 2)
ss_tot = np.sum((c_avg_data - np.mean(c_avg_data)) ** 2)
r2_score = float(1.0 - (ss_res / ss_tot))
deviation = float(abs(k_ldf_fitted - k_ldf_theory) / k_ldf_theory * 100.0)

# Display results
print("\n" + "=" * 50)
print("             LDF VALIDATION RESULTS              ")
print("=" * 50)
print(f"Theoretical k_LDF (Glueckauf): {k_ldf_theory:.4f} s^-1")
print(f"Fitted k_LDF from COMSOL:      {k_ldf_fitted:.4f} s^-1")
print(f"Relative Deviation:            {deviation:.2f} %")
print(f"Coefficient of Determ. (R^2):  {r2_score:.5f}")
print("=" * 50 + "\n")

# Visualization: COMSOL points vs LDF fit and theoretical curve
plt.figure(figsize=(7, 5), dpi=300)
plt.plot(t_data, c_avg_data, "ko", markersize=3, alpha=0.6, label="COMSOL (Micro-scale)")
plt.plot(t_data, ldf_model(t_data, k_ldf_fitted), "r-", linewidth=2.0, 
         label=f"LDF Fit (k = {k_ldf_fitted:.4f} s⁻¹)")
plt.plot(t_data, ldf_model(t_data, k_ldf_theory), "b--", linewidth=1.5, 
         label=f"Glueckauf Theory (k = {k_ldf_theory:.4f} s⁻¹)")

plt.title("Pellet-Scale CO₂ Diffusion: COMSOL vs LDF Model", fontsize=12, fontweight="bold")
plt.xlabel("Time (s)", fontsize=11)
plt.ylabel("Average Pellet Concentration (mol/m³)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(frameon=True, fontsize=10)
plt.tight_layout()

# Save publication-quality figure
plt.savefig(OUTPUT_PLOT)
print(f"Validation plot saved to: {OUTPUT_PLOT}")
