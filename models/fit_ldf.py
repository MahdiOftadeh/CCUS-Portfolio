import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. File Path Setup
DATA_PATH = os.path.join("data", "ldf_data.csv")
OUTPUT_PLOT = os.path.join("data", "ldf_fit_validation.png")

# 2. Physical Model Parameters
c_bulk = 40.0         # mol/m^3
R_p = 1.0e-3          # m (1 mm)
D_eff = 2.0e-8        # m^2/s

# Theoretical Glueckauf model (LDF approximation for spherical particle)
# k_theory = 15 * D_eff / R_p^2
k_ldf_theory = 15.0 * D_eff / (R_p**2)

# 3. Read COMSOL output data (COMSOL usually includes '%' for comment lines)
print(f"Reading data from {DATA_PATH}...")
try:
    # Use comment="%" to skip COMSOL header lines if present
    df = pd.read_csv(DATA_PATH, comment="%", sep=None, engine="python", header=None)
    t_data = df.iloc[:, 0].to_numpy()
    c_avg_data = df.iloc[:, 1].to_numpy()
except Exception as e:
    print(f"Error loading CSV: {e}")
    # Fallback if no header comments
    df = pd.read_csv(DATA_PATH)
    t_data = df.iloc[:, 0].to_numpy()
    c_avg_data = df.iloc[:, 1].to_numpy()

# 4. LDF Mathematical Model for Fitting: c_avg(t) = c_bulk * (1 - exp(-k * t))
def ldf_model(t, k):
    return c_bulk * (1.0 - np.exp(-k * t))

# 5. Curve Fitting
popt, pcov = curve_fit(ldf_model, t_data, c_avg_data, p0=[k_ldf_theory])
k_ldf_fitted = popt[0]
r2_score = 1.0 - np.sum((c_avg_data - ldf_model(t_data, k_ldf_fitted))**2) / np.sum((c_avg_data - np.mean(c_avg_data))**2)

print("\n" + "="*45)
print(f"Theoretical k_LDF (Glueckauf): {k_ldf_theory:.4f} s^-1")
print(f"Fitted k_LDF from COMSOL:      {k_ldf_fitted:.4f} s^-1")
print(f"Deviation:                      {abs(k_ldf_fitted - k_ldf_theory)/k_ldf_theory * 100:.2f} %")
print(f"R-squared of Fit:               {r2_score:.5f}")
print("="*45 + "\n")

# 6. Plot and Save Comparison Figure
plt.figure(figsize=(7, 5), dpi=300)
plt.plot(t_data, c_avg_data, 'k.', label='COMSOL 1D-Axisym (Actual Diffusion)', markersize=4, alpha=0.7)
plt.plot(t_data, ldf_model(t_data, k_ldf_fitted), 'r-', linewidth=2, label=f'LDF Fit (k = {k_ldf_fitted:.4f} 1/s)')
plt.plot(t_data, ldf_model(t_data, k_ldf_theory), 'b--', linewidth=1.5, label=f'Theoretical Glueckauf (k = {k_ldf_theory:.4f} 1/s)')

plt.title('COMSOL Micro-Scale Diffusion vs. LDF Model', fontsize=12, fontweight='bold')
plt.xlabel('Time (s)', fontsize=11)
plt.ylabel(r'Average Pellet Concentration $\bar{c}$ ($mol/m^3$)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(frameon=True)
plt.tight_layout()

plt.savefig(OUTPUT_PLOT)
print(f"Plot saved successfully at: {OUTPUT_PLOT}")
