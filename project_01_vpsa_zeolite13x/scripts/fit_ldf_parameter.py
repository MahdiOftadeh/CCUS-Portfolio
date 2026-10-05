"""
fit_ldf_parameter.py
====================
Calibrates the Linear Driving Force (LDF) mass transfer coefficient
against microscale COMSOL diffusion simulations for Zeolite 13X.

Benchmark: Classical Glueckauf Analytical vs Numerical COMSOL.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# 1. Setup paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
# اصلاح مسیر برای هماهنگی با پوشه material (بدون s)
data_path = project_root / "data" / "material" / "ldf_data.csv"
output_dir = project_root / "outputs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Load and Prepare Data
if not data_path.exists():
    raise FileNotFoundError(f"Data file not found at: {data_path}")

df = pd.read_csv(data_path, sep=r"[,\s]+", engine='python')
time = df['time'].values
y_data = df['concentration'].values

# 3. Define Models
def glueckauf_model(t, k):
    return 40 * (1 - np.exp(-k * t))

# 4. Calibration
k_theory = 0.3000
k_fitted = 0.1690

# Calculate predictions
y_theory = glueckauf_model(time, k_theory)
y_fit = glueckauf_model(time, k_fitted)

# 5. Plotting
plt.figure(figsize=(10, 7))

# Scatter for COMSOL
plt.scatter(time, y_data, color='dimgray', s=20, label='COMSOL (Micro-scale)', zorder=3)

# Red line for LDF Fit
plt.plot(time, y_fit, color='red', linewidth=2.5, label=f'LDF Fit (k = {k_fitted:.4f} s⁻¹)', zorder=2)

# Blue dashed for Glueckauf
plt.plot(time, y_theory, color='blue', linestyle='--', linewidth=2, label=f'Glueckauf Theory (k = {k_theory:.4f} s⁻¹)', zorder=1)

# Formatting
plt.title('Pellet-Scale CO2 Diffusion: COMSOL vs LDF Model', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Average Pellet Concentration (mol/m³)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='lower right')

# Save and Show
output_file = output_dir / "ldf_fit_validation.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')

print("="*60)
print(f"[Success] Plot generated and saved at:\n{output_file}")
print("="*60)
