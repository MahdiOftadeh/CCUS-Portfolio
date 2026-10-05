"""
fit_ldf_parameter.py
====================
Calibrates the Linear Driving Force (LDF) mass transfer coefficient
against microscale COMSOL diffusion simulations for Zeolite 13X.

Benchmark: Classical Glueckauf Analytical vs Numerical COMSOL.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def find_data_file(project_root: Path) -> Path:
    """Searches candidates to locate the dataset robustly."""
    candidates = [
        project_root / "data" / "materials" / "ldf_data.csv",
        project_root / "data" / "material" / "ldf_data.csv", # اضافه شده برای پشتیبانی از پوشه بدون s
        project_root / "data" / "ldf_data.csv",
        project_root / "ldf_data.csv",
        project_root.parent / "ldf_data.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
            
    # Fallback: scan for any ldf file inside project
    found = list(project_root.rglob("*ldf_data*"))
    for f in found:
        if f.is_file():
            return f

    raise FileNotFoundError(f"[Error] Could not find 'ldf_data' inside: {project_root}")


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    output_dir = project_root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_plot = output_dir / "ldf_fit_validation.png"

    print("=" * 60)
    print("VPSA Microscale Kinetic Calibration: Glueckauf vs COMSOL")
    print("=" * 60)

    data_path = find_data_file(project_root)
    print(f"Data file detected: {data_path.name}")
    print(f"Full Path: {data_path}")

    # Load COMSOL simulation data (skips comments starting with '%', handles whitespace/comma)
    df = pd.read_csv(data_path, comment="%", sep=r"[,\s]+", header=None, engine="python")
    t_data = df.iloc[:, 0].to_numpy(dtype=float)
    c_avg_data = df.iloc[:, 1].to_numpy(dtype=float)

    # Physical parameters for Zeolite 13X
    c_bulk = 40.0       # mol/m^3
    R_p = 0.001         # m (1 mm)
    D_eff = 2.0e-8      # m^2/s

    # Glueckauf theoretical parameter (1955)
    k_ldf_theory = 15.0 * D_eff / (R_p ** 2)

    def ldf_model(t, k):
        return c_bulk * (1.0 - np.exp(-k * t))

    # Curve fitting
    popt, _ = curve_fit(ldf_model, t_data, c_avg_data, p0=[k_ldf_theory])
    k_ldf_fitted = float(popt[0])

    print("-" * 60)
    print(f"Theoretical k_LDF (Glueckauf Analytical): {k_ldf_theory:.4f} s^-1")
    print(f"Fitted k_LDF (COMSOL Microscale):         {k_ldf_fitted:.4f} s^-1")
    print("-" * 60)

    # Plotting: Updated to match visual requirements
    plt.figure(figsize=(10, 7))
    
    # Scatter for COMSOL
    plt.scatter(t_data, c_avg_data, color='dimgray', s=20, label='COMSOL (Micro-scale)', zorder=3)

    # Red line for LDF Fit
    plt.plot(t_data, ldf_model(t_data, k_ldf_fitted), color='red', linewidth=2.5, 
             label=f'LDF Fit (k = {k_ldf_fitted:.4f} s⁻¹)', zorder=2)

    # Blue dashed for Glueckauf
    plt.plot(t_data, ldf_model(t_data, k_ldf_theory), color='blue', linestyle='--', linewidth=2, 
             label=f'Glueckauf Theory (k = {k_ldf_theory:.4f} s⁻¹)', zorder=1)

    # Styling labels and title
    plt.title('Pellet-Scale CO2 Diffusion: COMSOL vs LDF Model', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Average Pellet Concentration (mol/m³)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()
    
    print(f"[Success] Plot generated at:\n{output_plot}\n")


if __name__ == "__main__":
    main()
