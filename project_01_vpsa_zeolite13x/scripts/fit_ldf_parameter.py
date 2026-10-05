"""
fit_ldf_parameter.py
====================
Calibrates the Linear Driving Force (LDF) mass transfer coefficient
against microscale COMSOL diffusion simulations for Zeolite 13X.

This script benchmarks the classical Glueckauf analytical approximation 
against numerical simulation data.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def main():
    # 1. Define project directory structure
    # Assumes script is located at: project_01_vpsa_zeolite13x/scripts/
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    data_path = project_root / "data" / "ldf_data.csv"
    output_dir = project_root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_plot = output_dir / "ldf_fit_validation.png"

    print("=" * 60)
    print("VPSA Microscale Kinetic Calibration: Glueckauf vs COMSOL")
    print("=" * 60)

    # 2. Validate input file existence
    if not data_path.exists():
        raise FileNotFoundError(
            f"\n[Error] Input file not found:\n{data_path}\n"
            f"Please ensure ldf_data.csv is placed in the project/data/ directory."
        )

    # 3. Load COMSOL simulation data (skipping lines starting with '%')
    print(f"Loading COMSOL data from: {data_path.name}")
    df = pd.read_csv(data_path, comment="%", sep=r"\s+", header=None, engine="python")
    
    t_data = df.iloc[:, 0].to_numpy(dtype=float)
    c_avg_data = df.iloc[:, 1].to_numpy(dtype=float)

    # 4. Physical parameters for Zeolite 13X
    c_bulk = 40.0         # Bulk concentration [mol/m³]
    R_p = 0.001           # Pellet radius [m]
    D_eff = 2.0e-8        # Effective diffusivity [m²/s]

    # Theoretical Glueckauf constant (1955)
    k_ldf_theory = 15.0 * D_eff / (R_p ** 2)

    # 5. Define LDF model function
    def ldf_model(t, k):
        return c_bulk * (1.0 - np.exp(-k * t))

    # 6. Perform non-linear fitting
    popt, _ = curve_fit(ldf_model, t_data, c_avg_data, p0=[k_ldf_theory])
    k_ldf_fitted = float(popt[0])

    # 7. Statistical evaluation
    ss_res = np.sum((c_avg_data - ldf_model(t_data, k_ldf_fitted)) ** 2)
    ss_tot = np.sum((c_avg_data - np.mean(c_avg_data)) ** 2)
    r2_score = float(1.0 - (ss_res / ss_tot))
    deviation_pct = float(abs(k_ldf_fitted - k_ldf_theory) / k_ldf_theory * 100.0)

    # 8. Output benchmark results
    print("-" * 60)
    print(f"Theoretical k_LDF (Glueckauf Analytical): {k_ldf_theory:.4f} s^-1")
    print(f"Fitted k_LDF (COMSOL Microscale):         {k_ldf_fitted:.4f} s^-1")
    print(f"Relative Discrepancy (Overestimation):   {deviation_pct:.2f} %")
    print(f"Goodness of Fit (R² Score):               {r2_score:.4f}")
    print("-" * 60)

    # 9. Generate publication-quality plot
    plt.figure(figsize=(7.5, 5.2), dpi=300)
    
    plt.plot(t_data, c_avg_data, "ko", markersize=3.5, alpha=0.6, label="COMSOL Microscale Diffusion")
    plt.plot(t_data, ldf_model(t_data, k_ldf_fitted), "r-", linewidth=2.2, 
             label=f"Calibrated LDF ($k = {k_ldf_fitted:.4f}\\,\\mathrm{{s^{{-1}}}}$, $R^2 = {r2_score:.4f}$)")
    plt.plot(t_data, ldf_model(t_data, k_ldf_theory), "b--", linewidth=1.8, 
             label=f"Classical Glueckauf ($k = {k_ldf_theory:.4f}\\,\\mathrm{{s^{{-1}}}}$)")

    plt.title("Zeolite 13X: Intrapellet Diffusion vs Lumped Kinetic Models", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Time (s)", fontsize=10, fontweight="bold")
    plt.ylabel(r"Average Pellet Loading $\bar{c}\ (\mathrm{mol/m^3})$", fontsize=10, fontweight="bold")
    plt.xlim(left=0, right=60)
    plt.ylim(bottom=0, top=42)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9, loc="lower right")
    plt.tight_layout()

    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"[Success] Plot saved to:\n{output_plot}\n")


if __name__ == "__main__":
    main()
