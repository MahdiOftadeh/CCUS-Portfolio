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
        project_root / "data" / "materials" / "ldf_data",
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

    # Statistics
    ss_res = np.sum((c_avg_data - ldf_model(t_data, k_ldf_fitted)) ** 2)
    ss_tot = np.sum((c_avg_data - np.mean(c_avg_data)) ** 2)
    r2_score = float(1.0 - (ss_res / ss_tot))
    deviation_pct = float(abs(k_ldf_fitted - k_ldf_theory) / k_ldf_theory * 100.0)

    print("-" * 60)
    print(f"Theoretical k_LDF (Glueckauf Analytical): {k_ldf_theory:.4f} s^-1")
    print(f"Fitted k_LDF (COMSOL Microscale):         {k_ldf_fitted:.4f} s^-1")
    print(f"Relative Discrepancy (Overestimation):   {deviation_pct:.2f} %")
    print(f"Goodness of Fit (R^2 Score):              {r2_score:.4f}")
    print("-" * 60)

    # Plotting for Q1 paper publication
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
    print(f"[Success] Plot generated at:\n{output_plot}\n")


if __name__ == "__main__":
    main()
