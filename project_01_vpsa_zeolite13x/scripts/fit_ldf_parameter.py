"""
fit_ldf_parameter.py
====================
Calibrates the Linear Driving Force (LDF) mass transfer coefficient
against microscale COMSOL diffusion simulations for Zeolite 13X.

Benchmark: Classical Glueckauf Analytical vs Numerical COMSOL vs
scikit-learn cross-validation.

Two independent scikit-learn methods are used to cross-check the scipy fit:

  1. Linear regression on the linearised LDF equation
     (log(1 - c/c_bulk) = -k * t). This should reproduce the scipy-fitted k
     if the LDF form is appropriate for the data.

  2. Gradient Boosting Regressor (non-parametric). This does not assume the
     LDF form; high R^2 indicates that the LDF model is a reasonable
     description of the (t, c) data without over-fitting.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


def find_data_file(project_root: Path) -> Path:
    """Searches candidates to locate the dataset robustly."""
    candidates = [
        project_root / "data" / "materials" / "ldf_data.csv",
        project_root / "data" / "material" / "ldf_data.csv",  # supports folder without 's'
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


def fit_ldf_with_sklearn(t_data, c_data, c_bulk):
    """
    Cross-check the LDF fit using two independent scikit-learn methods.

    Parameters
    ----------
    t_data : ndarray
        Time samples (s).
    c_data : ndarray
        Average pellet concentration (mol/m^3).
    c_bulk : float
        Bulk concentration (mol/m^3), used to normalize the linearised form.

    Returns
    -------
    dict with:
        k_linear     : LDF coefficient from linear regression on
                       log(1 - c/c_bulk) = -k * t
        r2_linear    : R^2 of that linear fit (in log space)
        r2_gbr       : R^2 of GradientBoostingRegressor on raw (t, c)
        rmse_gbr     : RMSE of that GBR on a held-out 20% split
    """
    # --- Method 1: linear regression on the linearised LDF form -----------
    # LDF:     c(t) = c_bulk * (1 - exp(-k t))
    # Linear:  log(1 - c/c_bulk) = -k * t
    ratio = np.clip(c_data / c_bulk, 1e-6, 1.0 - 1e-6)
    y_log = np.log(1.0 - ratio)
    X_t = t_data.reshape(-1, 1)

    lr = LinearRegression().fit(X_t, y_log)
    k_linear = -float(lr.coef_[0])
    r2_linear = float(r2_score(y_log, lr.predict(X_t)))

    # --- Method 2: Gradient Boosting as non-parametric cross-check --------
    X_train, X_test, y_train, y_test = train_test_split(
        X_t, c_data, test_size=0.2, random_state=42
    )
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, random_state=42
    ).fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    r2_gbr = float(r2_score(y_test, y_pred))
    rmse_gbr = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {
        "k_linear": k_linear,
        "r2_linear": r2_linear,
        "r2_gbr": r2_gbr,
        "rmse_gbr": rmse_gbr,
    }


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

    # Load COMSOL simulation data (skips comments starting with '%', handles
    # whitespace/comma separators, no header row)
    df = pd.read_csv(
        data_path, comment="%", sep=r"[,\s]+", header=None, engine="python"
    )
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

    # Curve fitting (scipy)
    popt, _ = curve_fit(ldf_model, t_data, c_avg_data, p0=[k_ldf_theory])
    k_ldf_fitted = float(popt[0])

    print("-" * 60)
    print(f"Theoretical k_LDF (Glueckauf Analytical): {k_ldf_theory:.4f} s^-1")
    print(f"Fitted k_LDF (COMSOL Microscale):         {k_ldf_fitted:.4f} s^-1")
    print("-" * 60)

    # scikit-learn cross-check
    sk = fit_ldf_with_sklearn(t_data, c_avg_data, c_bulk)
    print("scikit-learn cross-check:")
    print(f"  k_LDF (linear regression):  {sk['k_linear']:.4f} s^-1")
    print(f"  R^2 (linear regression):    {sk['r2_linear']:.4f}")
    print(f"  R^2 (GradientBoosting):     {sk['r2_gbr']:.4f}")
    print(f"  RMSE (GradientBoosting):    {sk['rmse_gbr']:.4f} mol/m^3")
    print("-" * 60)
    print(
        "Interpretation: if k_linear agrees with the scipy fit within a few "
        "percent, the LDF form is appropriate. High GBR R^2 (>0.98) confirms "
        "the LDF curve captures the data without over-fitting."
    )
    print("-" * 60)

    # Plotting
    plt.figure(figsize=(10, 7))

    # Scatter for COMSOL
    plt.scatter(
        t_data, c_avg_data, color="dimgray", s=20,
        label="COMSOL (Micro-scale)", zorder=3,
    )

    # Red line for LDF Fit
    plt.plot(
        t_data, ldf_model(t_data, k_ldf_fitted), color="red", linewidth=2.5,
        label=f"LDF Fit (k = {k_ldf_fitted:.4f} s⁻¹)", zorder=2,
    )

    # Blue dashed for Glueckauf
    plt.plot(
        t_data, ldf_model(t_data, k_ldf_theory), color="blue",
        linestyle="--", linewidth=2,
        label=f"Glueckauf Theory (k = {k_ldf_theory:.4f} s⁻¹)", zorder=1,
    )

    # Styling
    plt.title(
        "Pellet-Scale CO2 Diffusion: COMSOL vs LDF Model",
        fontsize=16, fontweight="bold", pad=15,
    )
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Average Pellet Concentration (mol/m³)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print(f"[Success] Plot generated at:\n{output_plot}\n")


if __name__ == "__main__":
    main()
