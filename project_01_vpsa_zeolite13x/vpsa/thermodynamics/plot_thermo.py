"""
Definitive Thermodynamic Validation Engine & Q1 Publication Dashboard.
Robust local file detection and targeted module linking.
Authoritative standards: Cavenati et al. (2004) & Krishnamurthy et al. (2014).
"""

import glob
import importlib.util
import os
import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution and server stability
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Dynamic & Robust Module Loader
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Direct candidate prioritized as designated
primary_module_name = "Thermodynamics module 0.1.py"
primary_module_path = os.path.join(current_dir, primary_module_name)

thermo_path = None
if os.path.exists(primary_module_path):
    thermo_path = primary_module_path
else:
    candidate_patterns = [
        "Thermodynamics*.py",
        "thermodynamics*.py",
        "*thermo*.py",
    ]
    for pat in candidate_patterns:
        matches = glob.glob(os.path.join(current_dir, pat))
        for m in matches:
            base = os.path.basename(m)
            if base != os.path.basename(__file__):
                thermo_path = m
                break
        if thermo_path:
            break

if thermo_path is None or not os.path.exists(thermo_path):
    raise FileNotFoundError(
        f"Could not find thermodynamics module inside: {current_dir}\n"
        f"Expected 'Thermodynamics module 0.1.py' adjacent to this script."
    )

print(f"[INFO] Successfully linked core engine: {os.path.basename(thermo_path)}")
spec = importlib.util.spec_from_file_location("thermo_core", thermo_path)
thermo = importlib.util.module_from_spec(spec)
sys.modules["thermo_core"] = thermo
spec.loader.exec_module(thermo)

# -----------------------------------------------------------------------------
# 2. Plotting Style Configuration (Q1 Journal Standard)
# -----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 300,
        "mathtext.fontset": "stix",
        "lines.linewidth": 1.8,
    }
)

fig, axs = plt.subplots(2, 3, figsize=(15.5, 9.5))

# -----------------------------------------------------------------------------
# 3. Model Instantiation
# -----------------------------------------------------------------------------
# CO2: qm0=5.50 mol/kg, b0=2.85 1/bar, t0=0.48, dH=-35500.0 J/mol
co2_model = thermo.TemperatureDependentToth(
    qm0=5.50, b0=2.85, t0=0.48, dH=-35500.0, T0=298.15
)
# N2:  qm0=2.30 mol/kg, b0=0.12 1/bar, t0=0.95, dH=-18000.0 J/mol
n2_model = thermo.TemperatureDependentToth(
    qm0=2.30, b0=0.12, t0=0.95, dH=-18000.0, T0=298.15
)

iso_co2_298 = co2_model.at_temperature(298.15)
iso_n2_298 = n2_model.at_temperature(298.15)
iast_engine = thermo.BinaryIAST(co2_model, n2_model)

# =============================================================================
# Panel (a): Multicomponent Equilibrium vs Cavenati (2004) Experimental Data
# =============================================================================
p_eval = np.linspace(0.005, 1.5, 120)
temp_levels = [298.15, 308.15, 323.15]
palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for T_k, col in zip(temp_levels, palette):
    iso_co2_t = co2_model.at_temperature(T_k)
    iso_n2_t = n2_model.at_temperature(T_k)

    axs[0, 0].plot(
        p_eval,
        [iso_co2_t.loading(p) for p in p_eval],
        color=col,
        label=f"CO2 ({int(T_k)} K)",
    )
    axs[0, 0].plot(
        p_eval,
        [iso_n2_t.loading(p) for p in p_eval],
        color=col,
        linestyle="--",
        label=f"N2 ({int(T_k)} K)",
    )

# =============================================================================
# Recursive CSV Lookup
# =============================================================================
def find_file_recursive(filename, search_root):
    for root, dirs, files in os.walk(search_root):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Start search from the project root (going 3 levels up from current script)
search_start = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
target_csv = "cavenati_isotherms_clean.csv"
csv_file = find_file_recursive(target_csv, search_start)


if csv_file:
    try:
        df_exp = pd.read_csv(csv_file)
        df_exp.columns = [c.strip().lower() for c in df_exp.columns]
        c_col = next(c for c in df_exp.columns if "comp" in c)
        t_col = next(c for c in df_exp.columns if "temp" in c)
        p_col = next(c for c in df_exp.columns if "press" in c)
        q_col = next(c for c in df_exp.columns if "load" in c)

        for T_k, col in zip(temp_levels, palette):
            co2_exp = df_exp[
                (df_exp[c_col].str.upper() == "CO2")
                & (np.isclose(df_exp[t_col], T_k, atol=1.5))
            ]
            n2_exp = df_exp[
                (df_exp[c_col].str.upper() == "N2")
                & (np.isclose(df_exp[t_col], T_k, atol=1.5))
            ]
            if not co2_exp.empty:
                axs[0, 0].scatter(
                    co2_exp[p_col],
                    co2_exp[q_col],
                    color=col,
                    edgecolor="black",
                    s=25,
                    alpha=0.8,
                    zorder=4,
                )
            if not n2_exp.empty:
                axs[0, 0].scatter(
                    n2_exp[p_col],
                    n2_exp[q_col],
                    color=col,
                    marker="^",
                    edgecolor="black",
                    s=22,
                    alpha=0.8,
                    zorder=4,
                )
        print(f"[INFO] Loaded experimental data points from: {csv_file}")
    except Exception as e:
        print(f"[WARNING] Could not parse CSV points: {e}")
else:
    print("[WARNING] CSV dataset not found. Proceeding with pure model curves.")

axs[0, 0].set_title(r"(a) Adsorption Equilibrium (Model vs Cavenati Exp.)")
axs[0, 0].set_xlabel(r"Pressure $P$ (bar)")
axs[0, 0].set_ylabel(r"Equilibrium Uptake $q^*$ (mol/kg)")
axs[0, 0].grid(True, linestyle=":", alpha=0.6)
axs[0, 0].legend(ncol=2, frameon=True, loc="lower right", fontsize=8)

# =============================================================================
# Panel (b): Rigorous IAST Binary Selectivity Map
# =============================================================================
y_co2_grid = np.linspace(0.02, 0.98, 45)
pressure_levels = [0.5, 1.0, 1.5]
linestyles = [":", "-", "--"]

for P_tot, ls in zip(pressure_levels, linestyles):
    sel_vals = []
    for y1 in y_co2_grid:
        try:
            q1, q2, x1, psi = iast_engine.solve(P_tot, y1, T=298.15)
            x2 = 1.0 - x1
            y2 = 1.0 - y1
            alpha = (x1 / y1) / (x2 / y2)
            sel_vals.append(alpha)
        except Exception:
            sel_vals.append(np.nan)

    axs[0, 1].plot(
        y_co2_grid,
        sel_vals,
        color="#d62728",
        linestyle=ls,
        label=rf"$P = {P_tot}\ \mathrm{{bar}}$",
    )

axs[0, 1].axvspan(
    0.10, 0.15, color="gray", alpha=0.2, label=r"Flue Gas Window (10–15%)"
)
axs[0, 1].set_title(r"(b) Rigorous IAST Selectivity ($\alpha_{\mathrm{CO}_2/\mathrm{N}_2}$)")
axs[0, 1].set_xlabel(r"Gas Mole Fraction $y_{\mathrm{CO}_2}$")
axs[0, 1].set_ylabel(r"Selectivity $\alpha_{12} = (x_1/y_1)/(x_2/y_2)$")
axs[0, 1].grid(True, linestyle=":", alpha=0.6)
axs[0, 1].legend(loc="upper right")

# =============================================================================
# Panel (c): Heterogeneous Isosteric Heat of Adsorption
# =============================================================================
q_range = np.linspace(0.05, 4.8, 100)

if hasattr(iso_co2_298, "get_isosteric_heat"):
    dh_co2_vals = [iso_co2_298.get_isosteric_heat(q) / 1000.0 for q in q_range]
else:
    # Thermodynamic derivative fallback via Clausius-Clapeyron if empirical decay not exposed
    p_eq = [optimize.root_scalar(lambda p: iso_co2_298.loading(p) - q_val, bracket=[1e-5, 50.0], method='brentq').root for q_val in q_range]
    dh_co2_vals = [co2_model.isosteric_heat(p, 298.15) / 1000.0 for p in p_eq]

dh_n2_const = abs(n2_model.dH) / 1000.0

axs[0, 2].plot(
    q_range,
    dh_co2_vals,
    color="#8c564b",
    label=r"$\mathrm{CO}_2$ (Heterogeneous Site Decay)",
)
axs[0, 2].axhline(
    dh_n2_const,
    color="dimgray",
    linestyle="--",
    label=rf"$\mathrm{{N}}_2\ \mathrm{{Reference}}\ ({dh_n2_const:.1f}\ \mathrm{{kJ/mol}})$",
)
axs[0, 2].set_title(r"(c) Isosteric Enthalpy ($-\Delta H_{\mathrm{ads}}$)")
axs[0, 2].set_xlabel(r"Adsorbed Phase Loading $q$ (mol/kg)")
axs[0, 2].set_ylabel(r"$-\Delta H_{\mathrm{ads}}$ (kJ/mol)")
axs[0, 2].set_ylim(15, 42)
axs[0, 2].grid(True, linestyle=":", alpha=0.6)
axs[0, 2].legend(loc="upper right")

# =============================================================================
# Panel (d): Grand Potential & Inversion Solver Residual
# =============================================================================
p_pts = np.linspace(0.01, 1.5, 75)
pi_vals = np.array([iso_co2_298.spreading_pressure(p) for p in p_pts])
p_recovered = np.array(
    [iso_co2_298.invert_spreading_pressure(pi) for pi in pi_vals]
)
solver_residuals = np.abs(p_pts - p_recovered)

ax_d1 = axs[1, 0]
ax_d2 = ax_d1.twinx()

line1 = ax_d1.plot(
    p_pts,
    pi_vals,
    color="#9467bd",
    label=r"Grand Potential ($\pi A / RT$)",
)
line2 = ax_d2.semilogy(
    p_pts,
    solver_residuals,
    color="crimson",
    linestyle=":",
    label=r"Inversion Error $|P - P_{\mathrm{inv}}|$",
)

ax_d1.set_title(r"(d) Spreading Pressure & Root Inversion Residual")
ax_d1.set_xlabel(r"Pressure $P$ (bar)")
ax_d1.set_ylabel(r"Reduced Spreading Pressure (mol/kg)", color="#9467bd")
ax_d2.set_ylabel(r"Residual (bar) [Log Scale]", color="crimson")
ax_d1.grid(True, linestyle=":", alpha=0.6)

combined_lines = line1 + line2
combined_labels = [l.get_label() for l in combined_lines]
ax_d1.legend(combined_lines, combined_labels, loc="center right", fontsize=8)

# =============================================================================
# Panel (e): Thermodynamic Jacobian Continuity (PDE Stability)
# =============================================================================
dq_dp_co2 = [iso_co2_298.dq_dP(p) for p in p_eval]
dq_dp_n2 = [iso_n2_298.dq_dP(p) for p in p_eval]

axs[1, 1].plot(
    p_eval,
    dq_dp_co2,
    color="#e377c2",
    label=r"$\partial q_{\mathrm{CO}_2}/\partial P$",
)
axs[1, 1].plot(
    p_eval,
    dq_dp_n2,
    color="#7f7f7f",
    linestyle="--",
    label=r"$\partial q_{\mathrm{N}_2}/\partial P$",
)
axs[1, 1].set_title(r"(e) Thermodynamic Jacobian Continuity ($\partial q / \partial P$)")
axs[1, 1].set_xlabel(r"Pressure $P$ (bar)")
axs[1, 1].set_ylabel(r"$\partial q / \partial P$ (mol/(kg$\cdot$bar))")
axs[1, 1].grid(True, linestyle=":", alpha=0.6)
axs[1, 1].legend(loc="upper right")

# =============================================================================
# Panel (f): Asymptotic Henry Limit Verification (P -> 0)
# =============================================================================
p_asymp = np.logspace(-9, -2, 100)
q_over_p = [iso_co2_298.loading(p) / p for p in p_asymp]
kh_exact = iso_co2_298.henry_constant()

axs[1, 2].semilogx(
    p_asymp,
    q_over_p,
    color="#17becf",
    label=r"Numerical $q(P)/P$",
)
axs[1, 2].axhline(
    kh_exact,
    color="black",
    linestyle=":",
    label=rf"Analytical $K_H$ ({kh_exact:.2f})",
)
axs[1, 2].set_title(r"(f) Asymptotic Henry Boundary ($P \to 0$)")
axs[1, 2].set_xlabel(r"Pressure $P$ (bar) [Log Scale]")
axs[1, 2].set_ylabel(r"$q/P$ (mol/(kg$\cdot$bar))")
axs[1, 2].grid(True, which="both", linestyle=":", alpha=0.6)
axs[1, 2].legend(loc="lower left")

# -----------------------------------------------------------------------------
# Save Figure
# -----------------------------------------------------------------------------
plt.tight_layout()
out_file = os.path.join(current_dir, "definitive_thermo_q1_dashboard.png")
plt.savefig(out_file, dpi=300)
print(f"[OK] Validation dashboard successfully generated:\n     {out_file}")
