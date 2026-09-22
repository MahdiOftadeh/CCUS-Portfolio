"""
Test script for the 1D adsorption column: breakthrough curve simulation.
Runs simulate_adsorption_step and plots the outlet concentration over time.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thermodynamics.dual_site_langmuir import DualSiteLangmuir, DSLParameters
from src.kinetics.ldf import LinearDrivingForce
from src.hydrodynamics.ergun import ErgunEquation
from src.models.column import AdsorptionColumn1D


def main() -> None:
    # ---------------- Isotherm parameters (Dual Site Langmuir) ----------------
    # Representative parameters for Zeolite 13X (CO2 / N2)
    params = [
        DSLParameters(
            name="CO2",
            q_sat1=3.5,            # [mol/kg]
            q_sat2=1.5,            # [mol/kg]
            b0_1=1.2,              # [1/bar]
            b0_2=0.08,             # [1/bar]
            dH_1=-38000.0,         # [J/mol]
            dH_2=-25000.0,         # [J/mol]
            T_ref=298.15           # [K]
        ),
        DSLParameters(
            name="N2",
            q_sat1=1.0,            # [mol/kg]
            q_sat2=0.5,            # [mol/kg]
            b0_1=0.02,             # [1/bar]
            b0_2=0.005,            # [1/bar]
            dH_1=-15000.0,         # [J/mol]
            dH_2=-10000.0,         # [J/mol]
            T_ref=298.15           # [K]
        ),
    ]
    isotherm = DualSiteLangmuir(params)

    # ---------------- Kinetics (Linear Driving Force) ----------------
    ldf = LinearDrivingForce({"CO2": 0.05, "N2": 0.15})  # [1/s]

    # ---------------- Hydrodynamics (Ergun) ----------------
    ergun = ErgunEquation(
        bed_porosity=0.37,
        particle_diameter=0.002    # [m]
    )

    # ---------------- Column Setup ----------------
    column = AdsorptionColumn1D(
        length=1.0,                # [m]
        diameter=0.05,             # [m]
        voidage=0.35,
        bulk_density=650.0,        # [kg/m^3]
        n_nodes=50,
        isotherm=isotherm,
        kinetics=ldf,
        hydrodynamics=ergun
    )

    # ---------------- Simulation inputs ----------------
    temperature = 298.15             # Operating temperature [K]
    u_feed = 0.1                     # Superficial velocity [m/s]
    c_inlet = np.array([5.0, 20.0])  # Inlet concentrations [mol/m^3] (CO2, N2)
    duration = 600.0                 # Adsorption duration [s]

    print("Running breakthrough simulation...")
    t_eval, C_history, q_history = column.simulate_adsorption_step(
        duration=duration,
        u_feed=u_feed,
        c_inlet=c_inlet,
        temperature=temperature,
    )

    # Outlet concentration is at the last spatial node (-1)
    # Shape of C_history is typically (n_components, n_nodes, n_time) or (n_time, n_components, n_nodes)
    # Depending on column.py, outlet concentration for CO2 (index 0) over time:
        # Ensure correct indexing based on C_history shape
    if C_history.shape[0] == len(params):
        co2_outlet = C_history[0, -1, :]
        n2_outlet = C_history[1, -1, :]
    elif C_history.shape[1] == len(params):
        co2_outlet = C_history[:, 0, -1]
        n2_outlet = C_history[:, 1, -1]
    else:
        co2_outlet = C_history[:, -1, 0]
        n2_outlet = C_history[:, -1, 1]


    # ---------------- Ensure Results Directory Exists ----------------
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Plot ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(t_eval, co2_outlet, lw=2, color="crimson", label=f"CO2 Outlet (Feed={c_inlet[0]:.1f} mol/m³)")
    plt.plot(t_eval, n2_outlet, lw=2, color="navy", label=f"N2 Outlet (Feed={c_inlet[1]:.1f} mol/m³)")
    plt.axhline(c_inlet[0], color="crimson", ls="--", alpha=0.6, label="CO2 Feed")
    plt.axhline(c_inlet[1], color="navy", ls="--", alpha=0.6, label="N2 Feed")
    plt.xlabel("Time [s]")
    plt.ylabel("Concentration [mol/m³]")
    plt.title("Breakthrough Curves — Zeolite 13X (1D Adsorption Column)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    output_path = results_dir / "breakthrough_curve.png"
    plt.savefig(output_path, dpi=150)
    print("Simulation finished successfully.")
    print(f"Final CO2 outlet concentration: {co2_outlet[-1]:.4f} mol/m3")
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
