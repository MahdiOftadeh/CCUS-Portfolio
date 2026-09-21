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
    # Placeholder values: replace with fitted parameters for Zeolite 13X / Mg-MOF-74.
    params = [
        DSLParameters(
            name="CO2",
            q_sat1=3.0, b1=1e-6,       # site 1: saturation capacity [mol/kg], affinity [1/Pa]
            q_sat2=1.0, b2=1e-8,       # site 2
        ),
        DSLParameters(
            name="N2",
            q_sat1=0.5, b1=1e-8,
            q_sat2=0.0, b2=1e-10,
        ),
    ]
    isotherm = DualSiteLangmuir(params)

    # ---------------- Kinetics (Linear Driving Force) ----------------
    ldf = LinearDrivingForce({"CO2": 0.05, "N2": 0.15})  # LDF coefficients [1/s]

    # ---------------- Hydrodynamics (Ergun) ----------------
    ergun = ErgunEquation(
        particle_diameter=0.002,   # [m]
        bed_porosity=0.4,
        gas_viscosity=1.8e-5,      # [Pa.s]
    )

    # ---------------- Column 1D ----------------
    column = AdsorptionColumn1D(
        num_nodes=20,
        column_length=1.0,         # [m]
        column_diameter=0.05,      # [m]
        particle_density=1200.0,   # [kg/m3] (bulk-like reference density)
        isotherm=isotherm,
        kinetics=ldf,
    )

    # ---------------- Simulate the adsorption step ----------------
    pressure = 1e5        # 1 bar [Pa]
    temperature = 298.0   # [K]
    u_feed = 0.1          # superficial velocity [m/s]
    c_inlet = 5.0         # inlet gas-phase concentration [mol/m3]
    duration = 600.0      # [s]

    result = column.simulate_adsorption_step(
        duration=duration,
        u_feed=u_feed,
        c_inlet=c_inlet,
        temperature=temperature,
        pressure=pressure,
    )

    time = result["time"]
    breakthrough = result["breakthrough_C"]

    # ---------------- Plot ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(time, breakthrough, lw=2, label="Outlet concentration")
    plt.axhline(c_inlet, color="r", ls="--", lw=1, label=f"C_inlet = {c_inlet} mol/m3")
    plt.xlabel("Time [s]")
    plt.ylabel("Outlet concentration C [mol/m3]")
    plt.title("Breakthrough Curve — CO2 Adsorption (1D column)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "breakthrough_curve.png", dpi=150)
    plt.show()

    print("Simulation finished.")
    print(f"Final outlet concentration: {breakthrough[-1]:.4f} mol/m3")
    print(f"C_inlet:                   {c_inlet:.4f} mol/m3")


if __name__ == "__main__":
    main()
