<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/2/2c/Rotating_earth_%28large%29.gif" width="90" alt="Spinning Earth">

# CCUS-Portfolio | Industrial-Grade VPSA Simulation & Computational Suite

**Bridging First-Principles Chemical Engineering with High-Performance Computing, Stiff ODE Solvers, and Pilot-Validated Process Digital Twins**

[![Release](https://img.shields.io/badge/Release-v0.5.0-blue.svg?style=for-the-badge)](https://github.com/MahdiOftadeh/CCUS-Portfolio/releases)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Live Report](https://img.shields.io/badge/📊_Live_Validation_Report-GitHub_Pages-success.svg?style=for-the-badge)](https://MahdiOftadeh.github.io/CCUS-Portfolio/)

*A computational research portfolio focused on Carbon Capture, Utilization, and Storage (CCUS) optimization.*

[📖 Live Report](https://MahdiOftadeh.github.io/CCUS-Portfolio/) • [🚀 Latest Release](https://github.com/MahdiOftadeh.github.io/CCUS-Portfolio/releases) • [🏗️ Architecture](ARCHITECTURE.md)

</div>

---

## 🔄 Dynamic Process Visualization

<div align="center">
  <img src="project_01_vpsa_zeolite13x/vpsa_simulation_suite/thermodynamics/v0.4.0/v0.4.0_figures/fig14_vpsa_animation.gif" width="65%" style="border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);" alt="VPSA Cycle Animation">
  <p><i>Live simulation of the VPSA cycle trajectory (CO2 vs N2 loadings) moving through adsorption, blowdown, and purge steps.</i></p>
</div>

---

## 🎯 Research Vision

This repository serves as a centralized hub for advanced computational research in CCUS. The goal is to bridge the gap between classical chemical engineering principles and cutting-edge data-driven methodologies—including **Artificial Intelligence (AI)**, **Large Language Models (LLM)**, **Physics-Informed Machine Learning (PIML)**, and **Bayesian Optimization**—to accelerate the transition to sustainable energy systems.

---

## 📊 Visual Showcase — v0.5.0

<div align="center" style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; align-items: center; justify-items: center; max-width: 100%;">

  <div style="grid-column: 1 / -1; width: 100%;">
    <a href="https://MahdiOftadeh.github.io/CCUS-Portfolio/">
      <img src="project_01_vpsa_zeolite13x/vpsa_simulation_suite/thermodynamics/v0.5.0/Thermodynamics_v0.5.0_figures/20_dashboard.png"
           width="100%"
           style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);"
           alt="v0.5.0 Test Results Dashboard">
    </a>
  </div>

  <a href="https://MahdiOftadeh.github.io/CCUS-Portfolio/">
    <img src="project_01_vpsa_zeolite13x/vpsa_simulation_suite/thermodynamics/v0.5.0/Thermodynamics_v0.5.0_figures/01_pure_isotherms.png"
         width="100%"
         style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);"
         alt="Pure-Component Isotherms — Zeolite 13X">
  </a>

  <a href="https://MahdiOftadeh.github.io/CCUS-Portfolio/">
    <img src="project_01_vpsa_zeolite13x/vpsa_simulation_suite/thermodynamics/v0.5.0/Thermodynamics_v0.5.0_figures/04_iast_pressure_sweep.png"
         width="100%"
         style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);"
         alt="Binary IAST Pressure Sweep">
  </a>

  <a href="https://MahdiOftadeh.github.io/CCUS-Portfolio/">
    <img src="project_01_vpsa_zeolite13x/vpsa_simulation_suite/thermodynamics/v0.5.0/Thermodynamics_v0.5.0_figures/06_extended_dsl_jacobian.png"
         width="100%"
         style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);"
         alt="ExtendedDSL Analytical Jacobian — 6 Panels">
  </a>

  <a href="https://MahdiOftadeh.github.io/CCUS-Portfolio/">
    <img src="project_01_vpsa_zeolite13x/vpsa_simulation_suite/thermodynamics/v0.5.0/Thermodynamics_v0.5.0_figures/17_stability_surface.png"
         width="100%"
         style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);"
         alt="Stability Surface — 10 Pressure Decades × 150 K">
  </a>

</div>

<p align="center">
  <i>All figures are generated automatically by the <b>v0.5.0 validation suite</b> — 101/101 checks passed — and hosted on the <b><a href="https://MahdiOftadeh.github.io/CCUS-Portfolio/">Live Report</a></b>.</i>
</p>
---

## 🧠 Core Methodologies & AI Integration

*   **AI & LLM-Assisted Research**: The `llm/` module utilizes Large Language Models and AI agents to automate literature review, data extraction, and research summarization.
*   **Physics-Informed Machine Learning (PIML)**: Integrating thermodynamic and transport constraints directly into neural network architectures.
*   **Surrogate Modeling**: Developing high-fidelity, low-cost computational models to replace expensive mechanistic simulations.
*   **Bayesian Optimization**: Navigating complex, high-dimensional design spaces for optimal process configuration.

---

## 🧩 Comprehensive Engine Architecture

The VPSA simulation suite is a modular, multi-physics platform. Every engine is rigorously validated against literature benchmarks and experimental datasets.

| Engine Module | Description |
| :--- | :--- |
| **🌡️ Thermodynamics** | Analytic models (Toth, DSL, IAST) for mixture equilibrium and isosteric heat. |
| **💨 Mass Transfer** | Linear Driving Force (LDF) kinetics and breakthrough curve modeling. |
| **🌊 Fluid Transport** | Gas-phase transport properties, viscosity, and mass diffusivity. |
| **💧 Hydrodynamics** | Pressure drop, Ergun equation, and flow distribution across the bed. |
| **🔥 Thermal Wall** | Heat transfer coefficients, wall thermal mass, and energy balances. |
| **🏗️ Equipment & Boundary Network** | Compressor maps, valves, and piping network integration. |
| **🏛️ Column Modeling** | 1D/2D spatial discretization for adsorption columns. |
| **🔄 Multi-Bed Cycle** | Complex Skarstrom and VPSA cycle scheduling (CSS, Pressurization, Blowdown). |
| **📉 Aging & Degradation** | Long-term adsorbent degradation and cyclic capacity loss. |
| **💰 Economics** | CAPEX/OPEX correlations, process KPIs, and cost-of-capture models. |
| **⚖️ Multi-Objective Optimization** | Pareto front generation, surrogate modeling, and DOE. |
| **🌐 Digital Twin** | Real-time data assimilation and dynamic process mirroring. |
| **👑 Master Engine** | Unified orchestrator coupling all physics, economics, and optimization loops. |

> **🚀 Latest Highlight (v0.5.0)**: The Thermodynamics engine features a fully analytic framework. The `ExtendedDSL` model replaces the numerical IAST solver, yielding a **~7000x speedup** (~4.5 µs vs ~32 ms) while maintaining thermodynamic consistency across 101 automated test assertions.

---

## 📂 Repository Structure
```
CCUS-Portfolio/

│

├── .github/workflows/ # CI/CD pipelines (GitHub Actions)

├── docs/ # GitHub Pages (Live Validation Report)

├── llm/ # AI & LLM tools (paper summarizers)

│

├── project_01_vpsa_zeolite13x/ # Core VPSA CO2/N2 Separation Project

│ ├── data/ # Material and cycle configuration data

│ │ ├── cycle_configs/ # VPSA cycle JSON configurations

│ │ └── material/ # Literature isotherms & LDF data (CSV/JSON)

│ ├── docs/ # Theory and DOE targets

│ ├── outputs/ # Generated datasets, figures, and tables

│ ├── scripts/ # Execution & optimization scripts

│ │ ├── run_breakthrough.py

│ │ ├── run_vpsa_cycle.py

│ │ └── run_optimization.py

│ ├── tests/ # Integration tests (isotherms, mass balance)

│ │

│ ├── vpsa_commercial_suite/ # Commercial-grade dashboards and reports

│ │ ├── figures/ # High-level UI dashboards

│ │ └── reports/

│ │

│ └── vpsa_simulation_suite/ # Core Python Package

│ ├── auxiliary/ # Compressor maps & units network

│ ├── benchmarks/ # Jupyter notebooks for validation

│ ├── economics/ # Techno-economic KPIs

│ ├── mass_transfer/ # LDF kinetics

│ ├── optimization/ # DOE, Pareto, Surrogates

│ ├── solvers/ # CSS, Breakthrough, Cycle solvers

│ ├── thermodynamics/ # 🌟 Thermodynamics Engine

│ │ ├── archive/v0.1.0/ # Historical baseline

│ │ ├── v0.3.0/ # Legacy suite

│ │ ├── v0.4.0/ # Analytic engine · 57-check tests

│ │ └── v0.5.0/ # ⭐ LATEST · 101-check tests + 20 figures

│ │ ├── Thermodynamics_v0.5.0.py # Engine

│ │ ├── Thermodynamics_v0.5.0_test.py # 101-check validation suite

│ │ ├── Thermodynamics_v0.5.0_report.html # Self-contained HTML report

│ │ ├── Thermodynamics_v0.5.0_test_results.json # Machine-readable diagnostics

│ │ ├── Thermodynamics_v0.5.0_console_log.txt # Verbatim stdout transcript

│ │ └── Thermodynamics_v0.5.0_figures/ # 20 publication-quality PNGs

│ │ ├── 01_pure_isotherms.png

│ │ ├── 02_derivative_validation.png

│ │ ├── 03_isosteric_heat.png

│ │ ├── 04_iast_pressure_sweep.png

│ │ ├── 05_iast_composition_sweep.png

│ │ ├── 06_extended_dsl_jacobian.png

│ │ ├── 07_asymptotic_henry_psi.png

│ │ ├── 08_gibbs_duhem.png

│ │ ├── 09_vpsa_window.png

│ │ ├── 10_temperature_sensitivity.png

│ │ ├── 11_site_decomposition.png

│ │ ├── 12_jacobian_validation.png

│ │ ├── 13_selectivity_2d.png

│ │ ├── 14_henry_constants.png

│ │ ├── 15_working_capacity.png

│ │ ├── 16_jacobian_signs.png

│ │ ├── 17_stability_surface.png

│ │ ├── 18_cross_model_validation.png

│ │ ├── 19_vpsa_cycle.png

│ │ └── 20_dashboard.png

│ ├── transport/ # Hydrodynamics & kinetics

│ └── visualization/ # Plotting utilities

│

└── README.md


```

## 🚀 Quick Start

### 1. Requirements & Installation
Ensure you have Python 3.10+ installed.
```bash
pip install -r project_01_vpsa_zeolite13x/requirements.txt
pip install numpy scipy matplotlib pillow
```
2. Run the Core Thermodynamic Validation (v0.5.0)
Independently verify all 101 physics assertions and regenerate the 20 diagnostic figures:
```
python project_01_vpsa_zeolite13x/vpsa_simulation_suite/thermodynamics/v0.5.0/Thermodynamics_v0.5.0_test.py
```
3. Run the Full VPSA Simulation Tests
```
cd project_01_vpsa_zeolite13x/vpsa_simulation_suite
python -m pytest tests/ -v
```
📜 License & Citation
This project is licensed under the MIT License - see the LICENSE file for details.

Copyright © 2026 Mahdi Oftadeh

If you use this thermodynamic engine or the VPSA simulation suite in your research, please cite the underlying literature:

Cavenati, S., Grande, C. A., & Rodrigues, A. E. (2004). Adsorption equilibrium of methane, carbon dioxide, and nitrogen on zeolite 13X at high pressures.
Krishnamurthy, S., et al. (2014). CO2 capture from dry flue gas by vacuum swing adsorption: A pilot plant study.
<div align=“center”>

  <h3>🤝 Let's Connect & Collaborate</h3>
  <p>Feel free to reach out for research collaborations, discussions on CCUS, or just to say hi!</p>

  <a href="https://www.linkedin.com/in/mahdioftadeh" style="text-decoration: none; border: none; display: inline-block;"><img src="https://img.shields.io/badge/LinkedIn-Mahdi_Oftadeh-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" style="border: 0; outline: none; vertical-align: middle;"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="mailto:mahdi.oftadeh.1449@gmail.com" style="text-decoration: none; border: none; display: inline-block;"><img src="https://img.shields.io/badge/Email-Contact_Me-EA4335?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" style="border: 0; outline: none; vertical-align: middle;"></a>
  
  <br><br>
  <i>Built with ❤️ for a greener planet and a decarbonized future.</i>
</div>  
