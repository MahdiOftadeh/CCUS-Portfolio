
# Dual-Stage Vacuum Pressure Swing Adsorption (VPSA) for Post-Combustion CO2 Capture
### High-Performance Process Simulation, Cyclic Steady State (CSS) Convergence, and Multi-Objective Bayesian Optimization on Zeolite 13X

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/Modeling-1D_Non--isothermal_Porous-orange?style=flat)
![Standard](https://img.shields.io/badge/DOE_NETL-Compliant-green?style=flat)
![Optimization](https://img.shields.io/badge/Algorithm-Bayesian_Optimization-purple?style=flat)

---

## 🔬 Abstract & Executive Summary
Carbon Capture, Utilization, and Storage (CCUS) process intensification is critical for deep decarbonization of industrial flue gases. This repository houses an advanced, rigorous 1D dynamic simulation and optimization framework for a **Dual-Stage Vacuum Pressure Swing Adsorption (VPSA)** system utilizing **Zeolite 13X** as the selective stationary phase. 

The framework models dry post-combustion flue gas separation (15% CO2 / 85% N2 at 298 K). By coupling **Dual-Site Langmuir (DSL)** equilibrium isotherms, **Linear Driving Force (LDF)** mass transfer kinetics, and strict **Cyclic Steady State (CSS)** mass/energy balance convergence algorithms, the system maps the trade-offs between energetic penalty and separation performance. Multi-objective Bayesian Optimization is deployed to navigate the Pareto frontier, successfully surpassing United States Department of Energy (DOE / NETL) performance benchmarks.

---

## 📊 Key Performance Indicators (Optimized vs. Target)

| Performance Metric | Industrial / DOE Target | Achieved Optimization Result | Thermodynamic / Process Significance |
| :--- | :--- | :--- | :--- |
| **CO2 Purity** | >= 95.0 mol% | **95.13 mol%** | Meets pipeline-transport specifications for EOR and utilization. |
| **CO2 Recovery** | >= 90.0% | **91.65%** | Minimizes fugitive carbon loss and stack emissions. |
| **Specific Energy Consumption (SEC)** | Industrial Benchmark | **150.9 kWh/t CO2** | Competitive energetic footprint via optimized vacuum blowdown/purge management. |
| **Recycle Fraction** | Optimized Parameter | **0.75** | Maximizes internal mass utilization across the dual-stage staging. |
| **Rinse Ratio** | Sweep Optimum | **0.833** | Enhances high-purity front propagation during displacement purge steps. |

---

## 🏗️ Rigorous Repository Architecture
The repository follows strict modular engineering standards to ensure absolute reproducibility and clear separation of concerns across simulation, data logging, and post-processing:

project_01_vpsa_zeolite13x/
├── data/          # Raw input parameters, transient axial profiles, and CSS convergence logs
├── results/       # High-resolution performance dashboards, Pareto CSVs, and JSON summary metrics
├── src/           # Core mathematical solvers, VPSA cycle engines, and Bayesian optimization routines
└── README.md      # Comprehensive project documentation and academic context

---

## 💻 Mathematical Modeling & Technical Stack

* **Governing Equations:** Mass conservation (extended Maxwell-Stefan / multi-component molar balances), momentum conservation (Darcy-Forchheimer friction in packed beds), and energy conservation (non-isothermal solid/gas thermal equilibrium).
* **Adsorption Equilibrium:** Dual-Site Langmuir (DSL) isotherm capturing heterogeneous surface energetic sites on NaX zeolite.
* **Mass Transfer Kinetics:** Linear Driving Force (LDF) approximation for intra-crystalline diffusion resistance.
* **Numerical Core:** Finite difference spatial discretization coupled with robust time integration and Cyclic Steady State (CSS) convergence criteria (relative mass accumulation tolerance < 1e-5).
* **Optimization Engine:** Parametric sweeping combined with Gaussian Process-based Bayesian Optimization for SEC minimization constrained by purity and recovery boundaries.
* **Language & Libraries:** Python 3.10+ (NumPy, SciPy, Matplotlib, scikit-optimize).

---

## 🛠️ Quick Start & Execution Guide

1. Clone the repository to your local environment:
git clone https://github.com/yourusername/project_01_vpsa_zeolite13x.git

2. Navigate to the source code directory:
cd project_01_vpsa_zeolite13x/src

3. Execute the optimized dual-stage solver framework:
python run_corrected_dual_stage_vpsa.py

4. Inspect Generated Outputs:
Processed performance dashboards, axial concentration profiles, and convergence metrics will be automatically exported to the ../results/ directory.

---

## 📜 Academic & Industrial Context
* **Thermodynamic Rigor:** All specific energy consumption (SEC) computations account for vacuum pump polytropic efficiencies and mechanical loss factors. Real-world implementations should integrate specific blower/vacuum train characteristics.
* **Process Intensification:** The dual-stage configuration eliminates mass transfer bottlenecks observed in conventional single-stage VPSA columns, establishing a scalable blueprint for industrial-scale post-combustion carbon capture facilities.

---

## 👤 Author & Professional Contact

**Mahdi Oftadeh**  
Chemical Engineering BSc Student | CCUS & Process Optimization Researcher  
Islamic Azad University  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mahdi_Oftadeh-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mahdioftadeh/)
[![Gmail](https://img.shields.io/badge/Gmail-mahdi.oftadeh.1449@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:mahdi.oftadeh.1449@gmail.com)
