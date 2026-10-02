فهمیدم. این نسخه کامل README است که بدون هیچ‌گونه بلوک کد (Code Block) یا کاراکتر بکتیک (` ``` `) آماده شده است. کافی است این متن را مستقیماً در فایل README.md کپی کنی و ذخیره کنی.

# Dual-Stage VPSA Process for CO2 Capture
### High-Performance Adsorption Simulation on Zeolite 13X

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Target](https://img.shields.io/badge/DOE_NETL-Compliant-orange)
![Status](https://img.shields.io/badge/Optimization-Bayesian-green)

---

## 🚀 Executive Summary
This project provides a robust, multi-scale simulation and optimization framework for a Dual-Stage Vacuum Pressure Swing Adsorption (VPSA) system targeting post-combustion CO2 capture from dry flue gas (15% CO2 / 85% N2) using Zeolite 13X. 

By balancing thermodynamic trade-offs across the multi-cycle Pareto frontier, the dual-stage configuration simultaneously satisfies the stringent US Department of Energy (DOE) / NETL purity and recovery mandates while maintaining industrial energy viability.

## 📈 Key Performance Indicators (Optimized)

| Metric | Target / Standard | Achieved |
| :--- | :--- | :--- |
| CO2 Purity | >= 95.0% | 95.13% |
| CO2 Recovery | >= 90.0% | 91.65% |
| Energy (SEC) | Industrial Benchmark | 150.9 kWh/t CO2 |
| Recycle Fraction | Optimized Parameter | 0.75 |
| Rinse Ratio | Sweep Optimum | 0.833 |

## 🏗️ Repository Architecture
The project is structured to ensure modularity and reproducibility:

project_01_vpsa_zeolite13x/
├── data/          # Input parameters, sweep results, and state logs
├── results/       # Visual performance dashboards and JSON metrics
├── src/           # Core simulation, optimization, and solver scripts
└── README.md      # Project documentation

## 💻 Technical Stack
* Modeling: 1D Adsorption Column (DSL Isotherm, LDF Kinetics).
* Solver: Cyclic Steady State (CSS) convergence logic.
* Optimization: Parametric sweeps and Bayesian optimization routines.
* Languages: Python (NumPy, SciPy, Matplotlib).

## 🛠️ Quick Start

1. Clone the repository:
git clone https://github.com/yourusername/project_01_vpsa_zeolite13x.git

2. Run the optimized solver:
cd project_01_vpsa_zeolite13x/src
python run_corrected_dual_stage_vpsa.py

3. Check Results:
Output files will be generated automatically in the ../results/ directory, including the final performance dashboard.

## 📜 Academic / Industrial Context
* Engineering Philosophy: Results are framed as thermodynamic/kinetic limits. Real-world implementation should account for pump efficiencies (eta < 1) and pressure drops.
* Focus: Advancing Carbon Capture, Utilization, and Storage (CCUS) process intensification.

## 👤 Author

**Mahdi Oftadeh**  
Chemical Engineering | CCUS Researcher  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mahdi_Oftadeh-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mahdioftadeh/)
[![Gmail](https://img.shields.io/badge/Gmail-mahdi.oftadeh.1449@gmail.com-D14836?logo=gmail&logoColor=white)](mailto:mahdi.oftadeh.1449@gmail.com)
