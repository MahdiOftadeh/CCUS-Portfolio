<div align="center">

# 🧪 Dual-Stage VPSA Process for $\text{CO}_2$ Capture
### High-Performance Adsorption Simulation on Zeolite 13X

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Optimization](https://img.shields.io/badge/Optimization-Bayesian-FF6B6B?style=for-the-badge)
![Target](https://img.shields.io/badge/DOE_NETL-Compliant-00C853?style=for-the-badge)

*A professional, reproducible framework for simulating and optimizing dual-stage VPSA systems in CCUS applications.*

</div>

---

## 🚀 Executive Summary
This repository contains the simulation core and optimization routines for a **2-Stage Vacuum Pressure Swing Adsorption (VPSA)** system. Utilizing **Zeolite 13X**, the process is designed to capture $\text{CO}_2$ from flue gas ($15\% \text{ CO}_2 / 85\% \text{ N}_2$) while meeting rigid industrial benchmarks.

## 📈 Key Performance Indicators (Optimized)
Our model has successfully converged to the following targets, balancing purity and recovery against energy consumption:

| Metric | Target | **Achieved** |
| :--- | :---: | :---: |
| **$\text{CO}_2$ Purity** | $\ge 95.0\%$ | **$95.13\%$** ✅ |
| **$\text{CO}_2$ Recovery** | $\ge 90.0\%$ | **$91.65\%$** ✅ |
| **Energy (SEC)** | Industrial | **$150.9 \text{ kWh/t } \text{CO}_2$** |
| **Recycle Fraction** | Optimized | **$0.75$** |
| **Rinse Ratio** | Optimized | **$0.833$** |

---

## 🏗️ Repository Architecture
The project is structured to ensure modularity and reproducibility:
```text
project_01_vpsa_zeolite13x/
├── data/          # Input parameters, sweep results, and state logs
├── results/       # Visual performance dashboards and JSON metrics
├── src/           # Core simulation, optimization, and solver scripts
└── README.md      # Project documentation
💻 Technical Stack
Modeling: 1D Adsorption Column (DSL Isotherm, LDF Kinetics).
Solver: Cyclic Steady State (CSS) convergence logic.
Optimization: Parametric sweeps and Bayesian optimization routines.
Languages: Python (NumPy, SciPy, Matplotlib).
🛠️ Quick Start
Clone the repository:
bash
   git clone https://github.com/yourusername/project_01_vpsa_zeolite13x.git
   
Run the optimized solver:
bash
   cd project_01_vpsa_zeolite13x/src
   python run_corrected_dual_stage_vpsa.py
   
Check Results:Output files will be generated automatically in the ../results/ directory, including the final performance dashboard.
📜 Academic / Industrial Context
Engineering Philosophy: Results are framed as thermodynamic/kinetic limits. Real-world implementation should account for pump efficiencies (
𝜂
<
1
η<1
) and pressure drops.
Focus: Advancing Carbon Capture, Utilization, and Storage (CCUS) process intensification.
---
---

<div align="center">

<h3>Developed by Mahdi Oftadeh</h3>
<p><em>Chemical Engineering | CCUS Researcher</em></p>

<p>
  <a href="https://www.linkedin.com/in/mahdioftadeh/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-Mahdi_Oftadeh-0077B5?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
  <a href="mailto:mahdi.oftadeh.1449@gmail.com">
    <img src="https://img.shields.io/badge/Email-mahdi.oftadeh.1449%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white" alt="Email" />
  </a>
</p>

</div>


