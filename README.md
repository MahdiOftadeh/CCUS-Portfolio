# CCUS-Portfolio

A computational research portfolio focused on **Carbon Capture, Utilization, and Storage (CCUS)** optimization, leveraging Physics-Informed Machine Learning, Advanced AI/LLM integrations, Surrogate Modeling, and Bayesian Optimization.

## 🎯 Research Vision
This repository serves as a centralized hub for my computational research efforts in CCUS. My work aims to bridge the gap between classical chemical engineering principles and advanced data-driven methodologies—including **Artificial Intelligence (AI)**, **Large Language Models (LLM)**, and **Deep Learning**—to accelerate the transition to sustainable energy systems.

## 🛠️ Core Methodologies
The projects within this portfolio leverage a synergistic approach of:
* **Physics-Informed Machine Learning (PIML):** Integrating thermodynamic and transport phenomena constraints directly into neural network architectures.
* **AI & LLM-Assisted Research:** Utilizing AI agents and LLM pipelines to automate literature review, data extraction, and research summarization.
* **Surrogate Modeling:** Developing high-fidelity, low-cost computational models (using **Machine Learning**) to replace expensive mechanistic simulations.
* **Bayesian Optimization:** Navigating complex, high-dimensional design spaces for optimal process configuration.

## 📂 Repository Structure
The repository is organized into specific research domains and software packages:

```text
CCUS-Portfolio/
├── llm/                           # AI-powered tools (e.g., LLM paper summarizers)
├── project_01_vpsa_zeolite13x/    # Core VPSA CO2/N2 Separation Project
│   ├── data/                      # Material and cycle configuration data
│   ├── scripts/                   # Execution and optimization scripts
│   └── vpsa_simulation_suite/     # Core simulation library (thermodynamics, solvers)
├── vpsa_commercial_suite/         # Generated dashboards, figures, and reports
├── tools/                         # Utility scripts (e.g., repository scorer)
└── .github/workflows/             # CI/CD pipelines
🚀 Key Research Themes
Project 01 (VPSA Zeolite 13X): Multi-bed Vacuum Pressure Swing Adsorption modeling. This includes a comprehensive thermodynamics engine (Toth/DSL isotherms, IAST solver, coupled Jacobians) and process optimization.

LLM Paper Summarizer: A module built to leverage Large Language Models for automated extraction and summarization of critical insights from dense CCUS research papers.

Process KPI & Economics: Tools for evaluating the techno-economic viability of carbon capture processes.

🚦 Getting Started
Each project sub-directory contains its own README.md, requirements.txt, and execution scripts.
To run the core VPSA simulation tests:

bash
cd project_01_vpsa_zeolite13x/vpsa_simulation_suite
python -m pytest tests/ -v
📄 License
This project is licensed under the terms included in the LICENSE file in the root directory.
