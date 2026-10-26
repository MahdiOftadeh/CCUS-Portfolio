# Architecture

This document describes the current structure of the CCUS-Portfolio repository,
the implemented modules, the planned modules, and the data flow through the
simulation pipeline. It reflects the actual state of the codebase, not an
aspirational target.

## Overview

The repository houses a research portfolio on Carbon Capture, Utilization, and
Storage (CCUS). The primary project, `project_01_vpsa_zeolite13x`, builds a
Vacuum Pressure Swing Adsorption (VPSA) simulation framework for CO2 capture
using Zeolite 13X.

The codebase is organized as a **layered architecture**: each physics domain
(thermodynamics, mass transfer, transport, auxiliary units, economics) is a
self-contained module, and the solver layer composes them into breakthrough
and cycle simulations. This separation lets each physics model be validated
independently before integration.

## Repository Layout
CCUS-Portfolio/
├── README.md # Portfolio overview and research vision
├── ARCHITECTURE.md # This document
├── github_repository_scorer.py # Activity scorer (portfolio tooling)
│
└── project_01_vpsa_zeolite13x/
├── data/ # Input datasets and cycle configurations
│ ├── cycle_configs/ # VPSA cycle parameter sets
│ └── material/ # Isotherms, kinetics, LDF fitting data
├── docs/ # Theory notes and DOE targets
├── outputs/ # Simulation results and validation figures
├── scripts/ # CLI entry points (breakthrough, cycle, optimization)
├── tests/ # Unit and integration tests
├── vpsa_commercial_suite/ # Commercial packaging and dashboards
│ └── figures/ # Thermodynamics, hydrodynamics, mass transfer dashboards
└── vpsa_simulation_suite/ # Core simulation library (see below)
├── auxiliary units & network/ # Compressors, vacuum pumps, piping
├── benchmarks/ # Jupyter notebooks for validation
├── docs/ # Validation reports
├── economics/ # Techno-economic KPIs
├── mass_transfer/ # LDF model, adsorption kinetics
├── optimization/ # DOE, surrogate, Pareto
├── solvers/ # Breakthrough, CSS, cycle solvers
├── thermodynamics/ # Isotherms, gas properties, heat of adsorption
├── transport/ # Hydrodynamics, kinetics
└── visualization/ # Plotting and dashboard utilities

text

## Implemented Modules

The following modules contain working code and are the current foundation of
the framework.

### Thermodynamics — `vpsa_simulation_suite/thermodynamics/`

**`Thermodynamics module 0.1.py`** (~414 lines)
Core thermodynamic engine. Implements adsorption equilibrium models, ideal-gas
and real-gas property calculations, and isosteric heat of adsorption.

**`plot_thermo.py`** (~308 lines)
Diagnostic plotting utilities for thermodynamic validation. Produces the
thermodynamics dashboard in `vpsa_commercial_suite/figures/`.

### Mass Transfer — `vpsa_simulation_suite/mass_transfer/`

**`mass_transfer_academic_core.py`** (~593 lines)
Linear Driving Force (LDF) mass transfer model. Fits LDF coefficients from
experimental uptake data and provides the intraparticle mass transfer rate used
by the breakthrough solver. Validated against the Cavenati isotherm dataset
(see `outputs/figures/ldf_fit_validation.png`).

### Auxiliary Units and Network — `vpsa_simulation_suite/auxiliary units & network/`

**`auxiliary_units_network.py`** (~346 lines)
Models for compressors, vacuum pumps, and interconnecting piping. Computes
auxiliary power consumption from compressor maps (`compressor_map.json`) and
network pressure drops.

### Economics — `vpsa_simulation_suite/economics/`

**`process_kpi_technoeconomic (1).py`** (~412 lines)
Techno-economic KPI calculations. Converts simulation outputs (purity,
recovery, energy) into cost metrics (specific capture cost, energy penalty,
CAPEX/OPEX estimates).

### LDF Parameter Fitting — `project_01_vpsa_zeolite13x/scripts/`

**`fit_ldf_parameter.py`** (~86 lines)
Fits LDF mass transfer coefficients from `data/material/ldf_data.csv`. Produces
the validation figure `outputs/figures/ldf_fit_validation.png`.

### Core Lite — `vpsa_simulation_suite/vpsa_core_lite/`

**`core.py`** (~10 lines)
Minimal numerical core for early-stage prototyping.

## Planned Modules (Scaffolding)

The following modules exist as directory structure and placeholder files. They
define the intended extension points of the framework but are not yet
implemented.

### Solvers — `vpsa_simulation_suite/solvers/`
- `base_solver.py` — abstract solver interface (time-stepping, boundary conditions)
- `breakthrough.py` — 1D advection-dispersion-adsorption solver
- `css.py` — Cyclic Steady State convergence detector
- `cycle.py` — full VPSA cycle (pressurization, adsorption, blowdown, purge, repressurization)

### Optimization — `vpsa_simulation_suite/optimization/`
- `doe.py` — Design of Experiments (grid, Latin Hypercube)
- `doe_targets.py` — DOE objective definitions
- `surrogate.py` — surrogate models for fast design-space exploration
- `pareto.py` — multi-objective Pareto front (purity, recovery, energy)

### Transport — `vpsa_simulation_suite/transport/`
- `hydrodynamics.py` — Ergun equation, axial velocity, pressure drop
- `kinetics.py` — adsorption kinetics (beyond LDF)

### Thermodynamics (additional) — `vpsa_simulation_suite/thermodynamics/`
- `isotherms.py` — standalone isotherm models (Langmuir, Toth, etc.)
- `gas_properties.py` — standalone gas property functions
- `heat_of_adsorption.py` — standalone heat of adsorption models

### Visualization — `vpsa_simulation_suite/visualization/`
- `plots.py` — reusable plotting utilities

### CLI Scripts — `project_01_vpsa_zeolite13x/scripts/`
- `run_breakthrough.py`, `run_vpsa_cycle.py`, `run_optimization.py`, `run_sweep.py`

### Tests — `project_01_vpsa_zeolite13x/tests/`
- `test_isotherm.py`, `test_breakthrough.py`, `test_cycle_css.py`, `test_mass_balance.py`

## Data Flow
Input data (isotherms, kinetics, cycle configs)
│
▼
┌─────────────────────┐
│ Thermodynamics │ equilibrium capacity, heat of adsorption
└──────────┬──────────┘
│
▼
┌─────────────────────┐
│ Mass Transfer │ LDF coefficients, intraparticle rate
└──────────┬──────────┘
│
▼
┌─────────────────────┐
│ Transport │ hydrodynamics, pressure drop (planned)
└──────────┬──────────┘
│
▼
┌─────────────────────┐
│ Solvers │ breakthrough → CSS → full cycle (planned)
└──────────┬──────────┘
│
▼
┌─────────────────────┐
│ Auxiliary │ compressor / vacuum pump power
└──────────┬──────────┘
│
▼
┌─────────────────────┐
│ Economics │ techno-economic KPIs
└─────────────────────┘

text

## Data Assets

### Real datasets
- `data/material/cavenati_isotherms_clean.csv` (~2.7 KB) — CO2/N2 isotherm data on Zeolite 13X
- `data/material/ldf_data.csv` (~2.9 KB) — experimental uptake data for LDF fitting

### Placeholder datasets (to be populated)
- `data/material/krishnamurthy_2014.json` — literature kinetic parameters
- `data/cycle_configs/vpsa_default.json` — default VPSA cycle configuration

## Continuous Integration

`.github/workflows/ci.yml` runs the test suite on every push. As the solver
and optimization modules are implemented, this workflow will be extended to
run integration tests on full cycle simulations.

## Design Principles

**Physics-domain separation.** Each physics layer (thermodynamics, mass
transfer, transport, auxiliary, economics) exposes a pure function or class
interface. The solver layer depends on these interfaces, not on their
implementations. This lets isotherm models, kinetic models, and numerical
schemes be swapped independently.

**Validation against literature data.** The mass transfer model is validated
against the Cavenati isotherm dataset and produces a documented fit figure
(`outputs/figures/ldf_fit_validation.png`). Future modules will follow the
same validation-first pattern.

**Configuration-driven runs.** Cycle parameters (pressures, step times, column
geometry) live in JSON files under `data/cycle_configs/`. No hardcoded physical
constants in the solver code. This makes reproducibility and parameter sweeps
straightforward once the solvers are implemented.

**Incremental implementation.** The directory structure defines the target
architecture. Modules are implemented one physics layer at a time, validated
against literature, and only then integrated into the solver layer. This
avoids the common failure mode of building a monolith that cannot be tested.

## Roadmap

1. **Implement transport layer** — hydrodynamics and kinetics modules
2. **Implement solvers** — breakthrough, CSS detection, full cycle
3. **Populate tests** — unit tests for each physics module, integration tests for the cycle
4. **Implement optimization** — DOE, surrogate models, Pareto front
5. **Populate placeholder datasets** — Krishnamurthy kinetics, default cycle config
6. **Extend CI** — run full pipeline in GitHub Actions
