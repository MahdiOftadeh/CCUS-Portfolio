# IAST Solver

Ideal Adsorbed Solution Theory (IAST) solver for N-component gas
adsorption equilibria.

**v0.1.0** - basic but comprehensive. Pure, tested, ready to extend.

## What it does

Given pure-component isotherms and a gas mixture (P, y, T), computes:

- `q_i` - equilibrium loading of each component [mol/kg]
- `x_i` - adsorbed-phase mole fractions
- `psi` - common reduced spreading pressure [mol/kg]

## Install

    pip install -e ".[test]"

## Quick start

    from iast import IASTSolver, TothThermal

    co2 = TothThermal(qm=5.50, b0=2.85, t=0.480, Q=35500.0)
    n2  = TothThermal(qm=2.30, b0=0.12, t=0.950, Q=18000.0)

    solver = IASTSolver([co2, n2])
    res = solver.solve(P=1.0, y=[0.15, 0.85], T=298.15)
    print(res["q"])
    print(res["x"])

See examples/co2_n2_13x.py for a fuller demo.

## Theory

IAST assumes the adsorbed phase is an ideal solution in equilibrium with
the gas phase. Equilibrium requires:

1. Equal spreading pressure across all components:
   psi_i(P_i^0) = psi_j(P_j^0) = psi_mix for all i, j
   where psi_i(P) = integral_0^P q_i(P')/P' dP'

2. Phase equilibrium (Raoult-like):
   x_i * phi_i^0 * P_i^0 = y_i * phi_i * P

3. Reciprocal sum formula for total loading:
   1/q_mix = sum_i (x_i / q_i^0(P_i^0))
   q_i = x_i * q_mix

The solver finds psi_mix such that the above holds exactly.

## Numerical method

- Unknown: log(psi) - better conditioning than psi.
- Objective: log(sum_i x_i(psi)) = 0 when sum_i x_i = 1.
- Root-finder: Brent's method with automatic bracket expansion.
- Cache: P_i^0(psi, T) cached per solver instance.

## Features

- Pure isotherms: Langmuir, Dual-Site Langmuir (DSL), Toth.
- Temperature dependence: van't Hoff on b(T).
- Fugacity: optional Peng-Robinson EOS.
- Analytical dq/dP for all isotherms.
- Isosteric heat q_st via Clausius-Clapeyron.
- Jacobian: dq/dP, dq/dy, dq/dT.
- Gibbs-Duhem consistency check at runtime.

## Units

- P: bar
- q: mol/kg
- T: K
- y, x: dimensionless

## Limitations (honest)

v0.1.0 does NOT include:

1. RAST - non-ideal adsorbed phase (gamma_i != 1). v0.2 target.
2. Tabulated isotherms (spline). Pre-fit to Langmuir/DSL/Toth.
3. ML-based isotherms.
4. Hysteresis.
5. Coupled heat/mass balances.
6. Batch screening API. v0.2 target.
7. Validation against N > 3 experimental data.

## Tests

    pytest -v

## Reference

Theory: Myers & Prausnitz (1965), AIChE J. 11, 121-127.
Cavenati parameters: Cavenati et al. (2004), J. Chem. Eng. Data 49, 1095-1101.

## License

MIT