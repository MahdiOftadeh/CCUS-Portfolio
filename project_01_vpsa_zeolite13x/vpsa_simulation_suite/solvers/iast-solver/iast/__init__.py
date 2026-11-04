"""IAST Solver: Ideal Adsorbed Solution Theory for N-component mixtures.

Basic but comprehensive implementation with:
- Analytical pure-component isotherms (Langmuir, DSL, Toth)
- Temperature-dependent (van't Hoff) versions
- Robust log-psi bracketing solver
- Analytical Jacobian (dq/dP, dq/dy, dq/dT)
- Isosteric heat of adsorption (Clausius-Clapeyron)
- Optional Peng-Robinson EOS for fugacity coefficients
- Runtime Gibbs-Duhem consistency check
"""
from .isotherms import (
    BaseIsotherm, BaseThermalIsotherm,
    Langmuir, DSL, Toth,
    LangmuirThermal, DSLThermal, TothThermal,
    R_GAS,
)
from .solver import IASTSolver, IASTConvergenceError, IASTInputError
from .eos import PengRobinson

__version__ = "0.1.0"
__all__ = [
    "BaseIsotherm", "BaseThermalIsotherm",
    "Langmuir", "DSL", "Toth",
    "LangmuirThermal", "DSLThermal", "TothThermal",
    "IASTSolver", "IASTConvergenceError", "IASTInputError",
    "PengRobinson", "R_GAS",
]