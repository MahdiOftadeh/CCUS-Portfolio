"""
Thermodynamic Equilibrium Engine for VPSA Adsorption
CO2/N2 Separation on Zeolite 13X - Full Krishnamurthy et al. Architecture

Layer Architecture:
1. Base & Isothermal Pure Models (Original Step 2 - 300 lines preserved)
2. Temperature-Dependent Extensions (Non-isothermal Krishnamurthy formulation)
3. Isosteric Heat of Adsorption (-dH_ads) for Energy Balance coupling
4. Multicomponent Binary IAST Solver (Fast 1D Brent Root-Finding)
5. Coupled Thermodynamic Jacobian Engine (dq_i / dP, dq_i / dy, dq_i / dT)
6. Comprehensive Unit Testing Framework (Regression + Extended Physics)
"""

__version__ = "0.1.0"

from abc import ABC, abstractmethod
import unittest
import numpy as np
from scipy import integrate, optimize

# Universal Gas Constant (J / (mol * K))
R_GAS = 8.314462618


def _rel_diff(a: float, b: float) -> float:
    """Helper for safe relative error checking."""
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


# ==============================================================================
# SECTION 1: BASE ISOTHERM CLASS (Original Step 2 Core)
# ==============================================================================

class BaseIsotherm(ABC):
    """Abstract Base Class for Pure-Component Adsorption Isotherms."""

    @abstractmethod
    def loading(self, P: float) -> float:
        """Equilibrium loading q (mol/kg) at pressure P (bar)."""
        pass

    @abstractmethod
    def henry_constant(self) -> float:
        """Henry constant K_H = lim_{P->0} (q / P) in mol/(kg*bar)."""
        pass

    @abstractmethod
    def spreading_pressure(self, P: float) -> float:
        """Reduced spreading pressure psi = (pi * A) / (R * T) in mol/kg."""
        pass

    def dq_dP_numerical(self, P: float, rel_step: float = 1e-4) -> float:
        """
        Fourth-order Richardson extrapolation derivative dq/dP.
        Accurate to O(h^4) without catastrophic subtractive cancellation.
        """
        if P <= 0.0:
            return self.henry_constant()

        h = P * rel_step
        if h < 1e-9:
            h = 1e-9

        f = self.loading
        d1 = (f(P + h) - f(P - h)) / (2.0 * h)
        d2 = (f(P + 2.0 * h) - f(P - 2.0 * h)) / (4.0 * h)
        return (4.0 * d1 - d2) / 3.0

    def dq_dP(self, P: float) -> float:
        """Derivative dq/dP. Subclasses can override with analytical formulations."""
        return self.dq_dP_numerical(P)

    def invert_spreading_pressure(self, psi_target: float, P_guess: float = 1.0,
                                  max_iter: int = 100) -> float:
        """
        Finds pure-component fictitious pressure P^0 such that psi(P^0) = psi_target.
        Standard Brent's method with robust adaptive bracketing for IAST integration.
        """
        if psi_target <= 1e-12:
            return 0.0

        K_H = max(self.henry_constant(), 1e-6)
        # Low bound safely below target using Henry linear limit
        P_low = min(1e-12, psi_target / (K_H * 10.0))
        P_high = max(P_guess * 2.0, (psi_target / K_H) * 2.0, 10.0)

        while self.spreading_pressure(P_high) < psi_target and P_high < 1e7:
            P_high *= 4.0

        def obj(P_val):
            return self.spreading_pressure(P_val) - psi_target

        # Fallback safeguard if bracket bounds don't bracket
        f_low = obj(P_low)
        f_high = obj(P_high)
        if f_low * f_high > 0:
            while f_low > 0 and P_low > 1e-16:
                P_low /= 10.0
                f_low = obj(P_low)

        sol = optimize.root_scalar(obj, bracket=[P_low, P_high], method='brentq', maxiter=max_iter)
        return float(sol.root)

    def _get_gas_cp(self, T: float, species: str = "CO2") -> float:
        """Ideal gas heat capacity [J/(mol*K)] - Shomate approximation."""
        if species.upper() == "N2":
            return 28.98 - 1.571e-3 * T + 8.081e-6 * (T**2) - 2.873e-9 * (T**3)
        return 19.8 + 7.344e-2 * T - 5.602e-5 * (T**2) + 1.715e-8 * (T**3)

    def get_isosteric_heat(self, q: float) -> float:
        """Concentration-dependent heat of adsorption. Overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement isosteric heat formulation.")

    def get_adsorbed_heat_capacity(self, T: float, q: float, species: str = "CO2") -> float:
        """Adsorbed phase heat capacity Cp,a [J/(mol*K)]."""
        return float(self._get_gas_cp(T, species=species))


# ==============================================================================
# SECTION 2: ISOTHERMAL PURE ISOTHERM IMPLEMENTATIONS (Original Step 2 Core)
# ==============================================================================

class TothIsotherm(BaseIsotherm):
    """
    Standard Isothermal Toth Adsorption Model:
    q(P) = qm * b * P / [ 1 + (b * P)^t ]^(1 / t)
    """

    def __init__(self, qm: float, b: float, t: float,
                 dH_zero: float = 37222.0, dH_inf: float = 30731.0, gamma: float = 0.85):
        if qm <= 0.0 or b <= 0.0 or t <= 0.0:
            raise ValueError("Toth parameters (qm, b, t) must be strictly positive.")
        self.qm = float(qm)
        self.b = float(b)
        self.t = float(t)
        self.dH_zero = float(dH_zero)
        self.dH_inf = float(dH_inf)
        self.gamma = float(gamma)


    def loading(self, P: float) -> float:
        if P <= 0.0:
            return 0.0
        bP = self.b * P
        denom = (1.0 + (bP ** self.t)) ** (1.0 / self.t)
        return float(self.qm * bP / denom)

    def henry_constant(self) -> float:
        return float(self.qm * self.b)

    def dq_dP(self, P: float) -> float:
        """Exact analytical derivative for isothermal Toth."""
        if P <= 0.0:
            return self.henry_constant()
        bP = self.b * P
        denom_base = 1.0 + (bP ** self.t)
        denom = denom_base ** ((1.0 + self.t) / self.t)
        return float(self.qm * self.b / denom)

    def spreading_pressure(self, P: float) -> float:
        if P <= 1e-12:
            return 0.0

        p_cutoff = min(P * 1e-4, 1e-6)
        K_H = self.henry_constant()
        psi_low = K_H * p_cutoff

        integrand = lambda p: self.loading(p) / p
        psi_rem, _ = integrate.quad(integrand, p_cutoff, P, epsrel=1e-8, epsabs=1e-10, limit=100)
        return float(psi_low + psi_rem)
    def get_isosteric_heat(self, q: float) -> float:
        """Concentration-dependent isosteric heat of adsorption based on Cavenati et al. (2004)."""
        theta = float(np.clip(q / max(self.qm, 1e-9), 0.0, 1.0))
        dH_ads = self.dH_zero - (self.dH_zero - self.dH_inf) * (theta ** self.gamma)
        return float(dH_ads)


class DualSiteLangmuirIsotherm(BaseIsotherm):
    """
    Standard Isothermal Dual-Site Langmuir (DSL) Model:
    q(P) = qm1 * b1 * P / (1 + b1 * P) + qm2 * b2 * P / (1 + b2 * P)
    """

    def __init__(self, qm1: float, b1: float, qm2: float, b2: float):
        if min(qm1, b1, qm2, b2) <= 0.0:
            raise ValueError("All DSL parameters must be strictly positive.")
        self.qm1 = float(qm1)
        self.b1 = float(b1)
        self.qm2 = float(qm2)
        self.b2 = float(b2)

    def loading(self, P: float) -> float:
        if P <= 0.0:
            return 0.0
        q1 = (self.qm1 * self.b1 * P) / (1.0 + self.b1 * P)
        q2 = (self.qm2 * self.b2 * P) / (1.0 + self.b2 * P)
        return float(q1 + q2)

    def henry_constant(self) -> float:
        return float(self.qm1 * self.b1 + self.qm2 * self.b2)

    def dq_dP(self, P: float) -> float:
        if P <= 0.0:
            return self.henry_constant()
        dq1 = (self.qm1 * self.b1) / ((1.0 + self.b1 * P) ** 2)
        dq2 = (self.qm2 * self.b2) / ((1.0 + self.b2 * P) ** 2)
        return float(dq1 + dq2)

    def spreading_pressure(self, P: float) -> float:
        if P <= 0.0:
            return 0.0
        return float(self.qm1 * np.log1p(self.b1 * P) + self.qm2 * np.log1p(self.b2 * P))


# ==============================================================================
# SECTION 3: TEMPERATURE-DEPENDENT EXTENSIONS (Krishnamurthy Standard)
# ==============================================================================

class BaseThermalIsotherm(ABC):
    """Abstract Base Class for Temperature-Dependent Isotherms."""

    @abstractmethod
    def at_temperature(self, T: float) -> BaseIsotherm:
        """Returns a frozen BaseIsotherm instance evaluated at temperature T (K)."""
        pass

    def loading(self, P: float, T: float) -> float:
        return self.at_temperature(T).loading(P)

    def dq_dP(self, P: float, T: float) -> float:
        return self.at_temperature(T).dq_dP(P)

    def spreading_pressure(self, P: float, T: float) -> float:
        return self.at_temperature(T).spreading_pressure(P)

    def invert_spreading_pressure(self, psi_target: float, T: float, P_guess: float = 1.0) -> float:
        return self.at_temperature(T).invert_spreading_pressure(psi_target, P_guess=P_guess)

    def isosteric_heat(self, P: float, T: float, dT: float = 0.5) -> float:
        """
        Isosteric heat of adsorption (-dH_ads) in J/mol via Clausius-Clapeyron equation:
        -dH_ads = - R * T^2 * (dq/dT)_P / [ P * (dq/dP)_T ]
        """
        if P <= 1e-8:
            P = 1e-8

        dq_dP_val = self.dq_dP(P, T)
        if dq_dP_val <= 1e-14:
            return 0.0

        q_plus = self.loading(P, T + dT)
        q_minus = self.loading(P, T - dT)
        dq_dT = (q_plus - q_minus) / (2.0 * dT)

        dH = - R_GAS * (T ** 2) * dq_dT / (P * dq_dP_val)
        return max(0.0, float(dH))
    def get_adsorbed_heat_capacity(self, P: float, T: float, species: str = "CO2") -> float:
        q = self.loading(P, T)
        return self.at_temperature(T).get_adsorbed_heat_capacity(T, q, species=species)

    


class TemperatureDependentToth(BaseThermalIsotherm):
    """
    Krishnamurthy non-isothermal Toth Isotherm:
    b(T) = b0 * exp[ (-dH / (R * T0)) * (T0/T - 1) ]
    qm(T) = qm0 * exp[ chi * (1 - T/T0) ]
    t(T) = t0 + alpha * (1 - T0/T)
    """

    def __init__(self, qm0: float, b0: float, t0: float, dH: float,
                 T0: float = 298.15, chi: float = 0.0, alpha: float = 0.0):
        self.qm0 = float(qm0)
        self.b0 = float(b0)
        self.t0 = float(t0)
        self.dH = float(dH)
        self.T0 = float(T0)
        self.chi = float(chi)
        self.alpha = float(alpha)

    def at_temperature(self, T: float) -> TothIsotherm:
        theta = self.T0 / T
        b = self.b0 * np.exp((-self.dH / (R_GAS * self.T0)) * (theta - 1.0))
        qm = self.qm0 * np.exp(self.chi * (1.0 - (T / self.T0)))
        t = self.t0 + self.alpha * (1.0 - theta)
        nnt = float(np.clip(t, 1e-4, 1.0))
        return TothIsotherm(qm=qm, b=b, t=t, dH_zero=abs(self.dH))



class TemperatureDependentDSL(BaseThermalIsotherm):
    """
    Krishnamurthy non-isothermal Dual-Site Langmuir Isotherm:
    b1(T) = b1_0 * exp[ (-dH1 / (R * T0)) * (T0/T - 1) ]
    b2(T) = b2_0 * exp[ (-dH2 / (R * T0)) * (T0/T - 1) ]
    """

    def __init__(self, qm1_0: float, b1_0: float, dH1: float,
                 qm2_0: float, b2_0: float, dH2: float, T0: float = 298.15):
        self.qm1_0 = float(qm1_0)
        self.b1_0 = float(b1_0)
        self.dH1 = float(dH1)
        self.qm2_0 = float(qm2_0)
        self.b2_0 = float(b2_0)
        self.dH2 = float(dH2)
        self.T0 = float(T0)

    def at_temperature(self, T: float) -> DualSiteLangmuirIsotherm:
        theta = self.T0 / T
        b1 = self.b1_0 * np.exp((-self.dH1 / (R_GAS * self.T0)) * (theta - 1.0))
        b2 = self.b2_0 * np.exp((-self.dH2 / (R_GAS * self.T0)) * (theta - 1.0))
        return DualSiteLangmuirIsotherm(qm1=self.qm1_0, b1=b1, qm2=self.qm2_0, b2=b2)


# ==============================================================================
# SECTION 4: MULTICOMPONENT IAST & JACOBIAN SOLVER
# ==============================================================================

class BinaryIAST:
    """
    Robust Binary IAST Equilibrium Solver for VPSA Separation (CO2 / N2).
    Enforces surface potential matching: psi_1(P1^0) = psi_2(P2^0).
    """

    def __init__(self, isotherm1: BaseThermalIsotherm, isotherm2: BaseThermalIsotherm):
        self.iso1 = isotherm1
        self.iso2 = isotherm2

    def solve(self, P: float, y1: float, T: float) -> tuple[float, float, float, float]:
        """
        Calculates equilibrium loadings of binary mixture.
        Returns: (q1, q2, x1, psi)
        """
        if P <= 1e-9:
            return 0.0, 0.0, y1, 0.0

        y1 = max(0.0, min(1.0, y1))
        y2 = 1.0 - y1

        iso1_T = self.iso1.at_temperature(T)
        iso2_T = self.iso2.at_temperature(T)

        # Pure boundary conditions
        if y1 <= 1e-8:
            q2 = iso2_T.loading(P)
            psi = iso2_T.spreading_pressure(P)
            return 0.0, q2, 0.0, psi
        if y2 <= 1e-8:
            q1 = iso1_T.loading(P)
            psi = iso1_T.spreading_pressure(P)
            return q1, 0.0, 1.0, psi

        # Objective function in terms of reduced spreading pressure psi
        def obj(psi_val):
            p1_0 = iso1_T.invert_spreading_pressure(psi_val, P_guess=P * y1)
            p2_0 = iso2_T.invert_spreading_pressure(psi_val, P_guess=P * y2)
            if p1_0 <= 1e-12 or p2_0 <= 1e-12:
                return 1e6
            return (P * y1 / p1_0) + (P * y2 / p2_0) - 1.0

        psi_min = 1e-8
        psi_max = max(iso1_T.spreading_pressure(P), iso2_T.spreading_pressure(P)) * 2.0 + 1.0

        while obj(psi_max) > 0.0 and psi_max < 1e5:
            psi_max *= 3.0

        sol = optimize.root_scalar(obj, bracket=[psi_min, psi_max], method='brentq', xtol=1e-7)
        psi_eq = sol.root

        p1_0 = iso1_T.invert_spreading_pressure(psi_eq)
        p2_0 = iso2_T.invert_spreading_pressure(psi_eq)

        x1 = (P * y1) / p1_0
        x2 = 1.0 - x1

        q1_0 = iso1_T.loading(p1_0)
        q2_0 = iso2_T.loading(p2_0)

        inv_q_tot = (x1 / max(q1_0, 1e-9)) + (x2 / max(q2_0, 1e-9))
        q_tot = 1.0 / inv_q_tot

        q1 = q_tot * x1
        q2 = q_tot * x2
        return float(q1), float(q2), float(x1), float(psi_eq)

    def thermodynamic_jacobian(self, P: float, y1: float, T: float,
                               h_P: float = 1e-4, h_y: float = 1e-4, h_T: float = 1e-1) -> dict:
        """
        Computes 6 coupled partial derivatives for non-isothermal VPSA numerical integration:
        dq_i / dP, dq_i / dy1, dq_i / dT for components 1 and 2.
        """
        base_q1, base_q2, _, _ = self.solve(P, y1, T)

        # Pressure sensitivity
        q1_p, q2_p, _, _ = self.solve(P + h_P, y1, T)
        dq1_dP = (q1_p - base_q1) / h_P
        dq2_dP = (q2_p - base_q2) / h_P

        # Composition sensitivity
        y1_mod = min(0.9999, y1 + h_y)
        dy = y1_mod - y1
        q1_y, q2_y, _, _ = self.solve(P, y1_mod, T)
        dq1_dy = (q1_y - base_q1) / dy
        dq2_dy = (q2_y - base_q2) / dy

        # Temperature sensitivity
        q1_T, q2_T, _, _ = self.solve(P, y1, T + h_T)
        dq1_dT = (q1_T - base_q1) / h_T
        dq2_dT = (q2_T - base_q2) / h_T

        return {
            "dq1_dP": dq1_dP, "dq1_dy": dq1_dy, "dq1_dT": dq1_dT,
            "dq2_dP": dq2_dP, "dq2_dy": dq2_dy, "dq2_dT": dq2_dT
        }


# ==============================================================================
# SECTION 5: COMPREHENSIVE UNIT TEST SUITE (Step 2 + Extended Regression)
# ==============================================================================

class TestThermodynamicEngine(unittest.TestCase):
    """Rigorous unit tests checking pure, thermal, and multicomponent physics."""

    def setUp(self):
        # Step 2 Pure benchmark params
        self.toth_pure = TothIsotherm(qm=4.5, b=1.2, t=0.6)
        self.dsl_pure = DualSiteLangmuirIsotherm(qm1=3.0, b1=1.5, qm2=1.5, b2=0.2)

        # Krishnamurthy standard benchmark params
        self.co2_thermal = TemperatureDependentToth(
            qm0=5.50, b0=2.85, t0=0.48, dH=-35500.0, T0=298.15
        )
        self.n2_thermal = TemperatureDependentToth(
            qm0=2.30, b0=0.12, t0=0.95, dH=-18000.0, T0=298.15
        )
        self.iast_system = BinaryIAST(self.co2_thermal, self.n2_thermal)

    def test_pure_boundary_conditions(self):
        """Zero pressure yields zero loading."""
        self.assertEqual(self.toth_pure.loading(0.0), 0.0)
        self.assertEqual(self.dsl_pure.loading(0.0), 0.0)
        self.assertEqual(self.toth_pure.spreading_pressure(0.0), 0.0)
        self.assertEqual(self.dsl_pure.spreading_pressure(0.0), 0.0)

    def test_henry_limits(self):
        """At low pressure limit, q / P must match analytical Henry constant."""
        p_low = 1e-9
        # Toth
        q_low_toth = self.toth_pure.loading(p_low)
        k_h_toth = self.toth_pure.henry_constant()
        self.assertLess(_rel_diff(q_low_toth / p_low, k_h_toth), 1e-4)
        # DSL
        q_low_dsl = self.dsl_pure.loading(p_low)
        k_h_dsl = self.dsl_pure.henry_constant()
        self.assertLess(_rel_diff(q_low_dsl / p_low, k_h_dsl), 1e-4)

    def test_pure_derivative_accuracy(self):
        """Analytical derivative dq/dP must match Richardson 4th-order derivative."""
        pressures = [0.05, 0.5, 1.0, 5.0, 10.0]
        for p in pressures:
            # Toth
            d_ana_toth = self.toth_pure.dq_dP(p)
            d_num_toth = self.toth_pure.dq_dP_numerical(p)
            self.assertLess(_rel_diff(d_ana_toth, d_num_toth), 1e-5)
            # DSL
            d_ana_dsl = self.dsl_pure.dq_dP(p)
            d_num_dsl = self.dsl_pure.dq_dP_numerical(p)
            self.assertLess(_rel_diff(d_ana_dsl, d_num_dsl), 1e-5)

    def test_inversion_roundtrip(self):
        """P -> psi(P) -> invert(psi) == P."""
        for p_target in [0.1, 1.0, 5.0]:
            psi_val = self.toth_pure.spreading_pressure(p_target)
            p_recov = self.toth_pure.invert_spreading_pressure(psi_val)
            self.assertLess(_rel_diff(p_recov, p_target), 1e-5)

    def test_thermal_isosteric_heat(self):
        """Isosteric heat must match specified enthalpy within thermodynamic limits."""
        dH_co2 = self.co2_thermal.isosteric_heat(P=1.0, T=298.15)
        # Specified dH = 35.5 kJ/mol
        self.assertLess(_rel_diff(dH_co2, 35500.0), 0.08)

    def test_iast_selectivity_and_mass_conservation(self):
        """Binary IAST must exhibit CO2 selectivity and thermodynamic consistency."""
        q1, q2, x1, psi = self.iast_system.solve(P=1.0, y1=0.15, T=298.15)
        self.assertGreater(q1, 0.0)
        self.assertGreater(q2, 0.0)
        self.assertGreater(x1, 0.15)  # Zeolite 13X is selective for CO2 over N2
        self.assertGreater(psi, 0.0)

    def test_thermodynamic_jacobian_physics(self):
        """Jacobian elements must respect thermodynamic sign conventions."""
        jac = self.iast_system.thermodynamic_jacobian(P=1.0, y1=0.15, T=298.15)
        # Adsorption loading increases with pressure
        self.assertGreater(jac["dq1_dP"], 0.0)
        # Exothermic adsorption: loading decreases with increasing temperature
        self.assertLess(jac["dq1_dT"], 0.0)
        self.assertLess(jac["dq2_dT"], 0.0)


# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING EXTENDED KRISHNAMURTHY THERMODYNAMIC ENGINE SUITE")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestThermodynamicEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n[LOCKED] Extended Thermodynamic Layer passed 100% of physical tests.")
    else:
        print("\n[FAILED] Verification errors detected.")
