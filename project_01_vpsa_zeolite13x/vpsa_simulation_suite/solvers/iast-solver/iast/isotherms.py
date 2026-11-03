"""Pure-component adsorption isotherms.

Unit convention (all isotherms):
    P: bar
    q: mol/kg
    T: K
    R: J/(mol.K)

Spreading pressure:
    psi(P) = integral_0^P q(P')/P' dP'   [mol/kg]
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy import integrate, optimize

R_GAS = 8.314462618  # J/(mol.K)


# ============================================================================
# Base classes
# ============================================================================

class BaseIsotherm(ABC):
    """Abstract base class for isothermal pure-component isotherms."""

    @abstractmethod
    def loading(self, P: float) -> float:
        """Equilibrium loading q(P) [mol/kg]."""

    @abstractmethod
    def henry_constant(self) -> float:
        """Henry constant K_H = lim_{P->0} q/P [mol/(kg.bar)]."""

    @abstractmethod
    def spreading_pressure(self, P: float) -> float:
        """Reduced spreading pressure psi(P) [mol/kg]."""

    @abstractmethod
    def dq_dP(self, P: float) -> float:
        """Analytical derivative dq/dP [mol/(kg.bar)]."""

    def invert_spreading_pressure(self, psi_target: float,
                                  P_guess: float = 1.0) -> float:
        """Find pure-component pressure P0 such that psi(P0) = psi_target."""
        if psi_target <= 0.0:
            return 0.0
        K_H = self.henry_constant()
        if psi_target < 1e-20:
            return float(psi_target / K_H)
        P_lo = max(psi_target / (10.0 * K_H), 1e-30)
        P_hi = max(P_guess * 2.0, (psi_target / K_H) * 2.0, 1.0)
        while self.spreading_pressure(P_hi) < psi_target and P_hi < 1e8:
            P_hi *= 4.0
        return float(optimize.brentq(
            lambda P: self.spreading_pressure(P) - psi_target,
            P_lo, P_hi, xtol=1e-12, rtol=1e-12, maxiter=200,
        ))


class BaseThermalIsotherm(ABC):
    """Temperature-dependent isotherm (van't Hoff on affinity b)."""

    @abstractmethod
    def at_temperature(self, T: float) -> BaseIsotherm:
        """Return the frozen BaseIsotherm at T."""

    def loading(self, P, T):
        return self.at_temperature(T).loading(P)

    def spreading_pressure(self, P, T):
        return self.at_temperature(T).spreading_pressure(P)

    def dq_dP(self, P, T):
        return self.at_temperature(T).dq_dP(P)

    def dq_dT(self, P, T, h=0.5):
        """Central-difference dq/dT at constant P."""
        return (self.loading(P, T + h) - self.loading(P, T - h)) / (2.0 * h)

    def isosteric_heat(self, P, T, h=0.5):
        """q_st = -R*T^2/P * (dq/dT) / (dq/dP) via Clausius-Clapeyron."""
        dqP = self.dq_dP(P, T)
        if dqP <= 1e-14 or P <= 0.0:
            return 0.0
        dqT = self.dq_dT(P, T, h)
        qst = -R_GAS * T * T / P * dqT / dqP
        return float(max(0.0, qst))


# ============================================================================
# Isothermal models
# ============================================================================

class Langmuir(BaseIsotherm):
    """Langmuir: q = qm*b*P / (1 + b*P)."""

    def __init__(self, qm: float, b: float):
        if qm <= 0.0 or b <= 0.0:
            raise ValueError("qm and b must be strictly positive")
        self.qm = float(qm)
        self.b = float(b)

    def loading(self, P):
        if P <= 0.0:
            return 0.0
        return self.qm * self.b * P / (1.0 + self.b * P)

    def henry_constant(self):
        return self.qm * self.b

    def spreading_pressure(self, P):
        if P <= 0.0:
            return 0.0
        return self.qm * np.log1p(self.b * P)

    def dq_dP(self, P):
        if P <= 0.0:
            return self.henry_constant()
        return self.qm * self.b / (1.0 + self.b * P) ** 2


class DSL(BaseIsotherm):
    """Dual-Site Langmuir: sum of two Langmuir terms."""

    def __init__(self, qm1: float, b1: float, qm2: float, b2: float):
        if min(qm1, b1, qm2, b2) <= 0.0:
            raise ValueError("All DSL parameters must be strictly positive")
        self.qm1, self.b1 = float(qm1), float(b1)
        self.qm2, self.b2 = float(qm2), float(b2)

    def loading(self, P):
        if P <= 0.0:
            return 0.0
        q1 = self.qm1 * self.b1 * P / (1.0 + self.b1 * P)
        q2 = self.qm2 * self.b2 * P / (1.0 + self.b2 * P)
        return q1 + q2

    def henry_constant(self):
        return self.qm1 * self.b1 + self.qm2 * self.b2

    def spreading_pressure(self, P):
        if P <= 0.0:
            return 0.0
        return (self.qm1 * np.log1p(self.b1 * P)
                + self.qm2 * np.log1p(self.b2 * P))

    def dq_dP(self, P):
        if P <= 0.0:
            return self.henry_constant()
        d1 = self.qm1 * self.b1 / (1.0 + self.b1 * P) ** 2
        d2 = self.qm2 * self.b2 / (1.0 + self.b2 * P) ** 2
        return d1 + d2


class Toth(BaseIsotherm):
    """Toth: q = qm*b*P / [1 + (b*P)^t]^(1/t)."""

    def __init__(self, qm: float, b: float, t: float):
        if qm <= 0.0 or b <= 0.0 or t <= 0.0:
            raise ValueError("qm, b, t must be strictly positive")
        self.qm = float(qm)
        self.b = float(b)
        self.t = float(t)

    def loading(self, P):
        if P <= 0.0:
            return 0.0
        bP = self.b * P
        u = bP ** self.t
        return self.qm * bP / (1.0 + u) ** (1.0 / self.t)

    def henry_constant(self):
        return self.qm * self.b

    def dq_dP(self, P):
        if P <= 0.0:
            return self.henry_constant()
        bP = self.b * P
        u = bP ** self.t
        return self.qm * self.b / (1.0 + u) ** ((1.0 + self.t) / self.t)

    def spreading_pressure(self, P):
        if P <= 0.0:
            return 0.0
        p_cut = min(P * 1e-6, 1e-8)
        psi_low = self.henry_constant() * p_cut
        psi_rem, _ = integrate.quad(
            lambda p: self.loading(p) / p,
            p_cut, P, epsabs=1e-12, epsrel=1e-10, limit=200,
        )
        return float(psi_low + psi_rem)


# ============================================================================
# Temperature-dependent wrappers (van't Hoff on b)
# ============================================================================
# Convention:
#   b(T) = b0 * exp[ Q/R * (1/T - 1/T_ref) ]
#   where Q = -dH_ads > 0 for exothermic adsorption.

class LangmuirThermal(BaseThermalIsotherm):
    """Langmuir with van't Hoff temperature dependence on b."""

    def __init__(self, qm: float, b0: float, Q: float, T_ref: float = 298.15):
        if qm <= 0.0 or b0 <= 0.0 or T_ref <= 0.0:
            raise ValueError("qm, b0, T_ref must be positive")
        self.qm = float(qm)
        self.b0 = float(b0)
        self.Q = float(Q)
        self.T_ref = float(T_ref)

    def at_temperature(self, T):
        b = self.b0 * np.exp(self.Q / R_GAS * (1.0 / T - 1.0 / self.T_ref))
        return Langmuir(self.qm, b)


class DSLThermal(BaseThermalIsotherm):
    """DSL with van't Hoff on each site's affinity."""

    def __init__(self, qm1, b1_0, Q1, qm2, b2_0, Q2, T_ref=298.15):
        if min(qm1, b1_0, qm2, b2_0, T_ref) <= 0.0:
            raise ValueError("qm, b0, T_ref must be positive")
        self.qm1, self.b1_0, self.Q1 = float(qm1), float(b1_0), float(Q1)
        self.qm2, self.b2_0, self.Q2 = float(qm2), float(b2_0), float(Q2)
        self.T_ref = float(T_ref)

    def at_temperature(self, T):
        b1 = self.b1_0 * np.exp(self.Q1 / R_GAS * (1.0 / T - 1.0 / self.T_ref))
        b2 = self.b2_0 * np.exp(self.Q2 / R_GAS * (1.0 / T - 1.0 / self.T_ref))
        return DSL(self.qm1, b1, self.qm2, b2)


class TothThermal(BaseThermalIsotherm):
    """Toth with van't Hoff on b (t and qm kept constant)."""

    def __init__(self, qm: float, b0: float, t: float, Q: float,
                 T_ref: float = 298.15):
        if qm <= 0.0 or b0 <= 0.0 or t <= 0.0 or T_ref <= 0.0:
            raise ValueError("qm, b0, t, T_ref must be positive")
        self.qm = float(qm)
        self.b0 = float(b0)
        self.t = float(t)
        self.Q = float(Q)
        self.T_ref = float(T_ref)

    def at_temperature(self, T):
        b = self.b0 * np.exp(self.Q / R_GAS * (1.0 / T - 1.0 / self.T_ref))
        return Toth(self.qm, b, self.t)