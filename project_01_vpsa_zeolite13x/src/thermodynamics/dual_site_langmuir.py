"""
Dual-Site Langmuir (DSL) Isotherm implementation for multi-component adsorption.
"""

from dataclasses import dataclass
import numpy as np
from src.thermodynamics.base import BaseIsotherm, R_GAS


@dataclass
class DSLParameters:
    """Parameters for a single gas component on a dual-site adsorbent."""
    name: str
    q_sat1: float      # Site 1 saturation capacity [mol/kg]
    q_sat2: float      # Site 2 saturation capacity [mol/kg]
    b0_1: float        # Site 1 affinity constant at T_ref [1/bar]
    b0_2: float        # Site 2 affinity constant at T_ref [1/bar]
    dH_1: float        # Heat of adsorption Site 1 [J/mol] (negative value)
    dH_2: float        # Heat of adsorption Site 2 [J/mol] (negative value)
    T_ref: float = 298.15  # Reference temperature [K]

    def affinity(self, T: float) -> tuple[float, float]:
        """Calculates temperature-dependent affinity constants b1 and b2."""
        factor = (self.T_ref / T) - 1.0
        b1 = self.b0_1 * np.exp((-self.dH_1 / (R_GAS * self.T_ref)) * factor)
        b2 = self.b0_2 * np.exp((-self.dH_2 / (R_GAS * self.T_ref)) * factor)
        return b1, b2


class DualSiteLangmuir(BaseIsotherm):
    """Multi-component competitive Dual-Site Langmuir isotherm."""

    def __init__(self, components: list[DSLParameters]):
        self.components = components

    def loading(
        self, partial_pressures: np.ndarray, temperature: float
    ) -> np.ndarray:
        """
        Calculates equilibrium loadings q* [mol/kg] for all components.
        """
        n_comp = len(self.components)
        p = np.asarray(partial_pressures, dtype=float)

        b1_list, b2_list = [], []
        q1_list, q2_list = [], []

        for comp in self.components:
            b1, b2 = comp.affinity(temperature)
            b1_list.append(b1)
            b2_list.append(b2)
            q1_list.append(comp.q_sat1)
            q2_list.append(comp.q_sat2)

        b1_arr = np.array(b1_list)
        b2_arr = np.array(b2_list)
        q1_arr = np.array(q1_list)
        q2_arr = np.array(q2_list)

        if p.ndim == 1:
            denom1 = 1.0 + np.sum(b1_arr * p)
            denom2 = 1.0 + np.sum(b2_arr * p)
            return (q1_arr * b1_arr * p) / denom1 + (q2_arr * b2_arr * p) / denom2
        else:
            b1_exp = b1_arr[:, np.newaxis]
            b2_exp = b2_arr[:, np.newaxis]
            q1_exp = q1_arr[:, np.newaxis]
            q2_exp = q2_arr[:, np.newaxis]

            denom1 = 1.0 + np.sum(b1_exp * p, axis=0, keepdims=True)
            denom2 = 1.0 + np.sum(b2_exp * p, axis=0, keepdims=True)
            return (q1_exp * b1_exp * p) / denom1 + (q2_exp * b2_exp * p) / denom2
