"""
Abstract base class for all multi-component adsorption isotherms.
"""

from abc import ABC, abstractmethod
import numpy as np

# Universal Gas Constant [J / (mol * K)]
R_GAS = 8.314462


class BaseIsotherm(ABC):
    """
    Abstract interface for thermodynamic equilibrium models.
    All isotherm implementations (DSL, Toth, Sips, etc.) must inherit from this class.
    """

    @abstractmethod
    def loading(
        self, partial_pressures: np.ndarray, temperature: float
    ) -> np.ndarray:
        """
        Compute equilibrium solid-phase loading (q*) for all components.

        Parameters:
        -----------
        partial_pressures : np.ndarray
            Partial pressures [bar] of shape (n_components, ...)
        temperature : float
            Bed temperature [K]

        Returns:
        --------
        q_star : np.ndarray
            Equilibrium loadings [mol/kg]
        """
        pass
