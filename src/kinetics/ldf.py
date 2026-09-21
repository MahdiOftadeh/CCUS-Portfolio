from typing import Dict, Union
import numpy as np
from src.kinetics.base import BaseKinetics

class LinearDrivingForce(BaseKinetics):
    """Linear Driving Force (LDF) Mass Transfer Kinetics Model."""

    def __init__(self, k_ldf: Dict[str, float]):
        self.k_ldf = k_ldf
        self.species = list(k_ldf.keys())

    def calculate_rate(
        self,
        q_equilibrium: Union[Dict[str, float], np.ndarray],
        q_actual: Union[Dict[str, float], np.ndarray]
    ) -> Union[Dict[str, float], np.ndarray]:
        if isinstance(q_equilibrium, dict) and isinstance(q_actual, dict):
            rates = {}
            for comp in self.species:
                q_eq = q_equilibrium.get(comp, 0.0)
                q_act = q_actual.get(comp, 0.0)
                k = self.k_ldf.get(comp, 0.0)
                rates[comp] = k * (q_eq - q_act)
            return rates
        elif isinstance(q_equilibrium, np.ndarray) and isinstance(q_actual, np.ndarray):
            k_array = np.array([self.k_ldf[comp] for comp in self.species])
            return k_array * (q_equilibrium - q_actual)
        else:
            raise TypeError("q_equilibrium and q_actual must both be dicts or numpy arrays.")
