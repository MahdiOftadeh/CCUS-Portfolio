from abc import ABC, abstractmethod
from typing import Dict, Union
import numpy as np

class BaseKinetics(ABC):
    """Abstract Base Class for Adsorption Kinetics Models."""
    
    @abstractmethod
    def calculate_rate(
        self,
        q_equilibrium: Union[Dict[str, float], np.ndarray],
        q_actual: Union[Dict[str, float], np.ndarray]
    ) -> Union[Dict[str, float], np.ndarray]:
        pass
