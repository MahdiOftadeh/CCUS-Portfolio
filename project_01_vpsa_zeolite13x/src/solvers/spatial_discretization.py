# src/solvers/spatial_discretization.py
import numpy as np

class FiniteDifference1D:
    """
    1D Upwind Finite Difference Discretization for Advection-Dispersion equations.
    """
    def __init__(self, bed_length: float, num_nodes: int):
        self.L = bed_length
        self.N = num_nodes
        self.dz = bed_length / (num_nodes - 1)
        self.z_grid = np.linspace(0, bed_length, num_nodes)

    def first_derivative_upwind(self, field: np.ndarray, velocity: float) -> np.ndarray:
        """
        Calculates df/dz using 1st-order Upwind scheme.
        """
        df_dz = np.zeros_like(field)
        if velocity >= 0:
            # Forward flow (left to right)
            df_dz[1:] = (field[1:] - field[:-1]) / self.dz
            df_dz[0] = df_dz[1] # Boundary node approximation
        else:
            # Backward flow (right to left, e.g., during evacuation/blowdown)
            df_dz[:-1] = (field[1:] - field[:-1]) / self.dz
            df_dz[-1] = df_dz[-2]
            
        return df_dz
