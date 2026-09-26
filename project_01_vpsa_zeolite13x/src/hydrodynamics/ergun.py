# src/hydrodynamics/ergun.py
from typing import Dict, Any

class ErgunEquation:
    """
    Ergun equation for pressure drop calculation across a packed bed.
    """
    def __init__(self, bed_porosity: float, particle_diameter: float):
        """
        Parameters:
        -----------
        bed_porosity : float (e.g. 0.37)
            Void fraction of the bed (eps_b) [-]
        particle_diameter : float (e.g. 0.002)
            Mean diameter of adsorbent particles (d_p) [m]
        """
        self.eps_b = bed_porosity
        self.d_p = particle_diameter

    def calculate_pressure_gradient(self, velocity: float, fluid_density: float, fluid_viscosity: float) -> float:
        """
        Calculates dP/dz [Pa/m].

        Parameters:
        -----------
        velocity : float
            Superficial gas velocity u [m/s]
        fluid_density : float
            Gas mixture density rho_g [kg/m^3]
        fluid_viscosity : float
            Gas mixture dynamic viscosity mu [Pa.s]

        Returns:
        --------
        float
            Pressure gradient dP/dz [Pa/m]
        """
        eps = self.eps_b
        dp = self.d_p
        
        # Viscous term
        term_viscous = 150.0 * ((1.0 - eps)**2 / (eps**3)) * (fluid_viscosity * velocity / (dp**2))
        # Inertial term
        term_inertial = 1.75 * ((1.0 - eps) / (eps**3)) * (fluid_density * velocity * abs(velocity) / dp)
        
        return -(term_viscous + term_inertial)
