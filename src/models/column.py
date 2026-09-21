# src/models/column.py
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any

from src.solvers.spatial_discretization import FiniteDifference1D
from src.kinetics.ldf import LinearDrivingForce
from src.thermodynamics.dual_site_langmuir import DualSiteLangmuir



class AdsorptionColumn1D:
    """
    1D Dynamic Packed Bed Adsorption Column Model.
    Solves mass balance coupled with LDF kinetics and equilibrium isotherms.
    """
    def __init__(
        self,
        length: float,
        diameter: float,
        bed_porosity: float,
        particle_density: float,
        num_nodes: int,
        isotherm: DualSiteLangmuir,
        kinetics: LinearDrivingForce,
    ):
        self.L = length
        self.D = diameter
        self.eps_b = bed_porosity
        self.rho_s = particle_density
        self.N = num_nodes
        
        self.isotherm = isotherm
        self.kinetics = kinetics
        self.grid = FiniteDifference1D(bed_length=length, num_nodes=num_nodes)
        
        # Cross-sectional area
        self.area = np.pi * (diameter / 2.0)**2

    def _ode_system(self, t: float, y: np.ndarray, u_feed: float, c_inlet: float, T: float, P: float) -> np.ndarray:
        """
        Right-hand side of ODEs: dy/dt = f(t, y)
        y vector layout:
        y[0 : N]     = C (Gas phase concentration along bed, mol/m3)
        y[N : 2*N]   = q (Solid phase loading along bed, mol/kg)
        """
        N = self.N
        C = y[0:N]
        q = y[N:2*N]

        # 1. Calculate equilibrium loading q* at local gas concentration C
        # Ideal gas law assumption for partial pressure: P_i = C_i * R * T
        R_gas = 8.314  # J/(mol.K)
        p_co2 = C * R_gas * T  # Pa
        
        # Local equilibrium loading q*
        q_star = self.isotherm.loading(pressure=p_co2, temperature=T)

        # 2. Kinetic mass transfer rate: dq/dt = k_ldf * (q* - q)
        dq_dt = self.kinetics.rate(q_equilibrium=q_star, q_current=q)

        # 3. Gas phase advective spatial derivative: d(u*C)/dz
        # Enforce inlet boundary condition
        C_with_bc = C.copy()
        C_with_bc[0] = c_inlet
        dC_dz = self.grid.first_derivative_upwind(field=C_with_bc, velocity=u_feed)

        # 4. Gas phase mass balance:
        # eps_b * dC/dt = - u * dC/dz - (1 - eps_b) * rho_s * dq/dt
        dC_dt = (- u_feed * dC_dz - (1.0 - self.eps_b) * self.rho_s * dq_dt) / self.eps_b

        # Pack derivatives
        return np.concatenate([dC_dt, dq_dt])

    def simulate_adsorption_step(
        self,
        duration: float,
        u_feed: float,
        c_inlet: float,
        temperature: float,
        pressure: float,
        initial_C: np.ndarray = None,
        initial_q: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Simulates an adsorption step (breakthrough curve).
        """
        N = self.N
        if initial_C is None:
            initial_C = np.zeros(N)
        if initial_q is None:
            initial_q = np.zeros(N)

        y0 = np.concatenate([initial_C, initial_q])
        t_span = (0.0, duration)
        t_eval = np.linspace(0.0, duration, 100)

        # Solve system of ODEs
        sol = solve_ivp(
            fun=lambda t, y: self._ode_system(t, y, u_feed, c_inlet, temperature, pressure),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='Radau'  # Stiff solver for fast kinetics/steep breakthrough fronts
        )

        return {
            "time": sol.t,
            "z_grid": self.grid.z_grid,
            "C_history": sol.y[0:N, :],         # shape: (N, num_time_steps)
            "q_history": sol.y[N:2*N, :],       # shape: (N, num_time_steps)
            "breakthrough_C": sol.y[N - 1, :]   # Outlet concentration over time
        }
