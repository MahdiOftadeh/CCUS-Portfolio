"""
1D Dynamic Model for Adsorption Column Simulation Module.
Developed based on VSA/TVSA process modeling and gas-solid mass balance.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from src.thermodynamics.dual_site_langmuir import DualSiteLangmuir
from src.kinetics.ldf import LinearDrivingForce
from src.hydrodynamics.ergun import ErgunEquation

# Universal Gas Constant [J / (mol * K)]
R_GAS = 8.314462618


class AdsorptionColumn1D:
    """
    Dynamic 1D Model for Fixed-Bed Adsorption Column.

    Governing Equations:
    1. Gas phase mass balance (convection + mass transfer to solid)
    2. Solid phase adsorption kinetics (Linear Driving Force - LDF)
    3. Gas-solid thermodynamic equilibrium (Dual-Site Langmuir Isotherm)
    4. Hydrodynamic pressure drop across the bed (Ergun Equation)
    """

    def __init__(
        self,
        length: float,
        diameter: float,
        voidage: float,
        bulk_density: float,
        n_nodes: int,
        isotherm: DualSiteLangmuir,
        kinetics: LinearDrivingForce,
        hydrodynamics: ErgunEquation,
    ):
        """
        Initializes the column's geometric and physical parameters.

        Parameters:
        -----------
        length : float
            Column length [m]
        diameter : float
            Column inner diameter [m]
        voidage : float
            Bed porosity (Void Fraction) [-]
        bulk_density : float
            Bed bulk density [kg/m^3]
        n_nodes : int
            Number of spatial discretization nodes along the column length
        isotherm : DualSiteLangmuir
            Thermodynamic equilibrium model instance
        kinetics : LinearDrivingForce
            Mass transfer kinetics model instance
        hydrodynamics : ErgunEquation
            Hydrodynamic pressure drop model instance
        """
        self.length = length
        self.diameter = diameter
        self.voidage = voidage
        self.bulk_density = bulk_density
        self.n_nodes = n_nodes

        self.isotherm = isotherm
        self.kinetics = kinetics
        self.hydrodynamics = hydrodynamics

        # Spatial step size [m]
        self.dz = length / n_nodes

    def _ode_system(
        self,
        t: float,
        y: np.ndarray,
        u_feed: float,
        c_inlet: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """
        System of Ordinary Differential Equations (ODEs) derived from spatial discretization (Method of Lines).

        Arrangement of the state variable vector y:
        - y[0 : N*n_comp] : Gas phase concentrations C [mol/m^3]
        - y[N*n_comp : 2*N*n_comp] : Solid phase loading q [mol/kg]
        """
        n_comp = len(self.isotherm.components)
        N = self.n_nodes

        # Reshape state vector into (N, n_comp) matrices
        C = y[: N * n_comp].reshape((N, n_comp))
        q = y[N * n_comp :].reshape((N, n_comp))

        dC_dt = np.zeros_like(C)
        dq_dt = np.zeros_like(q)

        # Ratio of solid bed volume to void gas volume inside pores
        solid_to_gas_ratio = (1.0 - self.voidage) * self.bulk_density / self.voidage

        # Calculate rates at each spatial node
        for i in range(N):
            c_node = C[i, :]
            q_node = q[i, :]

            # 1. Calculate partial pressures of components using Ideal Gas Law and convert Pa to bar
            # P_i = C_i * R * T [Pa] -> divided by 1e5 to convert to bar for Langmuir model
            p_partial_bar = (c_node * R_GAS * temperature) / 1.0e5

            # 2. Calculate equilibrium loading from Dual-Site Langmuir isotherm [mol/kg]
            q_star = self.isotherm.loading(partial_pressures=p_partial_bar, temperature=temperature)

            # 3. Calculate mass transfer kinetics rate (LDF) [mol/(kg * s)]
            rate_solid = self.kinetics.calculate_rate(q_equilibrium=q_star, q_actual=q_node)
            dq_dt[i, :] = rate_solid

            # 4. Calculate gas phase mass balance using first-order Upwind differencing
            # Inlet boundary condition (node zero) and interior nodes
            if i == 0:
                dC_dz = (c_node - c_inlet) / self.dz
            else:
                dC_dz = (c_node - C[i - 1, :]) / self.dz

            # dC/dt = - (u / eps) * (dC/dz) - ((1 - eps) * rho_s / eps) * (dq/dt)
            dC_dt[i, :] = - (u_feed / self.voidage) * dC_dz - solid_to_gas_ratio * rate_solid

        # Return the concatenated array of time derivatives
        return np.concatenate([dC_dt.flatten(), dq_dt.flatten()])

    def simulate_adsorption_step(
        self,
        duration: float,
        u_feed: float,
        c_inlet: np.ndarray,
        temperature: float,
        initial_C: Optional[np.ndarray] = None,
        initial_q: Optional[np.ndarray] = None,
        n_time_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulates the adsorption step (column feed and breakthrough curve generation).

        Parameters:
        -----------
        duration : float
            Duration of the adsorption simulation step [seconds]
        u_feed : float
            Superficial gas velocity [m/s]
        c_inlet : np.ndarray
            Inlet concentration of components to the column [mol/m^3]
        temperature : float
            Operating temperature of the column [K]
        initial_C : Optional[np.ndarray]
            Initial gas phase concentration (default: clean column, zero)
        initial_q : Optional[np.ndarray]
            Initial solid phase loading (default: fully regenerated adsorbent, zero)
        n_time_points : int
            Number of time points for output data storage

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            - t_eval : Time array [seconds]
            - C_history : Gas phase concentration history (shape: time, column length, components)
            - q_history : Solid phase loading history (shape: time, column length, components)
        """
        n_comp = len(self.isotherm.components)
        N = self.n_nodes

        # Set initial conditions if not provided
        if initial_C is None:
            initial_C = np.zeros((N, n_comp))
        if initial_q is None:
            initial_q = np.zeros((N, n_comp))

        # Construct the initial state vector y0
        y0 = np.concatenate([initial_C.flatten(), initial_q.flatten()])

        # Time points for output evaluation
        t_eval = np.linspace(0.0, duration, n_time_points)

        # Solve the stiff ODE system using the stable Radau method
        solution = solve_ivp(
            fun=self._ode_system,
            t_span=(0.0, duration),
            y0=y0,
            t_eval=t_eval,
            method="Radau",
            args=(u_feed, np.asarray(c_inlet), temperature),
            rtol=1e-5,
            atol=1e-7,
        )

        if not solution.success:
            raise RuntimeError(f"ODE solver convergence failed: {solution.message}")

        # Reshape output data
        n_steps = len(solution.t)
        C_history = np.zeros((n_steps, N, n_comp))
        q_history = np.zeros((n_steps, N, n_comp))

        for step in range(n_steps):
            y_step = solution.y[:, step]
            C_history[step] = y_step[: N * n_comp].reshape((N, n_comp))
            q_history[step] = y_step[N * n_comp :].reshape((N, n_comp))

        return solution.t, C_history, q_history
