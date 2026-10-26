# -*- coding: utf-8 -*-
"""
auxiliary_units_network.py
Production-Grade Auxiliary Equipment and Boundary Network for VPSA/PSA on Zeolite 13X (CO2/N2)

Integrates with:
- bed_unified_parameters.py & bed_parameter_registry.csv (p_high=1.2e6 Pa, p_low=3000 Pa)
- thermo_final_v2_hefti.py / thermo_final.py (bar-native property bridge)
- mass_final_fully_coupled_ldf_benchmark_v4_fixed_verified.py (Calibrated LDF: CO2=0.1690 s^-1, N2=0.2200 s^-1)
- fluid_transport_properties_vpsa_advanced_v8_cached_verified_ldf_pro_integrated_v9_unified_gas_props.py (gas_props facade)
- hydrodynamics_pressure_drop_master_integrated_v25.py / v26
- column (1)-(3).py (OperatingConditions, Danckwerts inlet, BedState, PSACycleManager)
- vpsa_integrated_system.py (AuxiliaryNetwork time_step, set_mode, receive_column_outlet, summary)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List, Callable
import numpy as np

# Physical constants & unit bridge
R_GAS = 8.314462618  # J/mol/K
PA_PER_BAR = 1e5

# =============================================================================
# 0) LDF and Registry Interface (Calibrated Constants + Bridge)
# =============================================================================
CALIBRATED_LDF = {
    "CO2": {"k": 0.1690, "k_analytical": 0.3000, "r2": 0.9619, "discrepancy_pct": 43.66},
    "N2":  {"k": 0.2200, "k_analytical": 0.3900, "r2": 0.9580, "discrepancy_pct": 43.59},
}

class LDFInterface:
    def __init__(self, mode: str = "calibrated", mass_module: Any = None):
        self.mode = mode
        self.mass = mass_module
        self.overall: Any = None
        if self.mode == "overall" and self.mass is not None:
            try:
                params = self.mass.ZeoliteThirteenXParameters()
                self.overall = self.mass.OverallLDFModel(params)
            except Exception:
                self.overall = None

    def k(self, species: str, T_K: float = 298.15, P_bar: float = 1.0, u: float = 0.05,
          q: float = 0.0, q_sat: float = 1.0) -> float:
        sp_up = species.upper()
        if self.mode == "overall" and self.overall is not None:
            try:
                if sp_up == "CO2":
                    return float(self.overall.k_overall_CO2(T_K, P_bar, u, q, q_sat))
                else:
                    return float(self.overall.k_overall_N2(T_K, P_bar, u, q, q_sat))
            except Exception:
                pass
        return float(CALIBRATED_LDF.get(sp_up, {}).get("k", 0.3000))


# =============================================================================
# 1) ISA-75 Compressible Control Valve with Non-Linear Equal-Percentage Cv
# =============================================================================
def default_valve_characteristic(opening: float) -> float:
    opening = float(np.clip(opening, 0.0, 1.0))
    if opening <= 1e-6:
        return 0.0
    k_range = 30.0  # Rangeability parameter
    return float((math.exp(math.log(k_range) * opening) - 1.0) / (k_range - 1.0))

class ControlValve:
    def __init__(self, cv_max: float, opening: float = 1.0, x_t: float = 0.72, gamma: float = 1.40,
                 tag: str = "CV", characteristic: Callable[[float], float] = default_valve_characteristic):
        self.cv_max = float(cv_max)
        self.opening = float(np.clip(opening, 0.0, 1.0))
        self.x_t = float(x_t)
        self.gamma = float(gamma)
        self.tag = tag
        self.characteristic = characteristic

    def set_opening(self, opening: float) -> None:
        self.opening = float(np.clip(opening, 0.0, 1.0))

    def effective_cv(self) -> float:
        return self.cv_max * self.characteristic(self.opening)

    def flow(self, P_up_Pa: float, P_down_Pa: float, T_up_K: float, mw_mix_kg_per_mol: float,
             Z: float = 1.0) -> Dict[str, Any]:
        if P_up_Pa <= P_down_Pa or self.opening <= 1e-6:
            return {"m_dot": 0.0, "n_dot": 0.0, "choked": False, "dp": max(0.0, P_up_Pa - P_down_Pa), "Cv_eff": 0.0}
        
        dp = P_up_Pa - P_down_Pa
        x = dp / P_up_Pa
        Fg = self.gamma / 1.40
        x_choked = Fg * self.x_t
        
        if x >= x_choked:
            choked = True
            Y = 1.0 - (x_choked / (3.0 * Fg * self.x_t))
            x_eff = x_choked
        else:
            choked = False
            Y = 1.0 - (x / (3.0 * Fg * self.x_t))
            x_eff = x
            
        P_up_bar = P_up_Pa * 1e-5
        M_gmol = mw_mix_kg_per_mol * 1000.0
        cv_eff = self.effective_cv()
        
        # ISA-75 Compressible Gas Sizing Equation
        m_dot_kg_h = 27.3 * cv_eff * P_up_bar * Y * math.sqrt((x_eff * M_gmol) / max(1e-6, T_up_K * Z))
        m_dot = max(0.0, m_dot_kg_h / 3600.0)
        n_dot = m_dot / max(1e-12, mw_mix_kg_per_mol)
        
        return {"m_dot": m_dot, "n_dot": n_dot, "choked": choked, "dp": dp, "Cv_eff": cv_eff}


# =============================================================================
# 2) Industrial Vacuum Pump with Volumetric Efficiency & Klinkenberg Slip
# =============================================================================
class VacuumPump:
    def __init__(self, displacement_m3_h: float = 150.0, P_ult_Pa: float = 3000.0, eta_comp: float = 0.75,
                 tag: str = "VP", max_dP_dt: Optional[float] = None):
        self.Vd = float(displacement_m3_h)
        self.P_ult = float(P_ult_Pa)
        self.eta_comp = float(eta_comp)
        self.tag = tag
        self.max_dP_dt = max_dP_dt

    def suction(self, P_suction_Pa: float, P_discharge_Pa: float, T_suction_K: float,
                y: np.ndarray, mw: np.ndarray, dt: float = 1.0) -> Dict[str, Any]:
        if self.max_dP_dt is not None and (P_discharge_Pa - P_suction_Pa) / max(dt, 1e-6) > self.max_dP_dt:
            P_discharge_Pa = P_suction_Pa + self.max_dP_dt * dt
            
        if P_suction_Pa <= self.P_ult:
            return {"m_dot": 0.0, "n_dot": 0.0, "Q": 0.0, "slip": 1.0, "P_shaft_kW": 0.0, "eta_v": 0.0}
            
        # Volumetric efficiency drop-off near ultimate pressure (benchmark P_low = 3000 Pa)
        eta_v = 1.0 - (self.P_ult / max(self.P_ult + 1.0, P_suction_Pa))
        eta_v = float(np.clip(eta_v, 0.05, 1.0))
        
        # Knudsen / Klinkenberg rarefaction slip correction for deep vacuum
        Kn = 1e-6 / max(P_suction_Pa, 1.0)
        slip = 1.0 + (4.0 * Kn) / (1.0 + Kn)
        
        Q = (self.Vd / 3600.0) * eta_v * slip
        mw_mix = float(np.dot(y, mw))
        rho = max(1e-6, (P_suction_Pa * mw_mix) / (R_GAS * max(100.0, T_suction_K)))
        m_dot = Q * rho
        n_dot = m_dot / max(1e-12, mw_mix)
        
        gamma = 1.35
        rc = max(1.0, P_discharge_Pa / max(1.0, P_suction_Pa))
        w = (gamma / (gamma - 1.0)) * (R_GAS / mw_mix) * T_suction_K * (rc ** ((gamma - 1.0) / gamma) - 1.0)
        P_shaft_kW = (m_dot * w) / max(1e-3, self.eta_comp) / 1000.0
        
        return {"m_dot": m_dot, "n_dot": n_dot, "Q": Q, "slip": slip, "P_shaft_kW": max(0.0, P_shaft_kW), "eta_v": eta_v}


# =============================================================================
# 3) Multi-Component Dynamic Buffer Tank (Mass, Species & Energy Balance)
# =============================================================================
class BufferTank:
    def __init__(self, V_m3: float = 0.8, P_init_Pa: float = 1.01325e5, T_init_K: float = 298.15,
                 y_init: Optional[np.ndarray] = None, mw: np.ndarray = np.array([0.04401, 0.028014]), tag: str = "BT"):
        self.V = float(V_m3)
        self.P = float(P_init_Pa)
        self.T = float(T_init_K)
        self.y = np.array([0.15, 0.85] if y_init is None else y_init, dtype=float)
        self.mw = mw
        self.tag = tag
        self.n_total = (self.P * self.V) / (R_GAS * max(100.0, self.T))
        self.n_spec = self.n_total * self.y
        self.Cp_molar = 32.0  # J/mol/K (approximate mixture heat capacity)

    def step(self, dt: float, n_in: float, y_in: np.ndarray, T_in: float, n_out: float) -> Dict[str, Any]:
        dt = max(1e-6, dt)
        dn_in = max(0.0, n_in) * dt
        dn_out = max(0.0, n_out) * dt
        
        self.n_spec = np.maximum(1e-14, self.n_spec + dn_in * y_in - dn_out * self.y)
        self.n_total = float(np.sum(self.n_spec))
        self.y = self.n_spec / max(1e-14, self.n_total)
        
        H = self.n_total * self.Cp_molar * self.T
        H += dn_in * self.Cp_molar * T_in - dn_out * self.Cp_molar * self.T
        self.T = float(np.clip(H / max(1e-12, self.n_total * self.Cp_molar), 200.0, 600.0))
        self.P = (self.n_total * R_GAS * self.T) / max(1e-6, self.V)
        
        return {"P": self.P, "T": self.T, "y": self.y.copy(), "n_total": self.n_total}


# =============================================================================
# 4) Regulators, Pressure Ramp Limiters & Equalization Manifold
# =============================================================================
class PressureRegulator:
    def __init__(self, setpoint_Pa: float, tag: str = "PR"):
        self.setpoint = float(setpoint_Pa)
        self.tag = tag

    def outlet_pressure(self, P_in_Pa: float) -> float:
        return float(min(P_in_Pa, self.setpoint))

class RampLimiter:
    def __init__(self, max_rate: float, tag: str = "RL"):
        self.max_rate = float(max_rate)
        self.tag = tag
        self._last_value: Optional[float] = None

    def apply(self, target: float, dt: float) -> float:
        if self._last_value is None:
            self._last_value = float(target)
            return float(target)
        delta = float(target) - self._last_value
        max_delta = self.max_rate * max(1e-6, dt)
        delta_limited = float(np.clip(delta, -max_delta, +max_delta))
        self._last_value += delta_limited
        return self._last_value

class EqualizationManifold:
    def __init__(self, valve_cv: float = 25.0, tag: str = "EQ"):
        self.valve = ControlValve(cv_max=valve_cv, tag=f"{tag}_VALVE")
        self.tag = tag

    def connect(self, P_A: float, P_B: float, T: float, mw_mix: float, dt: float) -> Dict[str, Any]:
        if abs(P_A - P_B) < 10.0:
            return {"direction": "none", "n_transfer": 0.0, "m_transfer": 0.0, "choked": False}
        if P_A > P_B:
            f = self.valve.flow(P_A, P_B, T, mw_mix)
            direction = "A_to_B"
        else:
            f = self.valve.flow(P_B, P_A, T, mw_mix)
            direction = "B_to_A"
        return {"direction": direction, "n_transfer": f["n_dot"] * dt, "m_transfer": f["m_dot"] * dt, "choked": f["choked"]}


# =============================================================================
# 5) Main Boundary Network Manager (API for vpsa_integrated_system & column)
# =============================================================================
@dataclass
class BoundaryConfig:
    p_feed_Pa: float = 1.2e6      # 1.2 MPa (Registry benchmark)
    t_feed_K: float = 298.15
    y_feed: Tuple[float, float] = (0.15, 0.85)  # CO2, N2
    p_ambient_Pa: float = 1.01325e5
    p_low_Pa: float = 3000.0      # 30 mbar deep vacuum (Registry benchmark)
    mw_species: np.ndarray = field(default_factory=lambda: np.array([0.04401, 0.028014]))
    cv_feed: float = 20.0
    cv_purge: float = 12.0
    cv_vent: float = 35.0
    pump_disp_m3_h: float = 150.0
    pump_pult_Pa: float = 1500.0
    pump_max_dP_dt: Optional[float] = None
    tank_V_m3: float = 0.8

class AuxiliaryNetwork:
    """
    Standard-compliant AuxiliaryNetwork matching the contract expected by
    vpsa_integrated_system.py and column (1)-(3).py.
    """
    def __init__(self, column_area: float = 3.14159 * (0.01)**2, column_voidage: float = 0.35,
                 y_feed: Tuple[float, float] = (0.15, 0.85), T_feed: float = 298.15,
                 config: Optional[BoundaryConfig] = None, ldf_mode: str = "calibrated"):
        self.area = float(column_area)
        self.eps_b = float(column_voidage)
        self.cfg = BoundaryConfig(y_feed=y_feed, t_feed_K=T_feed) if config is None else config
        self.ldf = LDFInterface(mode=ldf_mode)
        
        # Equipment instances
        self.cv_feed = ControlValve(cv_max=self.cfg.cv_feed, tag="CV_FEED")
        self.cv_purge = ControlValve(cv_max=self.cfg.cv_purge, tag="CV_PURGE")
        self.cv_vent = ControlValve(cv_max=self.cfg.cv_vent, tag="CV_VENT")
        self.vacuum = VacuumPump(displacement_m3_h=self.cfg.pump_disp_m3_h,
                                 P_ult_Pa=self.cfg.pump_pult_Pa,
                                 tag="VP_MAIN",
                                 max_dP_dt=self.cfg.pump_max_dP_dt)
        self.product_tank = BufferTank(V_m3=self.cfg.tank_V_m3, P_init_Pa=self.cfg.p_ambient_Pa,
                                       T_init_K=self.cfg.t_feed_K, y_init=np.array(self.cfg.y_feed),
                                       mw=self.cfg.mw_species, tag="BT_PRODUCT")
        self.reg_feed = PressureRegulator(setpoint_Pa=self.cfg.p_feed_Pa, tag="PR_FEED")
        self.manifold = EqualizationManifold(valve_cv=25.0, tag="EQ")
        self.feed_ramp = RampLimiter(max_rate=1.0e5, tag="RL_FEED") # Pa/s
        
        self.current_mode = "FEED"
        self.history: List[Dict[str, Any]] = []
        self._t_clock = 0.0

    def set_mode(self, mode: str) -> None:
        self.current_mode = mode.upper()

    def ldf_rates(self) -> Dict[str, float]:
        return {"CO2": self.ldf.k("CO2"), "N2": self.ldf.k("N2")}

    def time_step(self, dt: float) -> Dict[str, Any]:
        """
        Computes boundary conditions for vpsa_integrated_system.py:
        Returns: {P_in (bar), u_in (m/s), y_feed (tuple), T_feed (K), power_W (float)}
        """
        dt = max(1e-6, dt)
        self._t_clock += dt
        mw_mix = float(np.dot(np.array(self.cfg.y_feed), self.cfg.mw_species))
        
        power_W = 0.0
        P_in_bar = self.cfg.p_feed_Pa * 1e-5
        u_in = 0.05  m/s superficial baseline
        y_feed_arr = self.cfg.y_feed
        T_feed = self.cfg.t_feed_K

        mode = self.current_mode
        if mode in ("FEED", "ADSORPTION", "PRESSURIZATION"):
            P_target_Pa = self.reg_feed.outlet_pressure(self.cfg.p_feed_Pa)
            P_applied_Pa = self.feed_ramp.apply(P_target_Pa, dt)
            P_in_bar = P_applied_Pa * 1e-5
            self.cv_feed.set_opening(1.0)
            # Estimate superficial velocity from valve flow / area / density
            rho_in = (P_applied_Pa * mw_mix) / (R_GAS * T_feed)
            f = self.cv_feed.flow(P_applied_Pa, max(1e3, P_applied_Pa * 0.95), T_feed, mw_mix)
            m_dot = f["m_dot"]
            u_in = m_dot / max(1e-6, rho_in * self.area)
            u_in = float(np.clip(u_in, 0.005, 0.5))
            
        elif mode in ("EVACUATION", "BLOWDOWN", "VACUUM"):
            # Deep vacuum evacuation toward p_low_Pa (3000 Pa)
            P_in_bar = max(self.cfg.p_low_Pa * 1e-5, 0.03)
            u_in = 0.02  # negative/counter-current handled by column manager
            # Compute vacuum shaft power
            vp = self.vacuum.suction(P_suction_Pa=self.cfg.p_low_Pa, P_discharge_Pa=self.cfg.p_ambient_Pa,
                                     T_suction_K=T_feed, y=np.array(self.cfg.y_feed), mw=self.cfg.mw_species, dt=dt)
            power_W = float(vp["P_shaft_kW"] * 1000.0)
            
        elif mode in ("PURGE", "RINSE"):
            P_in_bar = self.cfg.p_ambient_Pa * 1e-5
            u_in = 0.01
            
        elif mode in ("EQUALIZATION", "EQ"):
            P_in_bar = 0.5 * (self.cfg.p_feed_Pa + self.cfg.p_low_Pa) * 1e-5
            u_in = 0.0
            
        elif mode in ("VENT", "DEPRESSURIZE"):
            P_in_bar = self.cfg.p_ambient_Pa * 1e-5
            u_in = 0.02

        res = {
            "P_in": float(P_in_bar),
            "u_in": float(u_in),
            "y_feed": y_feed_arr,
            "T_feed": float(T_feed),
            "power_W": float(power_W)
        }
        
        self.history.append({"t": self._t_clock, "mode": mode, **res})
        return res

    def receive_column_outlet(self, Q_out_mol_s: float, P_out_bar: float) -> None:
        """Receives feedback from the column outlet for buffer tank / flare accounting."""
        if Q_out_mol_s > 0:
            n_in = Q_out_mol_s
            self.product_tank.step(dt=1.0, n_in=n_in, y_in=np.array(self.cfg.y_feed),
                                   T_in=self.cfg.t_feed_K, n_out=0.0)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_time_s": self._t_clock,
            "steps_recorded": len(self.history),
            "product_tank_P_bar": self.product_tank.P * 1e-5,
            "product_tank_T_K": self.product_tank.T,
            "ldf_rates": self.ldf_rates()
        }


# =============================================================================
# Self-Test Execution
# =============================================================================
if __name__ == "__main__":
    net = AuxiliaryNetwork()
    print("[INIT] AuxiliaryNetwork successfully initialized with Benchmark profile.")
    print("[LDF Rates]", net.ldf_rates())
    net.set_mode("FEED")
    bc = net.time_step(dt=1.0)
    print("[FEED Step BC]", bc)
    net.set_mode("EVACUATION")
    bc_vac = net.time_step(dt=1.0)
    print("[EVACUATION Step BC]", bc_vac)
    print("[Summary]", net.summary())
