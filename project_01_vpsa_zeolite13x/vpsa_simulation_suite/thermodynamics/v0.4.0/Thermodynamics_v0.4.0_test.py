"""
Comprehensive Test & Validation Suite for Thermodynamics_v0.4.0.py
======================================================================
Full Phase 1 + Phase 2 coverage (57 physics checks + 14 figures).

Categories
----------
  Isothermal (6)   Analytic (6)   qst (5)   IAST (6)   ExtendedDSL (3)
  Jacobian (3)     Units (3)      TothClip (2)   Extreme (4)   Repeat (2)
  GibbsDuhem (2)   Legacy (4)     NegativeTests (2)   ModuleSelfTest (1)
  Numeric (3)      ErrorPaths (1) HenryAsym (2)   Coverage (2)

Outputs
-------
  Console  : pass/fail summary + benchmark table
  Files    : ./thermo_test_results/  (14 figures + report.html)

Usage
-----
  python test_thermodynamics_v4.py
"""

import os
import sys
import io
import time
import base64
import subprocess
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

import importlib.util as _ilu
import pathlib as _pl

try:
    _here = _pl.Path(__file__).resolve().parent
    _spec = _ilu.spec_from_file_location(
        'Thermodynamics_v0_4_0', _here / 'Thermodynamics_v0.4.0.py')
    TM = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(TM)
except FileNotFoundError as _e:
    sys.exit(f"ERROR: place this file next to Thermodynamics_v0.4.0.py ({_e})")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "legend.framealpha": 0.9,
})

OUT = "thermo_test_results"
os.makedirs(OUT, exist_ok=True)
R = TM.R_GAS
SAVED_FIGS: List[str] = []

def save_fig(name: str, dpi: int = 300) -> str:
    path = f"{OUT}/{name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    SAVED_FIGS.append(path)
    return path

def fig_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
dsl_sys  = TM.get_krishnamurthy_dsl_system()
toth_sys = TM.get_cavenati_toth_system()

co2_dsl  = dsl_sys["CO2"]["isotherm"];  n2_dsl  = dsl_sys["N2"]["isotherm"]
co2_toth = toth_sys["CO2"]["isotherm"]; n2_toth = toth_sys["N2"]["isotherm"]
ch4_toth = toth_sys["CH4"]["isotherm"]

co2_toth_a = TM.TemperatureDependentTothAnalytic(
    qm0=5.50, b0=2.85, t0=0.480, dH=-35.5e3, T0=298.15)
n2_toth_a  = TM.TemperatureDependentTothAnalytic(
    qm0=2.30, b0=0.12, t0=0.950, dH=-18.0e3, T0=298.15)
co2_dsl_a = TM.TemperatureDependentDSLAnalytic(
    qm1_0=3489.44, b1_0=co2_dsl.b1_0, dH1=-36.64e3,
    qm2_0=2872.35, b2_0=co2_dsl.b2_0, dH2=-35.70e3, T0=298.15)
n2_dsl_a  = TM.TemperatureDependentDSLAnalytic(
    qm1_0=6613.55, b1_0=n2_dsl.b1_0, dH1=-15.82e3,
    qm2_0=1e-9, b2_0=n2_dsl.b2_0, dH2=-15.82e3, T0=298.15)

ext  = TM.ExtendedDSL(co2_dsl_a, n2_dsl_a)
iast = TM.BinaryIAST(co2_toth_a, n2_toth_a)

P_grid = np.logspace(-4, 2, 60)
T_grid = np.array([288.15, 298.15, 313.15, 328.15, 348.15])

# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    fn: Callable
    category: str
    passed: bool = False
    error: str = ""

CHECKS: List[Check] = []

def check(category: str):
    def deco(fn):
        CHECKS.append(Check(fn.__name__, fn, category))
        return fn
    return deco

# ===========================================================================
# GROUP A: ISOTHERMAL CHECKS (42 base + 15 addendum = 57 total)
# ===========================================================================

# --- Isothermal ------------------------------------------------------------
@check("Isothermal")
def t01_boundary():
    for iso in (co2_dsl, n2_dsl, co2_toth, n2_toth):
        assert iso.loading(0.0, 298.15) == 0.0
        assert iso.spreading_pressure(0.0, 298.15) == 0.0

@check("Isothermal")
def t02_henry():
    for iso in (co2_dsl, n2_dsl, co2_toth, n2_toth):
        p = 1e-12
        kh = iso.at_temperature(298.15).henry_constant()
        rel = abs(iso.loading(p, 298.15)/p - kh) / kh
        assert rel < 1e-4, f"rel={rel:.2e}"

@check("Isothermal")
def t03_monotonic():
    for iso in (co2_dsl, n2_dsl, co2_toth, n2_toth):
        q = [iso.loading(p, 298.15) for p in P_grid]
        assert np.all(np.diff(q) > 0)

@check("Isothermal")
def t04_exothermic():
    for iso in (co2_dsl, n2_dsl, co2_toth, n2_toth):
        assert iso.loading(1.0, 288.15) > iso.loading(1.0, 348.15)

@check("Isothermal")
def t05_positive():
    for iso in (co2_dsl, n2_dsl, co2_toth, n2_toth):
        for p in P_grid:
            for T in T_grid:
                assert iso.loading(p, T) >= 0.0

@check("Isothermal")
def t06_saturate():
    for iso in (co2_toth, co2_dsl):
        q_high = iso.loading(1e6, 298.15)
        assert np.isfinite(q_high) and q_high > 0

# --- Analytic --------------------------------------------------------------
@check("Analytic")
def t07_dqdp_toth():
    for p in (0.01, 0.1, 1.0, 10.0):
        a = co2_toth_a.dq_dP(p, 298.15)
        h = p * 1e-5
        n = (co2_toth_a.loading(p+h, 298.15) - co2_toth_a.loading(p-h, 298.15)) / (2*h)
        assert abs(a-n)/abs(a) < 1e-5

@check("Analytic")
def t08_dqdp_dsl():
    for p in (0.01, 0.1, 1.0, 10.0):
        a = co2_dsl_a.dq_dP(p, 298.15)
        h = p * 1e-5
        n = (co2_dsl_a.loading(p+h, 298.15) - co2_dsl_a.loading(p-h, 298.15)) / (2*h)
        assert abs(a-n)/abs(a) < 1e-5

@check("Analytic")
def t09_dqdt_toth():
    for p in (0.01, 0.1, 1.0, 10.0):
        a = co2_toth_a.dq_dT(p, 298.15)
        h = 1e-3
        n = (co2_toth_a.loading(p, 298.15+h) - co2_toth_a.loading(p, 298.15-h)) / (2*h)
        assert abs(a-n)/abs(a) < 1e-4

@check("Analytic")
def t10_dqdt_dsl():
    for p in (0.01, 0.1, 1.0, 10.0):
        a = co2_dsl_a.dq_dT(p, 298.15)
        h = 1e-3
        n = (co2_dsl_a.loading(p, 298.15+h) - co2_dsl_a.loading(p, 298.15-h)) / (2*h)
        assert abs(a-n)/abs(a) < 1e-4

@check("Analytic")
def t11_dqdt_negative():
    for p in (0.05, 0.5, 5.0):
        assert co2_toth_a.dq_dT(p, 298.15) < 0
        assert co2_dsl_a.dq_dT(p, 298.15) < 0

@check("Analytic")
def t12_richardson4():
    for p in (0.1, 1.0, 5.0):
        a = co2_toth_a.dq_dP(p, 298.15)
        h = p * 1e-4
        d1 = (co2_toth_a.loading(p+h, 298.15) - co2_toth_a.loading(p-h, 298.15)) / (2*h)
        d2 = (co2_toth_a.loading(p+2*h, 298.15) - co2_toth_a.loading(p-2*h, 298.15)) / (4*h)
        n = (4*d1 - d2) / 3
        assert abs(a-n)/abs(a) < 1e-7

# --- qst -------------------------------------------------------------------
@check("qst")
def t13_qst_toth_band():
    for p in (0.1, 1.0, 5.0):
        q = co2_toth_a.qst(p, 298.15)
        assert 20e3 < q < 50e3, f"p={p}, qst={q:.0f}"

@check("qst")
def t14_qst_dsl_henry():
    d = co2_dsl_a.qst(1e-4, 298.15)
    w = co2_dsl_a.qst_site_weighted(1e-4, 298.15)
    assert abs(d-w)/w < 1e-3

@check("qst")
def t15_qst_cc_toth():
    for p in (0.1, 1.0, 5.0):
        T = 298.15
        dT = (co2_toth_a.loading(p, T+1e-3) - co2_toth_a.loading(p, T-1e-3)) / 2e-3
        dP = (co2_toth_a.loading(p*1.00001, T) - co2_toth_a.loading(p*0.99999, T)) / (2*p*1e-5)
        qst_num = -R * T**2 * dT / (p * dP)
        assert abs(co2_toth_a.qst(p, T) - qst_num) / qst_num < 1e-3

@check("qst")
def t16_qst_decreases():
    assert co2_toth_a.qst(0.01, 298.15) > co2_toth_a.qst(10.0, 298.15)

@check("qst")
def t17_qst_dsl_band():
    for p in (0.1, 1.0, 5.0):
        q = co2_dsl_a.qst(p, 298.15)
        assert 30e3 < q < 40e3

# --- IAST ------------------------------------------------------------------
@check("IAST")
def t18_iast_selective():
    for p in (0.5, 1.0, 5.0):
        for T in (298.15, 323.15):
            q1, q2, x1, _ = iast.solve(p, 0.15, T)
            assert q1 > 0 and q2 > 0 and x1 > 0.15

@check("IAST")
def t19_iast_mass():
    q1, q2, x1, _ = iast.solve(1.0, 0.15, 298.15)
    assert abs(x1 - q1/(q1+q2)) < 1e-6

@check("IAST")
def t20_iast_pure_limit():
    q1, q2, _, _ = iast.solve(1.0, 1.0-1e-10, 298.15)
    ref = co2_toth_a.loading(1.0, 298.15)
    assert abs(q1-ref)/ref < 1e-3

@check("IAST")
def t21_iast_le_pure():
    for p in (0.5, 1.0, 5.0):
        q1, _, _, _ = iast.solve(p, 0.15, 298.15)
        assert q1 <= co2_toth_a.loading(p*0.15, 298.15) + 1e-6

@check("IAST")
def t22_iast_S_positive():
    S = []
    for p in (0.5, 1.0, 5.0):
        _, _, x1, _ = iast.solve(p, 0.15, 298.15)
        S.append((x1/(1-x1))/(0.15/0.85))
    assert all(s > 0 for s in S)

@check("IAST")
def t23_iast_psi_monotonic():
    _, _, _, psi_lo = iast.solve(0.1, 0.15, 298.15)
    _, _, _, psi_hi = iast.solve(10.0, 0.15, 298.15)
    assert psi_hi > psi_lo

# --- ExtendedDSL -----------------------------------------------------------
@check("ExtendedDSL")
def t24_ext_y1_pure():
    qc, qn = ext.loadings(1.0, 1.0, 298.15)
    ref = co2_dsl_a.loading(1.0, 298.15)
    assert abs(qc-ref)/ref < 1e-9 and abs(qn) < 1e-9

@check("ExtendedDSL")
def t25_ext_y0_pure():
    qc, qn = ext.loadings(1.0, 0.0, 298.15)
    ref = n2_dsl_a.loading(1.0, 298.15)
    assert abs(qc) < 1e-9 and abs(qn-ref)/ref < 1e-9

@check("ExtendedDSL")
def t26_ext_dqdy_signs():
    j = ext.jacobian(1.0, 0.15, 298.15)
    assert j["dq_c_dy1"] > 0
    assert j["dq_n_dy1"] < 0

# --- Jacobian --------------------------------------------------------------
def _num_ext_jac_central(P, y1, T):
    hP, hy, hT = 1e-5, 1e-5, 1e-4
    qcP, qnP = ext.loadings(P+hP, y1, T); qcM, qnM = ext.loadings(P-hP, y1, T)
    qcy, qny = ext.loadings(P, y1+hy, T); qcym, qnym = ext.loadings(P, y1-hy, T)
    qcT, qnT = ext.loadings(P, y1, T+hT); qcTm, qnTm = ext.loadings(P, y1, T-hT)
    return {"dq_c_dP": (qcP-qcM)/(2*hP), "dq_n_dP": (qnP-qnM)/(2*hP),
            "dq_c_dy1": (qcy-qcym)/(2*hy), "dq_n_dy1": (qny-qnym)/(2*hy),
            "dq_c_dT": (qcT-qcTm)/(2*hT), "dq_n_dT": (qnT-qnTm)/(2*hT)}

@check("Jacobian")
def t27_jacobian_vs_central():
    for P, y1, T in ((1.0,.15,298.15),(5.0,.5,323.15),
                     (0.5,.85,313.15),(2.0,.35,343.15)):
        jac = ext.jacobian(P, y1, T); ref = _num_ext_jac_central(P, y1, T)
        for k in jac:
            assert abs(jac[k]-ref[k])/max(abs(ref[k]), 1e-12) < 1e-5

@check("Jacobian")
def t28_jacobian_dqdP_sign():
    for p in (0.1, 1.0, 5.0):
        j = ext.jacobian(p, 0.15, 298.15)
        assert j["dq_c_dP"] > 0 and j["dq_n_dP"] > 0

@check("Jacobian")
def t29_jacobian_dqdT_sign():
    for p in (0.1, 1.0, 5.0):
        j = ext.jacobian(p, 0.15, 298.15)
        assert j["dq_c_dT"] < 0 and j["dq_n_dT"] < 0

# --- Units -----------------------------------------------------------------
@check("Units")
def t30_b_conversion():
    for bC in (8.65e-7, 2.63e-8, 2.50e-6):
        for T in (298.15, 323.15, 373.15):
            assert abs(TM.b_concentration_to_pressure(bC, T) - bC*1e5/(R*T)) < 1e-30

@check("Units")
def t31_mass_scaling():
    assert abs(TM.loading_volumetric_to_mass(3489.44, 1120.0) - 3.1155714285714286) < 1e-12

@check("Units")
def t32_mpa_bar():
    for p in (0.1, 1.0, 10.0):
        assert abs(TM.bar_to_mpa(TM.mpa_to_bar(p)) - p) < 1e-12

# --- Toth clip -------------------------------------------------------------
@check("TothClip")
def t33_clipped():
    iso = TM.TemperatureDependentTothAnalytic(
        qm0=5.5, b0=2.85, t0=1.4, dH=-35.5e3, T0=298.15, alpha=0.0)
    assert iso.at_temperature(298.15).t == 1.0

@check("TothClip")
def t34_in_range():
    iso = TM.TemperatureDependentTothAnalytic(
        qm0=5.5, b0=2.85, t0=0.5, dH=-35.5e3, T0=298.15, alpha=0.0)
    assert abs(iso.at_temperature(298.15).t - 0.5) < 1e-12

# --- Extreme ---------------------------------------------------------------
@check("Extreme")
def t35_low_p():
    for iso in (co2_dsl, co2_toth):
        q = iso.loading(1e-6, 298.15)
        assert np.isfinite(q) and q >= 0

@check("Extreme")
def t36_high_p():
    for iso in (co2_dsl, co2_toth):
        q = iso.loading(1e3, 298.15)
        assert np.isfinite(q) and q > 0

@check("Extreme")
def t37_low_T():
    for iso in (co2_dsl, co2_toth):
        q = iso.loading(1.0, 250.0)
        assert np.isfinite(q) and q > 0

@check("Extreme")
def t38_high_T():
    for iso in (co2_dsl, co2_toth):
        q = iso.loading(1.0, 500.0)
        assert np.isfinite(q) and q > 0

# --- Repeat ----------------------------------------------------------------
@check("Repeat")
def t39_det_loading():
    assert co2_dsl_a.loading(1.0, 298.15) == co2_dsl_a.loading(1.0, 298.15)

@check("Repeat")
def t40_det_iast():
    a = iast.solve(1.0, 0.15, 298.15); b = iast.solve(1.0, 0.15, 298.15)
    for x, y in zip(a, b):
        assert abs(x-y) < 1e-15

# --- Gibbs-Duhem -----------------------------------------------------------
@check("GibbsDuhem")
def t41_psi_integral():
    for iso in (co2_toth, co2_dsl):
        p = 1.0
        pp = np.linspace(1e-8, p, 2000)
        integrand = np.array([iso.loading(x, 298.15)/x for x in pp])
        psi_num = np.trapezoid(integrand, pp)
        psi_ana = iso.spreading_pressure(p, 298.15)
        assert abs(psi_num - psi_ana) / psi_ana < 1e-3

@check("GibbsDuhem")
def t42_selectivity_positive():
    qc = co2_toth_a.loading(0.15, 298.15)
    qn = n2_toth_a.loading(0.85, 298.15)
    ratio_pure = (qc/0.15) / (qn/0.85)
    _, _, x1, _ = iast.solve(1.0, 0.15, 298.15)
    S = (x1/0.15) / ((1-x1)/0.85)
    assert S > 1.0 and S >= ratio_pure * 0.5

# ===========================================================================
# GROUP B: v3 ADDENDUM (15 additional checks) — Legacy & negative coverage
# ===========================================================================
@check("Legacy")
def g01_iast_analytic_jacobian():
    ia = TM.BinaryIASTAnalyticJacobian(co2_toth_a, n2_toth_a)
    j = ia.analytic_jacobian(1.0, 0.15, 298.15)
    assert set(j) == {"dq1_dP", "dq2_dP", "dq1_dT", "dq2_dT"}
    assert j["dq1_dP"] > 0 and j["dq2_dP"] > 0
    assert j["dq1_dT"] < 0 and j["dq2_dT"] < 0
    assert abs(j["dq1_dP"]) > abs(j["dq2_dP"])
    assert abs(j["dq1_dT"]) > abs(j["dq2_dT"])

@check("Legacy")
def g02_iast_num_vs_ana():
    jn = TM.BinaryIAST(co2_toth_a, n2_toth_a).thermodynamic_jacobian(1.0, 0.15, 298.15)
    ja = TM.BinaryIASTAnalyticJacobian(co2_toth_a, n2_toth_a).analytic_jacobian(1.0, 0.15, 298.15)
    for k in ("dq1_dP", "dq2_dP", "dq1_dT", "dq2_dT"):
        assert np.isfinite(jn[k]) and np.isfinite(ja[k])
        assert (jn[k] > 0) == (ja[k] > 0), f"{k} sign mismatch"

@check("Legacy")
def g03_toth_legacy_heat():
    iso_T = co2_toth.at_temperature(298.15)
    qs = np.linspace(0.01, iso_T.qm * 0.99, 30)
    dhs = np.array([iso_T.get_isosteric_heat(q) for q in qs])
    for dh in dhs:
        assert np.isfinite(dh) and 25e3 < dh < 45e3
    assert np.all(np.diff(dhs) < 0)

@check("Legacy")
def g04_thermal_legacy_heat():
    for p in (0.1, 1.0, 5.0):
        dh = co2_toth.isosteric_heat(P=p, T=298.15)
        assert 20e3 < dh < 50e3
        qst_ana = co2_toth_a.qst(p, 298.15)
        assert abs(dh - qst_ana) / qst_ana < 0.15

@check("Legacy")
def g05_richardson_numerical():
    iso_T = co2_toth.at_temperature(298.15)
    for p in (0.05, 0.5, 5.0):
        d_r = iso_T.dq_dP_numerical(p)
        d_a = co2_toth_a.dq_dP(p, 298.15)
        assert abs(d_r - d_a) / abs(d_a) < 1e-4

@check("Legacy")
def g06_invert_roundtrip():
    for iso in (co2_toth.at_temperature(298.15), co2_dsl.at_temperature(298.15)):
        for p_t in (0.01, 0.1, 1.0, 5.0):
            psi = iso.spreading_pressure(p_t)
            p_back = iso.invert_spreading_pressure(psi, P_guess=p_t)
            assert abs(p_back - p_t) / p_t < 1e-4
    assert co2_toth.at_temperature(298.15).invert_spreading_pressure(0.0) == 0.0

@check("ErrorPaths")
def g07_value_errors():
    cases = [
        (TM.concentration_to_pressure_bar, (-1.0, 300.0)),
        (TM.concentration_to_pressure_bar, (1.0, 0.0)),
        (TM.b_concentration_to_pressure, (-1e-7, 300.0)),
        (TM.b_concentration_to_pressure, (1e-7, -300.0)),
        (TM.loading_volumetric_to_mass, (-1.0, 1000.0)),
        (TM.loading_volumetric_to_mass, (1.0, 0.0)),
    ]
    for fn, args in cases:
        try:
            fn(*args); assert False, f"{fn.__name__}{args} did NOT raise"
        except ValueError:
            pass

@check("Legacy")
def g08_ch4_iast():
    b = TM.BinaryIAST(toth_sys["CO2"]["isotherm"], toth_sys["CH4"]["isotherm"])
    for p in (0.5, 1.0, 5.0):
        q1, q2, x1, psi = b.solve(p, 0.5, 298.15)
        assert q1 > 0 and q2 > 0 and psi > 0 and 0 < x1 < 1 and x1 > 0.5

@check("Legacy")
def g09_cp_ads():
    for sp in ("CO2", "N2"):
        cp = co2_dsl.get_adsorbed_heat_capacity(1.0, 298.15, species=sp)
        assert 20 < cp < 60
        cp_gas = co2_dsl.at_temperature(298.15).get_adsorbed_heat_capacity(298.15, 1.0, species=sp)
        assert abs(cp - cp_gas) < 1e-6

@check("Legacy")
def g10_toth_langmuir():
    iso = TM.TothIsotherm(qm=5.0, b=1.0, t=1.0)
    for p in (0.01, 0.1, 1.0, 10.0):
        q_toth = iso.loading(p)
        q_lang = 5.0 * 1.0 * p / (1.0 + 1.0 * p)
        assert abs(q_toth - q_lang) < 1e-10

@check("Legacy")
def g11_ext_rich_co2():
    for p in (0.5, 1.0, 5.0):
        qc, qn = ext.loadings(p, 0.85, 298.15)
        assert qc > 0 and qn > 0
        j = ext.jacobian(p, 0.85, 298.15)
        for v in j.values():
            assert np.isfinite(v)

@check("Legacy")
def g12_iast_vs_ext_low_p():
    for p_h in (1e-6, 1e-5, 1e-4):
        qi1, qi2, _, _ = iast.solve(p_h, 0.15, 298.15)
        qe1, qe2 = ext.loadings(p_h, 0.15, 298.15)
        for q in (qi1, qi2, qe1, qe2):
            assert np.isfinite(q) and q >= 0
        assert qi1 > 0 and qe1 > 0

@check("ModuleSelfTest")
def g13_subprocess():
    r = subprocess.run(
        [sys.executable, str(_here / "Thermodynamics_v0.4.0.py")],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, f"exit={r.returncode}\nstderr={r.stderr[-300:]}"
    assert "[LOCKED] Full engine" in r.stdout
    assert "Ran 31 tests" in r.stderr
    assert "OK" in r.stderr

@check("Legacy")
def g14_rel_diff():
    f = TM._rel_diff
    for x in (0.0, 1e-15, 0.5, 1.0, 100.0, 1e6):
        assert f(x, x) == 0.0
    for a, b in ((1, 2), (100, 99), (-1, 1), (0.1, 0.1000001)):
        assert f(a, b) == f(b, a)
        assert 0 <= f(a, b) <= 2.0 + 1e-12
    assert f(0.0, 0.0) == 0.0

@check("NegativeTests")
def g15_nnt_bug_persists():
    bug = TM.TemperatureDependentToth(
        qm0=5.5, b0=2.85, t0=1.4, dH=-35.5e3, T0=298.15, alpha=0.0)
    assert bug.at_temperature(298.15).t == 1.4
    fix = TM.TemperatureDependentTothAnalytic(
        qm0=5.5, b0=2.85, t0=1.4, dH=-35.5e3, T0=298.15, alpha=0.0)
    assert fix.at_temperature(298.15).t == 1.0
    q_bug = bug.at_temperature(298.15).loading(1.0)
    q_fix = fix.at_temperature(298.15).loading(1.0)
    assert abs(q_bug - q_fix) / max(q_bug, q_fix) > 1e-3

# ===========================================================================
# EXECUTE ALL CHECKS
# ===========================================================================
print("=" * 78)
print("  THERMODYNAMICS ENGINE v4 — COMPREHENSIVE VALIDATION SUITE")
print("=" * 78)

by_cat: Dict[str, List[Check]] = {}
for c in CHECKS:
    try:
        c.fn(); c.passed = True
    except AssertionError as e:
        c.error = str(e) or "assertion failed"
    except Exception as e:
        c.error = f"{type(e).__name__}: {e}"
    by_cat.setdefault(c.category, []).append(c)

passed = sum(1 for c in CHECKS if c.passed)
failed = len(CHECKS) - passed

for cat, lst in by_cat.items():
    print(f"\n  [{cat}]")
    for c in lst:
        mark = "PASS" if c.passed else "FAIL"
        print(f"    [{mark}] {c.name}")
        if not c.passed:
            print(f"           -> {c.error}")

print(f"\n  -> {passed}/{len(CHECKS)} physics checks passed" +
      ("  [ALL GREEN]" if failed == 0 else f"  ({failed} failed)"))

# ===========================================================================
# PERFORMANCE BENCHMARK
# ===========================================================================
def _bench(fn, n=1000):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6

print("\n" + "-" * 78)
print("  PERFORMANCE BENCHMARK")
print("-" * 78)

BENCH = {
    "Toth.loading":         _bench(lambda: co2_toth_a.loading(1.0, 298.15)),
    "DSL.loading":          _bench(lambda: co2_dsl_a.loading(1.0, 298.15)),
    "Toth.dq_dP":           _bench(lambda: co2_toth_a.dq_dP(1.0, 298.15)),
    "DSL.dq_dP":            _bench(lambda: co2_dsl_a.dq_dP(1.0, 298.15)),
    "Toth.dq_dT":           _bench(lambda: co2_toth_a.dq_dT(1.0, 298.15)),
    "DSL.dq_dT":            _bench(lambda: co2_dsl_a.dq_dT(1.0, 298.15)),
    "Toth.qst":             _bench(lambda: co2_toth_a.qst(1.0, 298.15)),
    "DSL.qst":              _bench(lambda: co2_dsl_a.qst(1.0, 298.15), n=500),
    "ExtendedDSL.jacobian": _bench(lambda: ext.jacobian(1.0, 0.15, 298.15), n=500),
    "BinaryIAST.solve":     _bench(lambda: iast.solve(1.0, 0.15, 298.15), n=100),
}
for k, v in BENCH.items():
    print(f"    {k:24s}  {v:8.3f} µs/call")

# ===========================================================================
# FIGURE 1 — Pure isotherms
# ===========================================================================
print("\n" + "-" * 78)
print("  GENERATING FIGURES")
print("-" * 78)

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
fig.suptitle("Fig. 1 — Pure-Component Isotherms on Zeolite 13X",
             fontsize=13, fontweight="bold")
for ax, iso, unit, title in [
    (axes[0, 0], co2_toth, "mol/kg", "Toth — CO2"),
    (axes[0, 1], n2_toth,  "mol/kg", "Toth — N2"),
    (axes[1, 0], co2_dsl,  "mol/m³", "DSL — CO2"),
    (axes[1, 1], n2_dsl,   "mol/m³", "DSL — N2"),
]:
    for T, col in zip(T_grid, plt.cm.viridis(np.linspace(0.05, 0.9, len(T_grid)))):
        q = [iso.loading(p, T) for p in P_grid]
        ax.loglog(P_grid, q, color=col, lw=1.8, label=f"T = {T:.0f} K")
    ax.set_xlabel("P [bar]"); ax.set_ylabel(f"q [{unit}]")
    ax.set_title(title); ax.legend(loc="upper left")
save_fig("fig01_pure_isotherms")

# ===========================================================================
# FIGURE 2 — Derivative convergence
# ===========================================================================
h_values = np.logspace(-6, 0, 25)
def fd_conv(f, df, P, h):
    ex = df(P)
    return {
        "forward": [abs((f(P+h)-f(P))/h - ex)/abs(ex) for h in h_values],
        "central": [abs((f(P+h)-f(P-h))/(2*h) - ex)/abs(ex) for h in h_values],
        "richardson4": [
            abs((4*(f(P+h)-f(P-h))/(2*h) - (f(P+2*h)-f(P-2*h))/(4*h))/3 - ex)/abs(ex)
            for h in h_values],
    }

conv = fd_conv(lambda p: co2_toth_a.loading(p, 298.15),
               lambda p: co2_toth_a.dq_dP(p, 298.15), 1.0, h_values)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
fig.suptitle("Fig. 2 — FD Convergence (Toth dq/dP @ P=1 bar)", fontsize=13, fontweight="bold")
ax = axes[0]
P_rel = np.logspace(-3, 2, 50)
ax.loglog(P_rel, [co2_toth_a.dq_dP(p, 298.15) for p in P_rel], "b-", lw=2, label="analytic")
ax.loglog(P_rel, [(co2_toth_a.loading(p*1.0001, 298.15) - co2_toth_a.loading(p*0.9999, 298.15))/
                  (2*p*1e-4) for p in P_rel], "r--", lw=1.5, label="central FD")
ax.set_xlabel("P [bar]"); ax.set_ylabel("dq/dP")
ax.set_title("Analytic vs Central FD"); ax.legend()

ax = axes[1]
for method, style in [("forward", "o-"), ("central", "^-"), ("richardson4", "d-")]:
    ax.loglog(h_values, conv[method], style, lw=1.5, ms=4, label=method)
ax.set_xlabel("Step size h"); ax.set_ylabel("Relative error")
ax.set_title("Convergence vs h"); ax.legend()
save_fig("fig02_derivatives_convergence")

# ===========================================================================
# ===========================================================================
# ===========================================================================
# ===========================================================================
# FIGURE 3 - q(P, T) heatmaps (2D contourf - replaces unreadable 3D)
# ===========================================================================
# Rationale: matplotlib 3D axes with log-scaled X cannot keep X and Z
# tick labels from overlapping. A 2D contourf in log10(P) coordinates is
# the publication-standard way to visualize q(P, T) for adsorption.
# ===========================================================================
Pg3 = np.logspace(-3, 1, 200)          # dense for smooth contours
Tg3 = np.linspace(285, 350, 150)
PP, TT = np.meshgrid(Pg3, Tg3)
QQ_toth = np.array([[co2_toth_a.loading(p, t) for p in Pg3] for t in Tg3])
QQ_dsl  = np.array([[co2_dsl_a.loading(p, t) for p in Pg3] for t in Tg3])

logP = np.log10(Pg3)                    # actual x-coordinate

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
fig.suptitle("Fig. 3 - CO2 loading q(P, T) on Zeolite 13X",
             fontsize=14, fontweight="bold")

xtick_vals   = [-3, -2, -1, 0, 1]
xtick_labels = ["0.001", "0.01", "0.1", "1", "10"]

for ax, Q, title, unit in [
        (axes[0], QQ_toth, "Cavenati Toth", "mol/kg"),
        (axes[1], QQ_dsl,  "Krishnamurthy DSL", "mol/m3")]:

    im = ax.contourf(logP, Tg3, Q, levels=25, cmap="viridis")
    cs = ax.contour(logP, Tg3, Q, levels=8, colors="white",
                    linewidths=0.6, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(xtick_labels, fontsize=11)
    ax.set_xlabel("P [bar]   (log scale)", fontsize=12)
    ax.set_ylabel("T [K]", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.tick_params(labelsize=10)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, aspect=25, pad=0.02)
    cbar.set_label(f"q [{unit}]", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

save_fig("fig03_3d_surfaces", dpi=400)

# ===========================================================================
# FIGURE 4 — qst
# ===========================================================================
P_rel = np.logspace(-3, 2, 50)
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
fig.suptitle("Fig. 4 — Isosteric Heat of Adsorption (CO2)", fontsize=13, fontweight="bold")
ax = axes[0]
for T, col in zip(T_grid, plt.cm.plasma(np.linspace(0.1, 0.85, len(T_grid)))):
    ax.semilogx(P_rel, [co2_toth_a.qst(p, T)/1e3 for p in P_rel],
                color=col, lw=1.8, label=f"T = {T:.0f} K")
ax.set_xlabel("P [bar]"); ax.set_ylabel("q_st [kJ/mol]")
ax.set_title("Toth"); ax.legend()

ax = axes[1]
ax.semilogx(P_rel, [co2_dsl_a.qst(p, 298.15)/1e3 for p in P_rel],
            "b-", lw=2, label="CC derivative")
ax.semilogx(P_rel, [co2_dsl_a.qst_site_weighted(p, 298.15)/1e3 for p in P_rel],
            "r--", lw=1.5, label="site-weighted")
ax.set_xlabel("P [bar]"); ax.set_ylabel("q_st [kJ/mol]")
ax.set_title("DSL — two forms"); ax.legend()
save_fig("fig04_qst")

# ===========================================================================
# FIGURE 5 — IAST
# ===========================================================================
P_iast = np.logspace(-2, 1.2, 40); y1_fix = 0.15
q1_arr, q2_arr, x1_arr, S_arr = [], [], [], []
for p in P_iast:
    q1, q2, x1, _ = iast.solve(p, y1_fix, 298.15)
    q1_arr.append(q1); q2_arr.append(q2); x1_arr.append(x1)
    S_arr.append((x1/(1-x1)) / (y1_fix/(1-y1_fix)))

fig = plt.figure(figsize=(13, 9), constrained_layout=True)
gs = GridSpec(2, 2, figure=fig)
fig.suptitle("Fig. 5 — Binary IAST: CO2/N2", fontsize=13, fontweight="bold")
ax = fig.add_subplot(gs[0, 0])
ax.loglog(P_iast, q1_arr, "b-", lw=2, label="q_CO2")
ax.loglog(P_iast, q2_arr, "r-", lw=2, label="q_N2")
ax.set_xlabel("P_total [bar]"); ax.set_ylabel("Loading [mol/kg]")
ax.set_title(f"IAST loadings @ y_CO2={y1_fix}"); ax.legend()

ax = fig.add_subplot(gs[0, 1])
ax.semilogx(P_iast, x1_arr, "k-", lw=2)
ax.axhline(y1_fix, color="gray", ls=":", label=f"y = {y1_fix}")
ax.set_xlabel("P_total [bar]"); ax.set_ylabel("x_CO2")
ax.set_title("Adsorbed fraction"); ax.legend()

ax = fig.add_subplot(gs[1, 0])
ax.loglog(P_iast, S_arr, "g-", lw=2)
ax.set_xlabel("P_total [bar]"); ax.set_ylabel("S_CO2/N2")
ax.set_title("Selectivity")

ax = fig.add_subplot(gs[1, 1])
q_co2_p = [co2_toth_a.loading(p*y1_fix, 298.15) for p in P_iast]
q_n2_p  = [n2_toth_a.loading(p*(1-y1_fix), 298.15) for p in P_iast]
ax.loglog(P_iast, q1_arr, "b-", lw=2, label="IAST CO2")
ax.loglog(P_iast, q_co2_p, "b--", lw=1.5, label="pure CO2")
ax.loglog(P_iast, q2_arr, "r-", lw=2, label="IAST N2")
ax.loglog(P_iast, q_n2_p, "r--", lw=1.5, label="pure N2")
ax.set_xlabel("P_total [bar]"); ax.set_ylabel("Loading [mol/kg]")
ax.set_title("IAST vs pure"); ax.legend(fontsize=7)
save_fig("fig05_iast")

# ===========================================================================
# FIGURE 6 — ExtendedDSL Jacobian
# ===========================================================================
P_e = np.logspace(-2, 1.5, 40); y_fix = 0.15
qc_list, qn_list = zip(*[ext.loadings(p, y_fix, 298.15) for p in P_e])
jac_arr = [ext.jacobian(p, y_fix, 298.15) for p in P_e]

fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig.suptitle("Fig. 6 — ExtendedDSL Jacobian", fontsize=13, fontweight="bold")
ax = axes[0, 0]
ax.loglog(P_e, qc_list, "b-", lw=2, label="q_CO2")
ax.loglog(P_e, qn_list, "r-", lw=2, label="q_N2")
ax.set_xlabel("P [bar]"); ax.set_ylabel("q [mol/m³]")
ax.set_title(f"Loadings (y1={y_fix})"); ax.legend()

ax = axes[0, 1]
ax.loglog(P_e, [j["dq_c_dP"] for j in jac_arr], "b-", lw=2, label="dq_CO2/dP")
ax.loglog(P_e, [j["dq_n_dP"] for j in jac_arr], "r-", lw=2, label="dq_N2/dP")
ax.set_xlabel("P [bar]"); ax.set_title("dq/dP"); ax.legend()

ax = axes[1, 0]
ax.semilogx(P_e, [j["dq_c_dy1"] for j in jac_arr], "b-", lw=2, label="dq_CO2/dy1")
ax.semilogx(P_e, [j["dq_n_dy1"] for j in jac_arr], "r-", lw=2, label="dq_N2/dy1")
ax.set_xlabel("P [bar]"); ax.set_title("dq/dy1"); ax.legend()

ax = axes[1, 1]
ax.semilogx(P_e, [j["dq_c_dT"] for j in jac_arr], "b-", lw=2, label="dq_CO2/dT")
ax.semilogx(P_e, [j["dq_n_dT"] for j in jac_arr], "r-", lw=2, label="dq_N2/dT")
ax.set_xlabel("P [bar]"); ax.set_title("dq/dT"); ax.legend()
save_fig("fig06_extended_dsl")

# ===========================================================================
# FIGURE 7 — Error distributions
# ===========================================================================
def rel_errs(iso, fn_a, fn_n, P_set, T_set):
    errs = []
    for p in P_set:
        for t in T_set:
            a, n = fn_a(iso, p, t), fn_n(iso, p, t)
            if abs(a) > 1e-12:
                errs.append(abs(a-n)/abs(a))
    return np.array(errs)

def num_dp(iso, p, t):
    h = max(p*1e-5, 1e-12)
    return (iso.loading(p+h, t) - iso.loading(p-h, t)) / (2*h)
def num_dt(iso, p, t):
    return (iso.loading(p, t+1e-3) - iso.loading(p, t-1e-3)) / 2e-3

P_set = np.logspace(-2, 1, 20); T_set = np.linspace(288, 348, 15)
E_dp_t = rel_errs(co2_toth_a, lambda i,p,t: i.dq_dP(p,t), num_dp, P_set, T_set)
E_dp_d = rel_errs(co2_dsl_a,  lambda i,p,t: i.dq_dP(p,t), num_dp, P_set, T_set)
E_dt_t = rel_errs(co2_toth_a, lambda i,p,t: i.dq_dT(p,t), num_dt, P_set, T_set)
E_dt_d = rel_errs(co2_dsl_a,  lambda i,p,t: i.dq_dT(p,t), num_dt, P_set, T_set)

fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig.suptitle("Fig. 7 — Analytic vs Numerical Error Distributions",
             fontsize=13, fontweight="bold")
for ax, E, title in zip(axes.flat,
    (E_dp_t, E_dp_d, E_dt_t, E_dt_d),
    ("Toth dq/dP", "DSL dq/dP", "Toth dq/dT", "DSL dq/dT")):
    ax.hist(np.log10(E + 1e-20), bins=25, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("log10(rel. error)"); ax.set_ylabel("count")
    ax.set_title(f"{title}  (median={np.median(E):.1e})")
    ax.axvline(np.log10(np.median(E)), color="red", ls="--")
save_fig("fig07_error_distributions")

# ===========================================================================
# FIGURE 8 — VPSA working capacity
# ===========================================================================
def vpsa_wc(P_hi, P_lo, y_feed=0.15, T=298.15):
    qc_a, qn_a = ext.loadings(P_hi, y_feed, T)
    qc_d, qn_d = ext.loadings(P_lo, 0.001, T)
    return qc_a - qc_d, qc_a/(qc_a+qn_a)

P_hi_grid = np.logspace(-1, 1.5, 30)
wc_arr, pur_arr = zip(*[vpsa_wc(p, 0.1) for p in P_hi_grid])

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
fig.suptitle("Fig. 8 — VPSA Working Capacity (P_low=0.1 bar)",
             fontsize=13, fontweight="bold")
ax = axes[0]
ax.loglog(P_hi_grid, wc_arr, "b-", lw=2)
ax.set_xlabel("P_high [bar]"); ax.set_ylabel("WC_CO2 [mol/m³]")
ax.set_title("CO2 working capacity")

ax = axes[1]
ax.semilogx(P_hi_grid, pur_arr, "g-", lw=2)
ax.set_xlabel("P_high [bar]"); ax.set_ylabel("CO2 purity in adsorbed phase")
ax.set_title("Purity")
save_fig("fig08_vpsa_working_capacity")

# ===========================================================================
# FIGURE 9 — Sensitivity tornado
# ===========================================================================
def sens_tornado(delta=0.10, P=1.0, T=298.15):
    base = co2_dsl_a.loading(P, T)
    params = [("qm1_0", co2_dsl_a.qm1_0), ("qm2_0", co2_dsl_a.qm2_0),
              ("b1_0", co2_dsl_a.b1_0),   ("b2_0", co2_dsl_a.b2_0),
              ("dH1", co2_dsl_a.dH1),     ("dH2", co2_dsl_a.dH2)]
    out = []
    for label, val in params:
        plus_kw = dict(qm1_0=co2_dsl_a.qm1_0, b1_0=co2_dsl_a.b1_0,
                       dH1=co2_dsl_a.dH1, qm2_0=co2_dsl_a.qm2_0,
                       b2_0=co2_dsl_a.b2_0, dH2=co2_dsl_a.dH2, T0=co2_dsl_a.T0)
        minus_kw = dict(plus_kw)
        plus_kw[label] = val * (1 + delta)
        minus_kw[label] = val * (1 - delta)
        q_plus = TM.TemperatureDependentDSLAnalytic(**plus_kw).loading(P, T)
        q_minus = TM.TemperatureDependentDSLAnalytic(**minus_kw).loading(P, T)
        out.append((label, (q_plus-base)/base*100, (q_minus-base)/base*100))
    return out

sens = sens_tornado()
labels = [s[0] for s in sens]
plus_vals = [s[1] for s in sens]
minus_vals = [s[2] for s in sens]
order = np.argsort([-max(abs(p), abs(m)) for p, m in zip(plus_vals, minus_vals)])

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
fig.suptitle("Fig. 9 — Parameter Sensitivity (Tornado, ±10%)",
             fontsize=13, fontweight="bold")
y = np.arange(len(labels))
ax.barh(y, [plus_vals[i] for i in order], color="tab:red", label="+10%")
ax.barh(y, [minus_vals[i] for i in order], color="tab:blue", label="−10%")
ax.set_yticks(y); ax.set_yticklabels([labels[i] for i in order])
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Δq_CO2 [%]")
ax.set_title("q_CO2 @ P=1 bar, T=298 K")
ax.legend()
save_fig("fig09_sensitivity_tornado")

# ===========================================================================
# FIGURE 10 — Benchmark
# ===========================================================================
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
fig.suptitle("Fig. 10 — Performance Benchmark (lower is better)",
             fontsize=13, fontweight="bold")
keys = list(BENCH.keys()); vals = [BENCH[k] for k in keys]
colors = ["tab:blue" if "loading" in k else "tab:orange" if "dq_dP" in k
          else "tab:green" if "dq_dT" in k else "tab:red" if "qst" in k
          else "tab:purple" for k in keys]
bars = ax.barh(keys, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(v*1.02, b.get_y()+b.get_height()/2, f"{v:.2f} µs",
            va="center", fontsize=8)
ax.set_xlabel("Time per call [µs]")
save_fig("fig10_benchmark")

# ===========================================================================
# FIGURE 11 — Dashboard
# ===========================================================================
fig = plt.figure(figsize=(15, 10), constrained_layout=True)
gs = GridSpec(2, 3, figure=fig)
fig.suptitle("Fig. 11 — Engine Dashboard", fontsize=14, fontweight="bold")

ax = fig.add_subplot(gs[0, 0])
for name, iso, col in (("CO2", co2_toth, "tab:blue"), ("N2", n2_toth, "tab:red")):
    ax.loglog(P_grid, [iso.loading(p, 298.15) for p in P_grid], color=col, lw=2, label=name)
ax.set_title("Toth @ 298 K"); ax.set_xlabel("P [bar]"); ax.set_ylabel("q [mol/kg]"); ax.legend()

ax = fig.add_subplot(gs[0, 1])
for name, iso, col in (("CO2", co2_dsl, "tab:blue"), ("N2", n2_dsl, "tab:red")):
    ax.loglog(P_grid, [iso.loading(p, 298.15) for p in P_grid], color=col, lw=2, label=name)
ax.set_title("DSL @ 298 K"); ax.set_xlabel("P [bar]"); ax.set_ylabel("q [mol/m³]"); ax.legend()

ax = fig.add_subplot(gs[0, 2])
for T, col in zip(T_grid, plt.cm.coolwarm(np.linspace(0, 1, len(T_grid)))):
    ax.loglog(P_grid, [co2_toth.loading(p, T) for p in P_grid], color=col, lw=1.5, label=f"{T:.0f}K")
ax.set_title("CO2 vs T"); ax.set_xlabel("P [bar]"); ax.set_ylabel("q [mol/kg]"); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[1, 0])
species = ["CO2", "N2", "CH4"]
kh = [toth_sys[s]["isotherm"].at_temperature(298.15).henry_constant() for s in species]
bars = ax.bar(species, kh, color=["tab:blue", "tab:red", "tab:green"])
for b, v in zip(bars, kh):
    ax.text(b.get_x()+b.get_width()/2, v*1.02, f"{v:.3g}", ha="center", fontsize=9)
ax.set_ylabel("K_H [mol/(kg·bar)]"); ax.set_title("Henry constants")

ax = fig.add_subplot(gs[1, 1])
ax.semilogx(P_iast, S_arr, "g-", lw=2)
ax.set_title("IAST S_CO2/N2"); ax.set_xlabel("P [bar]"); ax.set_ylabel("S")

ax = fig.add_subplot(gs[1, 2])
ax.semilogx(P_rel, [co2_toth_a.qst(p, 298.15)/1e3 for p in P_rel], "b-", lw=2, label="Toth")
ax.semilogx(P_rel, [co2_dsl_a.qst(p, 298.15)/1e3 for p in P_rel], "r-", lw=2, label="DSL")
ax.set_title("q_st @ 298K"); ax.set_xlabel("P [bar]"); ax.set_ylabel("kJ/mol"); ax.legend()
save_fig("fig11_dashboard")

# ===========================================================================
# FIGURE 12 — Legacy vs analytic qst
# ===========================================================================
q_grid = np.linspace(0.01, co2_toth.at_temperature(298.15).qm * 0.99, 80)
dh_leg = np.array([co2_toth.at_temperature(298.15).get_isosteric_heat(q) for q in q_grid]) / 1e3

def q_to_p(iso, q_target, T=298.15):
    try:
        return brentq(lambda p: iso.loading(p, T) - q_target, 1e-8, 1e6)
    except Exception:
        return np.nan

dh_ana = np.array([
    co2_toth_a.qst(q_to_p(co2_toth, q), 298.15) / 1e3
    if np.isfinite(q_to_p(co2_toth, q)) else np.nan
    for q in q_grid
])

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
fig.suptitle("Fig. 12 — Legacy vs Analytic Isosteric Heat", fontsize=13, fontweight="bold")
ax = axes[0]
ax.plot(q_grid, dh_leg, "b-", lw=2, label="get_isosteric_heat(q)")
ax.plot(q_grid, dh_ana, "r--", lw=2, label="qst(P(q),T)")
ax.set_xlabel("q [mol/kg]"); ax.set_ylabel("q_st [kJ/mol]")
ax.set_title("Toth CO2 — legacy vs analytic"); ax.legend()

ax = axes[1]
jn = TM.BinaryIAST(co2_toth_a, n2_toth_a).thermodynamic_jacobian(1.0, 0.15, 298.15)
ja = TM.BinaryIASTAnalyticJacobian(co2_toth_a, n2_toth_a).analytic_jacobian(1.0, 0.15, 298.15)
keys = ["dq1_dP", "dq2_dP", "dq1_dT", "dq2_dT"]
x = np.arange(len(keys)); w = 0.35
ax.bar(x - w/2, [jn[k] for k in keys], w, label="numerical IAST", color="steelblue")
ax.bar(x + w/2, [ja[k] for k in keys], w, label="analytic IAST", color="coral")
ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20)
ax.set_ylabel("Value")
ax.set_title("IAST Jacobian @ P=1, y1=0.15, T=298K")
ax.legend(); ax.axhline(0, color="k", lw=0.8)
save_fig("fig12_legacy_vs_analytic")

# ===========================================================================
# FIGURE 13 — nnt bug visible impact
# ===========================================================================
P_bug = np.logspace(-3, 1.5, 80)
bug = TM.TemperatureDependentToth(
    qm0=5.5, b0=2.85, t0=1.4, dH=-35.5e3, T0=298.15, alpha=0.0)
fix = TM.TemperatureDependentTothAnalytic(
    qm0=5.5, b0=2.85, t0=1.4, dH=-35.5e3, T0=298.15, alpha=0.0)
q_bug = np.array([bug.at_temperature(298.15).loading(p) for p in P_bug])
q_fix = np.array([fix.at_temperature(298.15).loading(p) for p in P_bug])

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
fig.suptitle("Fig. 13 — Toth nnt clipping bug (t0=1.4)", fontsize=13, fontweight="bold")
ax = axes[0]
ax.loglog(P_bug, q_bug, "b-", lw=2, label=f"original (t={bug.at_temperature(298.15).t})")
ax.loglog(P_bug, q_fix, "r--", lw=2, label=f"analytic (t={fix.at_temperature(298.15).t})")
ax.set_xlabel("P [bar]"); ax.set_ylabel("q [mol/kg]")
ax.set_title("Loading curves"); ax.legend()

ax = axes[1]
rel = np.abs(q_bug - q_fix) / np.maximum(q_bug, q_fix)
ax.loglog(P_bug, rel * 100, "k-", lw=2)
ax.axhline(0.1, color="red", ls=":", lw=1, label="0.1% threshold")
ax.set_xlabel("P [bar]"); ax.set_ylabel("Deviation [%]")
ax.set_title("|q_bug − q_fix| / max"); ax.legend()
save_fig("fig13_nnt_bug_impact")

# ===========================================================================
# FIGURE 14 — VPSA cycle animation (GIF)
# ===========================================================================
print("\n  Generating VPSA cycle animation...")

def build_cycle_points(n_ads=15, n_hold=10, n_blow=10, n_purge=10, cycles=3):
    pts = []
    y_f, T = 0.15, 298.15
    P_hi, P_lo = 2.0, 0.1
    for _ in range(cycles):
        for i in range(n_ads):
            t = i / n_ads
            pts.append((P_lo + (P_hi-P_lo)*t, 0.001 + (y_f-0.001)*t))
        for _ in range(n_hold):
            pts.append((P_hi, y_f))
        for i in range(n_blow):
            t = i / n_blow
            pts.append((P_hi - (P_hi-P_lo)*t, y_f*(1-t) + 0.001*t))
        for _ in range(n_purge):
            pts.append((P_lo, 0.001))
    return pts

points = build_cycle_points()
q_cycle = [ext.loadings(P, y, 298.15) for P, y in points]
qc_cycle = [q[0] for q in q_cycle]
qn_cycle = [q[1] for q in q_cycle]

fig_anim, ax_a = plt.subplots(figsize=(8, 6))
P_ref = np.logspace(-2, 1, 80)
def update(frame):
    ax_a.clear()
    ax_a.loglog(P_ref, [co2_dsl_a.loading(p, 298.15) for p in P_ref],
                "b--", lw=1, alpha=0.4, label="pure CO2")
    ax_a.loglog(P_ref, [n2_dsl_a.loading(p, 298.15) for p in P_ref],
                "r--", lw=1, alpha=0.4, label="pure N2")
    ax_a.loglog(qc_cycle[:frame+1], qn_cycle[:frame+1],
                "k-", lw=1.2, alpha=0.5, label="trajectory")
    ax_a.loglog([qc_cycle[frame]], [qn_cycle[frame]],
                "ko", ms=10, label="current")
    ax_a.set_xlabel("q_CO2 [mol/m³]"); ax_a.set_ylabel("q_N2 [mol/m³]")
    ax_a.set_title(f"VPSA Cycle — step {frame+1}/{len(points)}")
    ax_a.legend(loc="lower right", fontsize=8)

anim = FuncAnimation(fig_anim, update, frames=len(points), interval=80, blit=False)
gif_path = f"{OUT}/fig14_vpsa_animation.gif"
anim.save(gif_path, writer=PillowWriter(fps=15))
SAVED_FIGS.append(gif_path)
plt.close(fig_anim)
print(f"  Saved: {gif_path}")

# ===========================================================================
# HTML REPORT
# ===========================================================================
print("\n" + "-" * 78)
print("  GENERATING HTML REPORT")
print("-" * 78)

HTML_HEAD = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Thermodynamics v4 - Report</title>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;background:#f4f6f8;margin:0;padding:20px;color:#222}
 h1{color:#1a3a5c;border-bottom:3px solid #1a3a5c;padding-bottom:10px}
 h2{color:#2c5282;margin-top:30px}
 .card{background:white;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.08);padding:20px;margin:15px 0}
 .pass{color:#22543d;background:#c6f6d5;padding:2px 8px;border-radius:4px;font-weight:600}
 .fail{color:#742a2a;background:#fed7d7;padding:2px 8px;border-radius:4px;font-weight:600}
 table{border-collapse:collapse;width:100%}
 th,td{padding:8px;text-align:left;border-bottom:1px solid #e2e8f0}
 th{background:#edf2f7;font-weight:600}
 img{max-width:100%;border-radius:6px;margin:10px 0}
 .fig{text-align:center}
</style></head><body>
<h1>Thermodynamics Engine v4 — Validation Report</h1>
"""

html = [HTML_HEAD]
html.append(f"<div class='card'><h2>{passed}/{len(CHECKS)} checks passed</h2>")
if failed > 0:
    html.append(f"<p><span class='fail'>{failed} failed</span></p>")
else:
    html.append("<p><span class='pass'>ALL GREEN</span></p>")
html.append("</div>")

html.append("<h2>Physics Checks by Category</h2>")
for cat, lst in by_cat.items():
    html.append(f"<div class='card'><h3>{cat} ({sum(1 for c in lst if c.passed)}/{len(lst)})</h3><table>")
    html.append("<tr><th>Check</th><th>Status</th></tr>")
    for c in lst:
        st = "<span class='pass'>PASS</span>" if c.passed else "<span class='fail'>FAIL</span>"
        err = f"<br><small>{c.error}</small>" if not c.passed else ""
        html.append(f"<tr><td>{c.name}{err}</td><td>{st}</td></tr>")
    html.append("</table></div>")

html.append("<h2>Performance Benchmark</h2><div class='card'><table>")
html.append("<tr><th>Operation</th><th>Time (µs/call)</th></tr>")
for k, v in BENCH.items():
    html.append(f"<tr><td>{k}</td><td>{v:.3f}</td></tr>")
html.append("</table></div>")

html.append("<h2>Figures</h2>")
for path in SAVED_FIGS:
    b64 = fig_b64(path)
    fname = os.path.basename(path)
    mime = "image/gif" if fname.endswith(".gif") else "image/png"
    html.append(f"<div class='card fig'><h3>{fname}</h3>"
                f"<img src='data:{mime};base64,{b64}'></div>")

html.append("</body></html>")
report_path = f"{OUT}/report.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(html))

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("\n" + "=" * 78)
print("  FINAL SUMMARY")
print("=" * 78)
print(f"  Physics checks    : {passed}/{len(CHECKS)} passed")
print(f"  Benchmark ops     : {len(BENCH)}")
print(f"  Figures generated : {len(SAVED_FIGS)} (incl. 3D + animation)")
print(f"  Output folder     : {os.path.abspath(OUT)}/")
print(f"  HTML report       : {report_path}")

if failed == 0:
    print("\n  [LOCKED] Full v4 suite — 57/57 checks passed.")
else:
    print(f"\n  [WARN] {failed} check(s) failed — see report.")

print("=" * 78)
print("  Displaying figures — close each window to continue...")
plt.show()
