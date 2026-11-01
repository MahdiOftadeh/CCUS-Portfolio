#!/usr/bin/env python3
"""
================================================================================
 Comprehensive Test & Visualization Suite for Thermodynamics_v0.3.0.py
================================================================================
• Phase 1 — Auto-discovers & runs every unittest class inside the module
• Phase 2 — 50 extended physics validations across 6 thematic sections
• Phase 3 — Generates 10 publication-quality figures for every capability
• Phase 4 — Self-contained HTML report with embedded figures
• Exports JSON diagnostics + verbatim console log

Usage
-----
    python Thermodynamics_v0.3.0_test.py
"""
import os, sys, io, json, time, base64, importlib.util, unittest, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.30,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "legend.framealpha": 0.92,
})
FIGURE_DPI = 130

HERE = Path(__file__).parent.resolve()
MODULE_FILE = HERE / "Thermodynamics_v0.3.0.py"
if not MODULE_FILE.exists():
    cands = sorted(HERE.glob("Thermodynamics_v0.3*.py"))
    if not cands:
        sys.exit(f"ERROR: Thermodynamics_v0.3.0.py not found in {HERE}")
    MODULE_FILE = cands[-1]
    print(f"[i] Using module: {MODULE_FILE.name}")

spec = importlib.util.spec_from_file_location("thermo_v030", str(MODULE_FILE))
thermo = importlib.util.module_from_spec(spec)
sys.modules["thermo_v030"] = thermo
spec.loader.exec_module(thermo)

OUT_DIR = HERE / f"test_output_{MODULE_FILE.stem}"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIGURES = []


def rel_diff(a, b):
    d = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / d


def json_safe(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    raise TypeError(f"not serializable: {type(o)}")


def save_fig(fig, fname, title, caption):
    p = FIG_DIR / fname
    fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append({"name": fname, "title": title, "caption": caption,
                    "path": str(p), "failed": False})
    print(f"   [fig] {fname}")


def safe_fig(fn):
    try:
        fn()
    except Exception:
        err = traceback.format_exc()
        print(f"   [ERR] {fn.__name__} failed")
        FIGURES.append({"name": f"__fail_{fn.__name__}.png",
                        "title": f"⚠ {fn.__name__} failed",
                        "caption": err, "path": None, "failed": True})


# ============================================================================
# PHASE 1 — RUN MODULE'S OWN UNIT TESTS
# ============================================================================
class Collector(unittest.TextTestResult):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows = []
    def _rec(self, t, st, d=""):
        self.rows.append({"class": type(t).__name__,
                          "test": getattr(t, "_testMethodName", str(t)),
                          "status": st, "detail": d})
    def addSuccess(self, t): super().addSuccess(t); self._rec(t, "PASS")
    def addFailure(self, t, e):
        super().addFailure(t, e); self._rec(t, "FAIL", self._exc_info_to_string(e, t))
    def addError(self, t, e):
        super().addError(t, e); self._rec(t, "ERROR", self._exc_info_to_string(e, t))
    def addSkip(self, t, r): super().addSkip(t, r); self._rec(t, "SKIP", r)


def run_unit_tests():
    print("\n" + "=" * 72)
    print("PHASE 1 · UNIT TEST DISCOVERY & EXECUTION")
    print("=" * 72)
    classes = []
    for n in dir(thermo):
        o = getattr(thermo, n)
        if not isinstance(o, type) or not issubclass(o, unittest.TestCase):
            continue
        if o is unittest.TestCase or o.__module__ != thermo.__name__:
            continue
        if o.__dict__.get("__test__", None) is False:
            continue
        classes.append(o)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for c in classes:
        try:
            suite.addTests(loader.loadTestsFromTestCase(c))
        except Exception as e:
            print(f"   [warn] {c.__name__}: {e}")
    print(f"[i] {len(classes)} classes, {suite.countTestCases()} methods\n")
    for c in classes:
        print(f"    • {c.__name__}")
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout,
                                     resultclass=Collector)
    return runner.run(suite)


# ============================================================================
# PHASE 2 — 50 EXTENDED PHYSICS VALIDATIONS
# ============================================================================
def run_physics_validations():
    print("\n" + "=" * 72)
    print("PHASE 2 · 50 EXTENDED PHYSICS VALIDATIONS")
    print("=" * 72)

    T0 = 298.15
    toth = thermo.get_cavenati_toth_system()
    dsl = thermo.get_krishnamurthy_dsl_system()
    iast_toth = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    iast_dsl = thermo.BinaryIAST(dsl["CO2"]["isotherm"], dsl["N2"]["isotherm"])

    results = []
    pass_cnt = 0

    def check(name, fn):
        nonlocal pass_cnt
        try:
            fn()
            results.append({"name": name, "status": "PASS", "detail": ""})
            print(f"   [✓] {name}")
            pass_cnt += 1
        except AssertionError as e:
            results.append({"name": name, "status": "FAIL",
                            "detail": str(e) or "assertion failed"})
            print(f"   [✗] {name}")
        except Exception:
            results.append({"name": name, "status": "ERROR",
                            "detail": traceback.format_exc().splitlines()[-1]})
            print(f"   [!] {name} (error)")

    # ---------------- Section A: pure isotherm physics (V01-V10) -----------
    def v01():
        for sp in ("CO2", "N2", "CH4"):
            iso = toth[sp]["isotherm"]
            for P in (0.01, 0.1, 1.0, 5.0, 20.0):
                for T in (288.15, 298.15, 323.15, 373.15):
                    assert iso.loading(P, T) >= 0.0
    check("V01 · Toth loadings non-negative (all sp × T × P)", v01)

    def v02():
        for sp in ("CO2", "N2"):
            iso = dsl[sp]["isotherm"]
            for P in (0.01, 0.1, 1.0, 5.0, 20.0):
                for T in (288.15, 298.15, 323.15, 373.15):
                    assert iso.loading(P, T) >= 0.0
    check("V02 · DSL loadings non-negative (all sp × T × P)", v02)

    def v03():
        for sp in ("CO2", "N2", "CH4"):
            iso = toth[sp]["isotherm"]
            qs = [iso.loading(p, T0) for p in np.linspace(0.01, 20, 60)]
            assert np.all(np.diff(qs) > 0)
    check("V03 · Toth loading strictly increasing with P", v03)

    def v04():
        for sp in ("CO2", "N2"):
            iso = dsl[sp]["isotherm"]
            qs = [iso.loading(p, T0) for p in np.linspace(0.01, 20, 60)]
            assert np.all(np.diff(qs) > 0)
    check("V04 · DSL loading strictly increasing with P", v04)

    def v05():
        for sp in ("CO2", "N2", "CH4"):
            iso = toth[sp]["isotherm"]
            assert iso.loading(1.0, 293.15) > iso.loading(1.0, 353.15)
    check("V05 · Toth loadings decrease with T (exothermic)", v05)

    def v06():
        for sp in ("CO2", "N2"):
            iso = dsl[sp]["isotherm"]
            assert iso.loading(1.0, 293.15) > iso.loading(1.0, 353.15)
    check("V06 · DSL loadings decrease with T (exothermic)", v06)

    def v07():
        for sp in ("CO2", "N2", "CH4"):
            iso = toth[sp]["isotherm"].at_temperature(T0)
            qP = iso.loading(1e-12) / 1e-12
            assert rel_diff(qP, iso.henry_constant()) < 5e-3
    check("V07 · Henry limit q/P → K_H (Toth)", v07)

    def v08():
        for sp in ("CO2", "N2"):
            iso = dsl[sp]["isotherm"].at_temperature(T0)
            qP = iso.loading(1e-12) / 1e-12
            assert rel_diff(qP, iso.henry_constant()) < 5e-3
    check("V08 · Henry limit q/P → K_H (DSL)", v08)

    def v09():
        iso = toth["CO2"]["isotherm"].at_temperature(T0)
        psi = [iso.spreading_pressure(p) for p in np.logspace(-3, 1.5, 50)]
        assert np.all(np.diff(psi) > 0)
    check("V09 · Spreading pressure strictly increasing with P", v09)

    def v10():
        iso = toth["CO2"]["isotherm"].at_temperature(T0)
        for P0 in (0.05, 0.5, 5.0):
            Pr = iso.invert_spreading_pressure(iso.spreading_pressure(P0))
            assert rel_diff(Pr, P0) < 1e-4
    check("V10 · ψ-inversion roundtrip exact to 1e-4", v10)

    # ---------------- Section B: temperature dependence (V11-V18) ----------
    def v11():
        for sp in ("CO2", "N2"):
            assert hasattr(toth[sp]["isotherm"], "at_temperature")
            assert hasattr(dsl[sp]["isotherm"], "at_temperature")
            _ = toth[sp]["isotherm"].at_temperature(298.15).loading(1.0)
            _ = dsl[sp]["isotherm"].at_temperature(298.15).loading(1.0)
    check("V11 · at_temperature returns usable isotherm objects", v11)

    def v12():
        iso = toth["CO2"]["isotherm"]
        q_low = iso.loading(1.0, 283.15)
        q_mid = iso.loading(1.0, 298.15)
        q_high = iso.loading(1.0, 323.15)
        assert q_low > q_mid > q_high
    check("V12 · TemperatureDependentToth monotone in T", v12)

    def v13():
        iso = dsl["CO2"]["isotherm"]
        q_low = iso.loading(1.0, 283.15)
        q_mid = iso.loading(1.0, 298.15)
        q_high = iso.loading(1.0, 323.15)
        assert q_low > q_mid > q_high
    check("V13 · TemperatureDependentDSL monotone in T", v13)

    def v14():
        # qm(T) with chi behavior (via test module constructor)
        iso_mod = thermo.TemperatureDependentToth(
            qm0=5.5, b0=2.85, t0=0.48, dH=-35.5e3, T0=298.15, chi=0.1)
        qm_low = iso_mod.at_temperature(283.15).qm
        qm_high = iso_mod.at_temperature(323.15).qm
        assert qm_low > qm_high
    check("V14 · qm(T) decays with T when chi > 0", v14)

    def v15():
        # alpha in t(T) makes t drift; verify at_temperature returns valid t
        iso_mod = thermo.TemperatureDependentToth(
            qm0=5.5, b0=2.85, t0=0.48, dH=-35.5e3, T0=298.15, alpha=0.02)
        t_low = iso_mod.at_temperature(283.15).t
        t_high = iso_mod.at_temperature(323.15).t
        assert np.isfinite(t_low) and np.isfinite(t_high)
        assert t_low > 0 and t_high > 0
    check("V15 · t(T) with alpha stays finite & positive", v15)

    def v16():
        iso = toth["CO2"]["isotherm"]
        for P in (0.1, 1.0, 5.0):
            qst = iso.isosteric_heat(P, T0)
            assert np.isfinite(qst) and qst > 0
    check("V16 · isosteric_heat (CC method) finite & positive", v16)

    def v17():
        iso = toth["CO2"]["isotherm"].at_temperature(T0)
        for theta in (0.05, 0.3, 0.7, 0.95):
            q = theta * iso.qm
            qst = iso.get_isosteric_heat(q)
            assert np.isfinite(qst) and qst > 0
    check("V17 · get_isosteric_heat (Cavenati) finite & positive", v17)

    def v18():
        iso = toth["CO2"]["isotherm"]
        cp = iso.get_adsorbed_heat_capacity(P=1.0, T=T0, species="CO2")
        assert np.isfinite(cp) and 20 < cp < 100
    check("V18 · Adsorbed heat capacity in physical band", v18)

    # ---------------- Section C: derivative validation (V19-V24) -----------
    def v19():
        for sp in ("CO2", "N2", "CH4"):
            iso = toth[sp]["isotherm"].at_temperature(T0)
            for P in (0.05, 0.5, 1.0, 5.0, 10.0):
                d_ana = iso.dq_dP(P)
                d_num = iso.dq_dP_numerical(P)
                assert rel_diff(d_ana, d_num) < 1e-5
    check("V19 · Toth dq/dP analytic = Richardson numerical", v19)

    def v20():
        for sp in ("CO2", "N2"):
            iso = dsl[sp]["isotherm"].at_temperature(T0)
            for P in (0.05, 0.5, 1.0, 5.0, 10.0):
                d_ana = iso.dq_dP(P)
                d_num = iso.dq_dP_numerical(P)
                assert rel_diff(d_ana, d_num) < 1e-5
    check("V20 · DSL dq/dP analytic = Richardson numerical", v20)

    def v21():
        for sp in ("CO2", "N2", "CH4"):
            iso = toth[sp]["isotherm"].at_temperature(T0)
            for P in (0.01, 0.1, 1.0, 10.0):
                assert iso.dq_dP(P) > 0
    check("V21 · dq/dP strictly positive everywhere (Toth)", v21)

    def v22():
        for sp in ("CO2", "N2", "CH4"):
            iso = toth[sp]["isotherm"].at_temperature(T0)
            assert rel_diff(iso.dq_dP(0.0), iso.henry_constant()) < 1e-12
    check("V22 · dq/dP at P=0 exactly equals K_H", v22)

    def v23():
        for sp in ("CO2", "N2"):
            iso = toth[sp]["isotherm"].at_temperature(T0)
            d_low = iso.dq_dP(0.01)
            d_high = iso.dq_dP(10.0)
            assert d_low > d_high
    check("V23 · dq/dP decreasing with P (saturation)", v23)

    def v24():
        iso = toth["CO2"]["isotherm"]
        # The monotone-in-T claim for dq/dP holds in the Henry-to-moderate
        # pressure regime (dq/dP ~ qm*b there). Near the saturation elbow,
        # b(T) shifts the effective bP crossover and the trend can invert --
        # this is correct Toth/Langmuir physics, not a bug.
        for P in (0.01, 0.05, 0.1, 0.5):
            d_cold = iso.dq_dP(P, 283.15)
            d_hot = iso.dq_dP(P, 323.15)
            assert d_cold > d_hot
    check("V24 · dq/dP decreases with T in Henry-moderate P regime", v24)

    # ---------------- Section D: IAST & Jacobian (V25-V35) -----------------
    def v25():
        for P in (0.5, 1.0, 5.0):
            for T in (298.15, 323.15):
                q1, q2, _, _ = iast_toth.solve(P, 0.15, T)
                assert q1 > q2
    check("V25 · CO₂ > N₂ loading (Cavenati Toth IAST)", v25)

    def v26():
        q1, q2, x1, _ = iast_toth.solve(1.0, 0.15, T0)
        qt = q1 + q2
        assert rel_diff(q1, qt * x1) < 1e-9
        assert rel_diff(q2, qt * (1 - x1)) < 1e-9
    check("V26 · IAST mass balance q_i = q_tot · x_i", v26)

    def v27():
        for y1 in (0.05, 0.15, 0.30, 0.50):
            _, _, x1, _ = iast_toth.solve(1.0, y1, T0)
            assert x1 > y1
    check("V27 · CO₂ enriched in adsorbed phase (x₁ > y₁)", v27)

    def v28():
        for y1 in (1e-6, 1e-4, 0.5, 1 - 1e-4, 1 - 1e-6):
            q1, q2, x1, psi = iast_toth.solve(1.0, y1, T0)
            assert np.isfinite(q1) and np.isfinite(q2)
            assert np.isfinite(x1) and np.isfinite(psi)
    check("V28 · IAST finite at extreme compositions", v28)

    def v29():
        for P in (1e-6, 1e-3, 0.5, 5.0):
            q1, q2, _, _ = iast_toth.solve(P, 0.15, T0)
            assert np.isfinite(q1) and np.isfinite(q2)
    check("V29 · IAST finite at extreme pressures (VPSA band)", v29)

    def v30():
        q1, q2, _, _ = iast_toth.solve(1.0, 1e-12, T0)
        ref = iast_toth.iso2.at_temperature(T0).loading(1.0)
        assert rel_diff(q2, ref) < 1e-4
        assert q1 < 1e-6
    check("V30 · IAST (y₁→0) recovers pure N₂", v30)

    def v31():
        q1, q2, _, _ = iast_toth.solve(1.0, 1 - 1e-12, T0)
        ref = iast_toth.iso1.at_temperature(T0).loading(1.0)
        assert rel_diff(q1, ref) < 1e-4
        assert q2 < 1e-4
    check("V31 · IAST (y₁→1) recovers pure CO₂", v31)

    def v32():
        jac = iast_toth.thermodynamic_jacobian(1.0, 0.15, T0)
        assert jac["dq1_dP"] > 0
        assert jac["dq2_dP"] > 0
    check("V32 · Jacobian dq_i/dP > 0 for both species", v32)

    def v33():
        jac = iast_toth.thermodynamic_jacobian(1.0, 0.15, T0)
        assert jac["dq1_dT"] < 0
        assert jac["dq2_dT"] < 0
    check("V33 · Jacobian dq_i/dT < 0 for both species", v33)

    def v34():
        jac = iast_toth.thermodynamic_jacobian(1.0, 0.15, T0)
        assert abs(jac["dq1_dP"]) > abs(jac["dq2_dP"])
        assert abs(jac["dq1_dT"]) > abs(jac["dq2_dT"])
    check("V34 · CO₂ dominates Jacobian magnitudes", v34)

    def v35():
        jac = iast_toth.thermodynamic_jacobian(1.0, 0.15, T0)
        for k, v in jac.items():
            assert np.isfinite(v), f"{k} not finite"
    check("V35 · All six Jacobian entries finite", v35)

    # ---------------- Section E: reference database (V36-V45) --------------
    def v36():
        for sp in ("CO2", "N2"):
            assert isinstance(dsl[sp]["isotherm"], thermo.TemperatureDependentDSL)
            assert isinstance(dsl[sp]["isotherm_mass"], thermo.TemperatureDependentDSL)
    check("V36 · Krishnamurthy factory returns DSL + mass", v36)

    def v37():
        for sp in ("CO2", "N2", "CH4"):
            assert isinstance(toth[sp]["isotherm"], thermo.TemperatureDependentToth)
    check("V37 · Cavenati factory returns Toth for 3 species", v37)

    def v38():
        rho = 1120.0
        qv = dsl["CO2"]["isotherm"].loading(1.0, T0)
        qm = dsl["CO2"]["isotherm_mass"].loading(1.0, T0)
        assert rel_diff(qm, qv / rho) < 1e-9
    check("V38 · Mass scaling q_mass = q_vol / ρ exact", v38)

    def v39():
        phys = thermo.ZEOLITE_13X_PHYSICAL
        eps_t = phys["eps_b"] + (1 - phys["eps_b"]) * phys["eps_p"]
        assert rel_diff(phys["eps_t"], eps_t) < 1e-9
        assert phys["rho_b"] < phys["rho_p"]
    check("V39 · Zeolite 13X physical properties consistent", v39)

    def v40():
        for C in (0.1, 1.0, 10.0):
            for T in (280.0, 298.15, 323.15):
                P = thermo.concentration_to_pressure_bar(C, T)
                assert abs(P - C * thermo.R_GAS * T / 1e5) < 1e-12
    check("V40 · Ideal-gas C → P conversion exact", v40)

    def v41():
        for b in (8.65e-7, 2.63e-8, 2.50e-6):
            for T in (288.15, 298.15, 323.15):
                expected = b * 1e5 / (thermo.R_GAS * T)
                got = thermo.b_concentration_to_pressure(b, T)
                assert abs(got - expected) / abs(expected) < 1e-12
    check("V41 · b-concentration → b-pressure identity exact", v41)

    def v42():
        for p in (0.1, 1.0, 5.0, 20.0):
            assert rel_diff(thermo.bar_to_mpa(thermo.mpa_to_bar(p)), p) < 1e-12
    check("V42 · MPa ↔ bar roundtrip exact", v42)

    def v43():
        for q, rho in ((3489.44, 1120.0), (6613.55, 1120.0), (100, 500)):
            got = thermo.loading_volumetric_to_mass(q, rho)
            assert rel_diff(got, q / rho) < 1e-12
    check("V43 · q_vol → q_mass identity exact", v43)

    def v44():
        with self.assertRaises(ValueError) if False else _expect_raises(ValueError):
            thermo.concentration_to_pressure_bar(-1.0, 300.0)
        with _expect_raises(ValueError):
            thermo.loading_volumetric_to_mass(1.0, 0.0)
    check("V44 · Conversion utilities reject negatives", v44)

    def v45():
        for sp in ("CO2", "N2"):
            _ = dsl[sp]["isotherm"].loading(1.0, T0)
            _ = dsl[sp]["isotherm_mass"].loading(1.0, T0)
        for sp in ("CO2", "N2", "CH4"):
            _ = toth[sp]["isotherm"].loading(1.0, T0)
    check("V45 · All factory objects usable at T0", v45)

    # ---------------- Section F: cross-model & cycle (V46-V50) -------------
    def v46():
        for P in (0.5, 1.0, 5.0):
            assert toth["CO2"]["isotherm"].loading(P, T0) > \
                   toth["N2"]["isotherm"].loading(P, T0)
    check("V46 · Cross-model: Cavenati Toth keeps CO₂ > N₂", v46)

    def v47():
        Pa, Pd = 1.5, 0.15
        dq_c = toth["CO2"]["isotherm"].loading(Pa, T0) - \
               toth["CO2"]["isotherm"].loading(Pd, T0)
        assert dq_c > 0
    check("V47 · VPSA working capacity positive", v47)

    def v48():
        for P in (0.1, 1.0, 5.0):
            qst_c = toth["CO2"]["isotherm"].isosteric_heat(P, T0)
            qst_n = toth["N2"]["isotherm"].isosteric_heat(P, T0)
            assert qst_c > qst_n
    check("V48 · q_st(CO₂) > q_st(N₂) (stronger adsorption)", v48)

    def v49():
        qst = toth["CO2"]["isotherm"].isosteric_heat(1.0, T0)
        assert 20e3 < qst < 50e3
    check("V49 · q_st(CO₂) in physically reasonable band", v49)

    def v50():
        P = np.logspace(-2, 1, 30)
        qst = [toth["CO2"]["isotherm"].isosteric_heat(p, T0) for p in P]
        diffs = np.abs(np.diff(qst))
        assert np.all(diffs < 0.3 * np.array(qst[:-1]))
    check("V50 · q_st(P) smooth — no discontinuities", v50)

    print(f"\n[summary] {pass_cnt}/{len(results)} passed")
    return results


class _expect_raises:
    def __init__(self, exc): self.exc = exc
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"expected {self.exc.__name__}")
        return issubclass(exc_type, self.exc)


# ============================================================================
# PHASE 3 — 10 FIGURES
# ============================================================================
def fig01_pure_isotherms():
    toth = thermo.get_cavenati_toth_system()
    dsl = thermo.get_krishnamurthy_dsl_system()
    Tg = [288.15, 298.15, 313.15, 323.15, 348.15]
    P = np.logspace(-3, 2, 200)
    cm = plt.cm.viridis
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.28)
    for j, sp in enumerate(("CO2", "N2", "CH4")):
        ax = fig.add_subplot(gs[0, j]); iso = toth[sp]["isotherm"]
        for k, T in enumerate(Tg):
            ax.plot(P, [iso.loading(p, T) for p in P], lw=1.8,
                    color=cm(k / (len(Tg) - 1)), label=f"{T:.1f} K")
        ax.set_xscale("log"); ax.set_xlabel("P (bar)")
        ax.set_ylabel("q (mol/kg)")
        ax.set_title(f"Toth · {sp} (Cavenati 2004)", fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
    for j, sp in enumerate(("CO2", "N2")):
        ax = fig.add_subplot(gs[1, j]); iso = dsl[sp]["isotherm"]
        for k, T in enumerate(Tg):
            ax.plot(P, [iso.loading(p, T) for p in P], lw=1.8,
                    color=cm(k / (len(Tg) - 1)))
        ax.set_xscale("log"); ax.set_xlabel("P (bar)")
        ax.set_ylabel("q (mol/m³ pellet)")
        ax.set_title(f"DSL · {sp} (Krishnamurthy 2014)", fontweight="bold")
    ax = fig.add_subplot(gs[1, 2]); T = 298.15
    ax.plot(P, [toth["CO2"]["isotherm"].loading(p, T) for p in P],
            lw=2.2, color="C0", label="Toth (Cavenati)")
    ax.plot(P, [dsl["CO2"]["isotherm_mass"].loading(p, T) for p in P],
            lw=2.2, color="C3", ls="--", label="DSL (Krishnamurthy)")
    ax.set_xscale("log"); ax.set_xlabel("P (bar)")
    ax.set_ylabel("q (mol/kg)")
    ax.set_title(f"CO₂ — Cross-Validation @ {T} K", fontweight="bold")
    ax.legend()
    fig.suptitle("Figure 1 · Pure-Component Isotherms — Zeolite 13X",
                 fontsize=15, fontweight="bold", y=0.995)
    save_fig(fig, "01_pure_isotherms.png", "Fig 1 · Pure Isotherms",
             "Toth + DSL across 5 temperatures + cross-validation.")


def fig02_derivative_validation():
    toth = thermo.get_cavenati_toth_system()
    dsl = thermo.get_krishnamurthy_dsl_system()
    P = np.logspace(-3, 1.3, 60); T = 298.15
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        (ax[0, 0], toth["CO2"]["isotherm"].at_temperature(T),
         "Toth CO₂ dq/dP", "C0"),
        (ax[0, 1], dsl["CO2"]["isotherm"].at_temperature(T),
         "DSL CO₂ dq/dP", "C3"),
        (ax[1, 0], toth["N2"]["isotherm"].at_temperature(T),
         "Toth N₂ dq/dP", "C1"),
        (ax[1, 1], dsl["N2"]["isotherm"].at_temperature(T),
         "DSL N₂ dq/dP", "C2"),
    ]
    for a, iso, title, color in panels:
        ana = [iso.dq_dP(p) for p in P]
        num = [iso.dq_dP_numerical(p) for p in P]
        a.plot(P, ana, lw=2.2, color=color, label="Analytical")
        a.scatter(P[::5], num[::5], s=45, color="k",
                  edgecolor="yellow", zorder=5, label="Richardson-4")
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("P (bar)"); a.set_ylabel("dq/dP (mol·kg⁻¹·bar⁻¹)")
        a.set_title(title, fontweight="bold"); a.legend()
    fig.suptitle("Figure 2 · Analytical vs Numerical dq/dP",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "02_derivative_validation.png", "Fig 2 · Derivative Validation",
             "Chain-rule dq/dP coincides with Richardson-4 numerical.")


def fig03_spreading_pressure():
    toth = thermo.get_cavenati_toth_system()
    T = 298.15
    P = np.logspace(-3, 2, 100)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for sp, c in (("CO2", "C0"), ("N2", "C1"), ("CH4", "C2")):
        iso = toth[sp]["isotherm"].at_temperature(T)
        ax[0].plot(P, [iso.spreading_pressure(p) for p in P],
                   lw=2.2, color=c, label=sp)
    ax[0].set_xscale("log"); ax[0].set_xlabel("P (bar)")
    ax[0].set_ylabel("ψ (mol/kg)")
    ax[0].set_title("Reduced spreading pressure", fontweight="bold")
    ax[0].legend()
    iso = toth["CO2"]["isotherm"].at_temperature(T)
    errors = []
    P_tests = np.logspace(-2, 1, 40)
    for P0 in P_tests:
        psi0 = iso.spreading_pressure(P0)
        Pr = iso.invert_spreading_pressure(psi0)
        errors.append(abs(Pr - P0) / P0)
    ax[1].semilogy(P_tests, errors, "o-", lw=2, markersize=6, color="C3")
    ax[1].axhline(1e-4, color="red", ls="--", label="1e-4 tol")
    ax[1].set_xscale("log"); ax[1].set_xlabel("P target (bar)")
    ax[1].set_ylabel("Relative error after inversion")
    ax[1].set_title("ψ-inversion roundtrip error", fontweight="bold")
    ax[1].legend()
    fig.suptitle("Figure 3 · Spreading Pressure & Inversion",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "03_spreading_pressure.png", "Fig 3 · ψ & Inversion",
             "Monotone ψ(P) and exact P → ψ → P⁻¹ recovery.")


def fig04_isosteric_heat_loading():
    toth = thermo.get_cavenati_toth_system()
    T = 298.15
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for sp, c in (("CO2", "C0"), ("N2", "C1")):
        iso = toth[sp]["isotherm"].at_temperature(T)
        qm = iso.qm
        thetas = np.linspace(0.02, 0.98, 40)
        qs = thetas * qm
        qst = [iso.get_isosteric_heat(q) for q in qs]
        ax[0].plot(thetas, np.array(qst) / 1000, lw=2.2, color=c, label=sp)
    ax[0].set_xlabel("Surface coverage θ = q / qm")
    ax[0].set_ylabel("q_st (kJ/mol)")
    ax[0].set_title("Cavenati loading-dependent q_st", fontweight="bold")
    ax[0].legend()
    ax[0].axhspan(20, 45, alpha=0.10, color="green", label="Physical band")
    for sp, c in (("CO2", "C0"), ("N2", "C1")):
        iso = toth[sp]["isotherm"].at_temperature(T)
        qm = iso.qm
        qs = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95]) * qm
        qst = [iso.get_isosteric_heat(q) for q in qs]
        ax[1].bar(np.arange(len(qs)) + (0.2 if sp == "N2" else 0),
                  np.array(qst) / 1000, width=0.35, color=c, label=sp)
    ax[1].set_xticks(np.arange(6) + 0.1)
    ax[1].set_xticklabels(["5%", "20%", "40%", "60%", "80%", "95%"])
    ax[1].set_xlabel("Coverage fraction")
    ax[1].set_ylabel("q_st (kJ/mol)")
    ax[1].set_title("Side-by-side q_st", fontweight="bold")
    ax[1].legend()
    fig.suptitle("Figure 4 · Isosteric Heat (Cavenati correlation)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "04_isosteric_heat_loading.png", "Fig 4 · q_st(q)",
             "Cavenati empirical correlation vs surface coverage.")


def fig05_isosteric_heat_cc():
    toth = thermo.get_cavenati_toth_system()
    P = np.logspace(-2, 1, 50)
    Tg = [288.15, 298.15, 313.15, 323.15]
    cm = plt.cm.plasma
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for k, T in enumerate(Tg):
        qst = [toth["CO2"]["isotherm"].isosteric_heat(p, T) / 1000 for p in P]
        ax[0].plot(P, qst, lw=2, color=cm(k / (len(Tg) - 1)), label=f"{T:.1f} K")
    ax[0].set_xscale("log"); ax[0].set_xlabel("P (bar)")
    ax[0].set_ylabel("q_st (kJ/mol)")
    ax[0].set_title("Toth CO₂ — Clausius-Clapeyron", fontweight="bold")
    ax[0].legend()
    T = 298.15
    for sp, c in (("CO2", "C0"), ("N2", "C1")):
        qst = [toth[sp]["isotherm"].isosteric_heat(p, T) / 1000 for p in P]
        ax[1].plot(P, qst, lw=2.2, color=c, label=sp)
    ax[1].set_xscale("log"); ax[1].set_xlabel("P (bar)")
    ax[1].set_ylabel("q_st (kJ/mol)")
    ax[1].set_title(f"CO₂ vs N₂ @ {T} K", fontweight="bold")
    ax[1].legend()
    fig.suptitle("Figure 5 · Isosteric Heat via Clausius-Clapeyron",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "05_isosteric_heat_cc.png", "Fig 5 · q_st via CC",
             "Numerical Clausius-Clapeyron across T and species.")


def fig06_iast_pressure_sweep():
    toth = thermo.get_cavenati_toth_system()
    iast = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    P = np.logspace(-2, 1.5, 50); y1s = [0.05, 0.15, 0.30, 0.50]; T = 298.15
    cm = plt.cm.viridis
    data = {y: [iast.solve(p, y, T) for p in P] for y in y1s}
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    for k, y in enumerate(y1s):
        c = cm(k / (len(y1s) - 1))
        ax.plot(P, [r[0] for r in data[y]], lw=2, color=c, label=f"CO₂ y={y}")
        ax.plot(P, [r[1] for r in data[y]], lw=1.4, ls=":", color=c)
    ax.set_xscale("log"); ax.set_xlabel("P (bar)"); ax.set_ylabel("q (mol/kg)")
    ax.set_title("IAST loadings vs P", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax = fig.add_subplot(gs[0, 1])
    for k, y in enumerate(y1s):
        c = cm(k / (len(y1s) - 1))
        S = [r[0] / max(r[1], 1e-12) * (1 - y) / y for r in data[y]]
        ax.plot(P, S, lw=2, color=c, label=f"y={y}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("P (bar)"); ax.set_ylabel("S(CO₂/N₂)")
    ax.set_title("Selectivity", fontweight="bold"); ax.legend()
    ax = fig.add_subplot(gs[0, 2])
    for k, y in enumerate(y1s):
        c = cm(k / (len(y1s) - 1))
        ax.plot(P, [r[2] for r in data[y]], lw=2, color=c, label=f"y={y}")
    ax.set_xscale("log"); ax.set_xlabel("P (bar)")
    ax.set_ylabel("x(CO₂) adsorbed")
    ax.set_title(f"Adsorbed composition @ {T} K", fontweight="bold")
    ax.legend()
    fig.suptitle("Figure 6 · IAST Pressure Sweep",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "06_iast_pressure_sweep.png", "Fig 6 · IAST Pressure",
             "Loadings, selectivity, and x(CO₂) across P.")


def fig07_iast_composition_sweep():
    toth = thermo.get_cavenati_toth_system()
    iast = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    y = np.linspace(0.01, 0.99, 60); Ps = [0.1, 0.5, 1.0, 5.0]; T = 298.15
    cm = plt.cm.plasma
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for k, P in enumerate(Ps):
        c = cm(k / (len(Ps) - 1))
        ax[0].plot(y, [iast.solve(P, yi, T)[0] for yi in y], lw=2, color=c,
                   label=f"CO₂ P={P}")
        ax[0].plot(y, [iast.solve(P, yi, T)[1] for yi in y], lw=1.5,
                   ls="--", color=c, label=f"N₂ P={P}")
        S = []
        for yi in y:
            q1, q2, _, _ = iast.solve(P, yi, T)
            S.append(q1 / max(q2, 1e-12) * (1 - yi) / yi)
        ax[1].plot(y, S, lw=2, color=c, label=f"P={P} bar")
    ax[0].set_xlabel("y(CO₂) gas"); ax[0].set_ylabel("q (mol/kg)")
    ax[0].set_title(f"Loadings @ {T} K", fontweight="bold")
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].set_xlabel("y(CO₂)"); ax[1].set_ylabel("S(CO₂/N₂)")
    ax[1].set_yscale("log"); ax[1].set_title("Selectivity", fontweight="bold")
    ax[1].legend()
    fig.suptitle("Figure 7 · IAST Composition Sweep",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "07_iast_composition_sweep.png", "Fig 7 · IAST Composition",
             "Selectivity peaks in the dilute-CO₂ regime.")


def fig08_jacobian_heatmap():
    toth = thermo.get_cavenati_toth_system()
    iast = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    P = np.logspace(-2, 1.3, 20); y = np.linspace(0.05, 0.95, 20); T = 298.15
    keys = ["dq1_dP", "dq2_dP", "dq1_dy", "dq2_dy", "dq1_dT", "dq2_dT"]
    labels = {"dq1_dP": "∂q_CO₂/∂P", "dq2_dP": "∂q_N₂/∂P",
              "dq1_dy": "∂q_CO₂/∂y₁", "dq2_dy": "∂q_N₂/∂y₁",
              "dq1_dT": "∂q_CO₂/∂T", "dq2_dT": "∂q_N₂/∂T"}
    J = {k: np.zeros((len(y), len(P))) for k in keys}
    for i, yi in enumerate(y):
        for j, pj in enumerate(P):
            jc = iast.thermodynamic_jacobian(pj, yi, T)
            for k in keys:
                J[k][i, j] = jc[k]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, k in zip(axes.flat, keys):
        d = J[k]
        vmax = float(np.max(np.abs(d))) or 1e-12
        im = ax.imshow(d, aspect="auto", origin="lower",
                       extent=[P[0], P[-1], y[0], y[-1]],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xscale("log"); ax.set_xlabel("P (bar)"); ax.set_ylabel("y(CO₂)")
        ax.set_title(labels[k], fontweight="bold", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Figure 8 · IAST Thermodynamic Jacobian @ {T} K",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "08_jacobian_heatmap.png", "Fig 8 · Jacobian Heatmap",
             "All six mixture partials across the (P, y) domain.")


def fig09_vpsa_window():
    toth = thermo.get_cavenati_toth_system()
    iast = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    T = 298.15; P = np.logspace(-2, 1.5, 150)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for sp, c in (("CO2", "C0"), ("N2", "C1")):
        ax[0].plot(P, [toth[sp]["isotherm"].loading(p, T) for p in P],
                   lw=2.2, color=c, label=sp)
    Pa, Pd = 1.5, 0.15
    ax[0].axvspan(Pd, Pa, alpha=0.13, color="green",
                  label=f"VPSA window ({Pd}–{Pa} bar)")
    for P_op, mk, fc in ((Pa, "o", "yellow"), (Pd, "s", "orange")):
        for sp in ("CO2", "N2"):
            q = toth[sp]["isotherm"].loading(P_op, T)
            ax[0].scatter([P_op], [q], s=180, marker=mk, zorder=10,
                          edgecolor="k", facecolor=fc)
    ax[0].set_xscale("log"); ax[0].set_xlabel("P (bar)")
    ax[0].set_ylabel("q (mol/kg)")
    ax[0].set_title(f"Pure isotherms + VPSA window @ {T} K", fontweight="bold")
    ax[0].legend()
    y1 = 0.15
    qa1, qa2, _, _ = iast.solve(Pa, y1, T)
    qd1, qd2, _, _ = iast.solve(Pd, y1, T)
    x = np.arange(2); w = 0.35
    ax[1].bar(x - w/2, [qa1, qd1], w, label="CO₂", color="C0", edgecolor="k")
    ax[1].bar(x + w/2, [qa2, qd2], w, label="N₂", color="C1", edgecolor="k")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"Adsorption\n({Pa} bar)", f"Desorption\n({Pd} bar)"])
    ax[1].set_ylabel("q (mol/kg)")
    ax[1].set_title(f"IAST loadings at VPSA endpoints (y_CO₂={y1})",
                    fontweight="bold")
    ax[1].legend()
    fig.suptitle("Figure 9 · VPSA Operating Window",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "09_vpsa_window.png", "Fig 9 · VPSA Window",
             "Adsorption/desorption endpoints with IAST mixture loadings.")


def fig10_dashboard(unit_result, physics_results):
    unit_rows = unit_result.rows
    n_u_pass = sum(1 for r in unit_rows if r["status"] == "PASS")
    n_u_fail = sum(1 for r in unit_rows if r["status"] == "FAIL")
    n_u_err = sum(1 for r in unit_rows if r["status"] == "ERROR")
    n_p_pass = sum(1 for r in physics_results if r["status"] == "PASS")
    n_p_fail = sum(1 for r in physics_results if r["status"] == "FAIL")
    n_p_err = sum(1 for r in physics_results if r["status"] == "ERROR")
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    labels = ["PASS", "FAIL", "ERROR"]
    vals_u = [n_u_pass, n_u_fail, n_u_err]
    colors_u = ["#10b981", "#ef4444", "#f59e0b"]
    bars = ax.bar(labels, vals_u, color=colors_u, edgecolor="k")
    for b, v in zip(bars, vals_u):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, str(v),
                ha="center", fontweight="bold")
    ax.set_title(f"Unit tests ({len(unit_rows)} total)", fontweight="bold")
    ax.set_ylabel("Count")
    ax = fig.add_subplot(gs[0, 1])
    vals_p = [n_p_pass, n_p_fail, n_p_err]
    bars = ax.bar(labels, vals_p, color=colors_u, edgecolor="k")
    for b, v in zip(bars, vals_p):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5, str(v),
                ha="center", fontweight="bold")
    ax.set_title(f"Physics validations ({len(physics_results)} total)",
                 fontweight="bold")
    ax.set_ylabel("Count")
    ax = fig.add_subplot(gs[0, 2])
    total = len(unit_rows) + len(physics_results)
    total_pass = n_u_pass + n_p_pass
    total_fail = n_u_fail + n_p_fail + n_u_err + n_p_err
    ax.pie([total_pass, total_fail], labels=["PASS", "FAIL"],
           colors=["#10b981", "#ef4444"], autopct="%1.1f%%",
           startangle=90, explode=(0.02, 0.02),
           textprops={"fontsize": 12, "fontweight": "bold"})
    ax.set_title(f"Overall ({total} checks, {total_pass} passed)",
                 fontweight="bold")
    fig.suptitle("Figure 10 · Test Results Dashboard",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "10_dashboard.png", "Fig 10 · Dashboard",
             "Summary of all unit tests and physics validations.")


# ============================================================================
# PHASE 4 — HTML REPORT
# ============================================================================
def build_html(unit_result, phys_results, elapsed):
    print("\n" + "=" * 72)
    print("PHASE 4 · BUILDING HTML REPORT")
    print("=" * 72)
    n_u = len(unit_result.rows)
    n_up = sum(1 for r in unit_result.rows if r["status"] == "PASS")
    n_uf = sum(1 for r in unit_result.rows if r["status"] in ("FAIL", "ERROR"))
    n_p = len(phys_results)
    n_pp = sum(1 for r in phys_results if r["status"] == "PASS")
    def sc(s):
        return {"PASS": "ok", "FAIL": "fail", "ERROR": "err"}.get(s, "")
    def render_rows(rows, key_fn):
        out = ""
        for r in rows:
            d = (f"<details><summary>detail</summary><pre>{r['detail']}</pre></details>"
                 if r["detail"] else "")
            out += (f"<tr><td>{key_fn(r)}</td>"
                    f"<td><span class='badge {sc(r['status'])}'>{r['status']}</span></td>"
                    f"<td>{d}</td></tr>")
        return out
    unit_html = render_rows(unit_result.rows,
                            lambda r: f"{r['class']}::{r['test']}")
    phys_html = render_rows(phys_results, lambda r: r["name"])
    fig_html = ""
    for i, f in enumerate(FIGURES, 1):
        if f["failed"]:
            fig_html += (f"<div class='figure-card failed'>"
                         f"<h3>⚠ Figure {i} failed</h3>"
                         f"<pre>{f['caption']}</pre></div>")
            continue
        with open(f["path"], "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        fig_html += (f"<div class='figure-card'>"
                     f"<img src='data:image/png;base64,{b64}' alt='{f['title']}'>"
                     f"<div class='fig-caption'><h3>{f['title']}</h3>"
                     f"<p>{f['caption']}</p></div></div>")
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Thermodynamics_v0.3.0 — Full Test Report</title>
<style>
body{{font-family:-apple-system,"Segoe UI","DejaVu Sans",sans-serif;
background:#f5f7fa;color:#1e293b;margin:0;padding:0;line-height:1.55}}
.container{{max-width:1320px;margin:0 auto;padding:0 24px}}
header.hero{{background:linear-gradient(135deg,#059669 0%,#10b981 100%);
color:#fff;padding:48px 0 40px;margin-bottom:32px}}
header.hero h1{{margin:0;font-size:2.2rem;font-weight:700}}
header.hero .subtitle{{font-size:1.15rem;opacity:0.92;margin-top:4px}}
header.hero .meta{{margin-top:16px;font-size:0.9rem;opacity:0.75}}
.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:16px;margin-bottom:40px}}
.stat{{background:#fff;border-radius:10px;padding:20px;
box-shadow:0 1px 3px rgba(0,0,0,0.06);text-align:center;
border-top:3px solid #3b82f6}}
.stat.success{{border-top-color:#10b981}}
.stat.danger{{border-top-color:#ef4444}}
.stat .num{{display:block;font-size:2rem;font-weight:700}}
.stat .lbl{{display:block;font-size:0.85rem;color:#64748b;
margin-top:6px;text-transform:uppercase}}
section{{margin-bottom:48px}}
section h2{{font-size:1.45rem;font-weight:700;
border-bottom:2px solid #3b82f6;padding-bottom:8px;margin-bottom:20px}}
table{{width:100%;background:#fff;border-collapse:collapse;
border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.06);
font-size:0.92rem}}
th{{background:#f1f5f9;text-align:left;padding:12px 16px;
border-bottom:1px solid #e2e8f0}}
td{{padding:10px 16px;border-bottom:1px solid #e2e8f0}}
.badge{{display:inline-block;padding:2px 10px;border-radius:999px;
font-size:0.75rem;font-weight:600;color:#fff;background:#64748b}}
.badge.ok{{background:#10b981}}
.badge.fail{{background:#ef4444}}
.badge.err{{background:#f59e0b}}
details summary{{cursor:pointer;color:#3b82f6;font-size:0.85rem}}
details pre{{background:#f1f5f9;padding:10px;border-radius:6px;
font-size:0.78rem;overflow:auto;max-height:320px;white-space:pre-wrap}}
.figure-card{{background:#fff;border-radius:12px;
box-shadow:0 1px 3px rgba(0,0,0,0.06);overflow:hidden;margin-bottom:32px}}
.figure-card img{{width:100%;display:block;background:#fff;padding:8px}}
.fig-caption{{padding:16px 20px 20px;border-top:1px solid #e2e8f0}}
.fig-caption h3{{margin:0 0 6px;font-size:1.05rem}}
.fig-caption p{{margin:0;color:#64748b;font-size:0.9rem}}
footer{{text-align:center;padding:40px 0;color:#64748b;font-size:0.85rem;
border-top:1px solid #e2e8f0;margin-top:40px}}
code{{background:#f1f5f9;padding:2px 6px;border-radius:4px}}
</style></head><body>
<header class="hero"><div class="container">
<h1>Thermodynamics_v0.3.0</h1>
<p class="subtitle">Full Test &amp; Visualization Report — 18 unit tests + 50 physics validations</p>
<p class="meta">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
Python {sys.version.split()[0]} · Module: <code>{MODULE_FILE.name}</code></p>
</div></header>
<div class="container">
<section><div class="summary">
<div class="stat"><span class="num">{n_u}</span><span class="lbl">Unit Tests</span></div>
<div class="stat success"><span class="num">{n_up}</span><span class="lbl">Unit Passed</span></div>
<div class="stat danger"><span class="num">{n_uf}</span><span class="lbl">Unit Failed</span></div>
<div class="stat"><span class="num">{n_p}</span><span class="lbl">Physics Checks</span></div>
<div class="stat success"><span class="num">{n_pp}</span><span class="lbl">Physics Passed</span></div>
<div class="stat"><span class="num">{len(FIGURES)}</span><span class="lbl">Figures</span></div>
<div class="stat"><span class="num">{elapsed:.2f}s</span><span class="lbl">Runtime</span></div>
</div></section>
<section><h2>Unit Test Results — {n_up}/{n_u}</h2>
<table><thead><tr><th style="width:35%">Test</th>
<th style="width:10%">Status</th><th>Detail</th></tr></thead>
<tbody>{unit_html}</tbody></table></section>
<section><h2>Physics Validations — {n_pp}/{n_p}</h2>
<table><thead><tr><th style="width:55%">Validation</th>
<th style="width:12%">Status</th><th>Detail</th></tr></thead>
<tbody>{phys_html}</tbody></table></section>
<section><h2>Visualizations — {len([f for f in FIGURES if not f['failed']])} figures</h2>
{fig_html}</section>
</div>
<footer>Generated by <code>Thermodynamics_v0.3.0_test.py</code> ·
Thermodynamic Equilibrium Engine · Zeolite 13X (CO₂/N₂)</footer>
</body></html>"""
    path = OUT_DIR / "Thermodynamics_v0.3.0_report.html"
    path.write_text(html, encoding="utf-8")
    print(f"   [report] {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return str(path)


# ============================================================================
# MAIN
# ============================================================================
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
    def flush(self):
        for st in self.streams: st.flush()

LOG_BUF = io.StringIO()

def main():
    print("=" * 74)
    print("  THERMODYNAMICS_v0.3.0 — FULL TEST & VISUALIZATION SUITE")
    print("=" * 74)
    print(f"[i] Module    : {MODULE_FILE.name}")
    print(f"[i] Output    : {OUT_DIR}")
    print(f"[i] Timestamp : {datetime.now().isoformat(timespec='seconds')}")
    t0 = time.time()

    unit_result = run_unit_tests()
    phys_results = run_physics_validations()

    print("\n" + "=" * 72)
    print("PHASE 3 · GENERATING 10 FIGURES")
    print("=" * 72)
    fig_fns = [
        fig01_pure_isotherms, fig02_derivative_validation, fig03_spreading_pressure,
        fig04_isosteric_heat_loading, fig05_isosteric_heat_cc,
        fig06_iast_pressure_sweep, fig07_iast_composition_sweep,
        fig08_jacobian_heatmap, fig09_vpsa_window,
        lambda: fig10_dashboard(unit_result, phys_results),
    ]
    for i, fn in enumerate(fig_fns, 1):
        print(f"[{i:2d}/{len(fig_fns)}] {getattr(fn, '__name__', 'fig10_dashboard')}")
        safe_fig(fn)

    elapsed = time.time() - t0
    report = build_html(unit_result, phys_results, elapsed)

    diag = {
        "module": MODULE_FILE.name,
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": elapsed,
        "summary": {
            "n_unit": len(unit_result.rows),
            "n_unit_passed": sum(1 for r in unit_result.rows if r["status"] == "PASS"),
            "n_physics": len(phys_results),
            "n_physics_passed": sum(1 for r in phys_results if r["status"] == "PASS"),
            "n_figures": len(FIGURES),
            "n_figures_failed": sum(1 for f in FIGURES if f["failed"]),
        },
        "unit_tests": unit_result.rows,
        "physics_validations": phys_results,
        "figures": [{"name": f["name"], "title": f["title"],
                     "failed": f["failed"]} for f in FIGURES],
    }
    jp = OUT_DIR / "Thermodynamics_v0.3.0_test_results.json"
    jp.write_text(json.dumps(diag, indent=2, default=json_safe), encoding="utf-8")
    print(f"   [json] {jp}")

    lp = OUT_DIR / "Thermodynamics_v0.3.0_console_log.txt"
    lp.write_text(LOG_BUF.getvalue(), encoding="utf-8")
    print(f"   [log]  {lp}")

    n_up = sum(1 for r in unit_result.rows if r["status"] == "PASS")
    n_pp = sum(1 for r in phys_results if r["status"] == "PASS")
    total_tests = len(unit_result.rows) + len(phys_results)
    total_pass = n_up + n_pp
    figs_ok = sum(1 for f in FIGURES if not f["failed"])
    print("\n" + "=" * 74)
    print("  FINAL SUMMARY")
    print("=" * 74)
    print(f"  Unit tests          : {n_up}/{len(unit_result.rows)} passed")
    print(f"  Physics validations : {n_pp}/{len(phys_results)} passed")
    print(f"  TOTAL               : {total_pass}/{total_tests} passed")
    print(f"  Figures             : {figs_ok}/{len(FIGURES)} generated")
    print(f"  Runtime             : {elapsed:.2f}s")
    print(f"  Report              : {report}")
    print("=" * 74)
    ok = (total_pass == total_tests and figs_ok == len(FIGURES))
    print("\n[LOCKED] All checks passed — module is production-ready."
          if ok else "\n[WARN] Some checks failed — review report.html")
    return 0 if ok else 1


if __name__ == "__main__":
    _real_stdout = sys.stdout
    sys.stdout = _Tee(_real_stdout, LOG_BUF)
    try:
        sys.exit(main())
    finally:
        sys.stdout = _real_stdout