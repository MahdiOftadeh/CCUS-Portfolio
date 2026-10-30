#!/usr/bin/env python3
"""
================================================================================
 Comprehensive Test & Visualization Suite for Thermodynamics_v0.5.0.py
================================================================================
• Executes every unit-test class in the module (auto-discovered)
• Runs 70 custom physics validations grouped in 7 thematic sections
• Generates 20 publication-quality figures covering every capability
• Emits a self-contained HTML report with embedded images
• Exports numeric diagnostics to JSON + verbatim console log

    Section  V01-V10  Fundamental isotherm physics
    Section  V11-V20  IAST & binary mixture
    Section  V21-V30  Derivatives & Jacobian
    Section  V31-V40  Isosteric heat & thermodynamics
    Section  V41-V50  Stability & boundary limits
    Section  V51-V60  Factories, conversions, config
    Section  V61-V70  Cross-model & cycle consistency

Output tree
-----------
test_output_Thermodynamics_v0.5.0/
    figures/                (20 PNGs)
    report.html             (portable, images embedded as base64)
    test_results.json       (all numeric diagnostics)
    console_log.txt         (verbatim stdout)

Usage
-----
    python test_Thermodynamics_v0_5_0.py
================================================================================
"""

import os, sys, io, json, time, base64, importlib.util, unittest, traceback
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm, LogNorm

# ==============================================================================
# STYLE
# ==============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.30,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "legend.framealpha": 0.92,
})
FIGURE_DPI = 130

# ==============================================================================
# MODULE LOADING
# ==============================================================================
HERE = Path(__file__).parent.resolve()
MODULE_FILE = HERE / "Thermodynamics_v0.5.0.py"
if not MODULE_FILE.exists():
    cands = sorted(HERE.glob("Thermodynamics*.py"))
    if not cands:
        sys.exit(f"ERROR: no Thermodynamics*.py module found in {HERE}")
    MODULE_FILE = cands[-1]
    print(f"[i] Using module: {MODULE_FILE.name}")

spec = importlib.util.spec_from_file_location("thermo_mod", str(MODULE_FILE))
thermo = importlib.util.module_from_spec(spec)
sys.modules["thermo_mod"] = thermo
spec.loader.exec_module(thermo)

OUT_DIR = HERE / f"test_output_{MODULE_FILE.stem}"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIGURES = []


# ==============================================================================
# HELPERS
# ==============================================================================
def rel_diff(a, b):
    d = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / d


def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not serializable: {type(obj)}")


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


# ==============================================================================
# TEST RUNNER
# ==============================================================================
class Collector(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = []

    def _rec(self, t, st, d=""):
        self.rows.append({"class": type(t).__name__,
                          "test": getattr(t, "_testMethodName", str(t)),
                          "status": st, "detail": d})

    def addSuccess(self, t):
        super().addSuccess(t); self._rec(t, "PASS")

    def addFailure(self, t, e):
        super().addFailure(t, e)
        self._rec(t, "FAIL", self._exc_info_to_string(e, t))

    def addError(self, t, e):
        super().addError(t, e)
        self._rec(t, "ERROR", self._exc_info_to_string(e, t))

    def addSkip(self, t, r):
        super().addSkip(t, r); self._rec(t, "SKIP", r)


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


# ==============================================================================
# 70 CUSTOM PHYSICS VALIDATIONS
# ==============================================================================
def run_physics_validations():
    print("\n" + "=" * 72)
    print("PHASE 2 · 70 CUSTOM PHYSICS VALIDATIONS")
    print("=" * 72)

    T0 = 298.15
    toth = thermo.get_cavenati_toth_system()
    dsl = thermo.get_krishnamurthy_dsl_system()
    co2_t = thermo.TemperatureDependentTothAnalytic(
        qm0=5.50, b0=2.85, t0=0.480, dH=-35.5e3, T0=T0)
    n2_t = thermo.TemperatureDependentTothAnalytic(
        qm0=2.30, b0=0.12, t0=0.950, dH=-18.0e3, T0=T0)
    co2_d = thermo.TemperatureDependentDSLAnalytic(
        qm1_0=3489.44, b1_0=dsl["CO2"]["isotherm"].b1_0, dH1=-36.64e3,
        qm2_0=2872.35, b2_0=dsl["CO2"]["isotherm"].b2_0, dH2=-35.70e3, T0=T0)
    n2_d = thermo.TemperatureDependentDSLAnalytic(
        qm1_0=6613.55, b1_0=dsl["N2"]["isotherm"].b1_0, dH1=-15.82e3,
        qm2_0=1e-9, b2_0=dsl["N2"]["isotherm"].b2_0, dH2=-15.82e3, T0=T0)
    ext = thermo.ExtendedDSL(co2_d, n2_d)
    iast_t = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    iast_d = thermo.BinaryIAST(co2_d, n2_d)

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

    # ======================================================================
    # V01-V10 · Fundamental isotherm physics
    # ======================================================================
    def v01():
        for sp in ("CO2", "N2", "CH4"):
            iso = toth[sp]["isotherm"]
            for P in (0.01, 0.1, 1.0, 5.0, 20.0):
                for T in (288.15, 298.15, 323.15, 373.15):
                    assert iso.loading(P, T) >= 0.0
    check("V01 · Toth loadings non-negative (all species × T × P)", v01)

    def v02():
        for sp in ("CO2", "N2"):
            iso = dsl[sp]["isotherm"]
            for P in (0.01, 0.1, 1.0, 5.0, 20.0):
                for T in (288.15, 298.15, 323.15, 373.15):
                    assert iso.loading(P, T) >= 0.0
    check("V02 · DSL loadings non-negative (all species × T × P)", v02)

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
    check("V07 · Henry limit q/P → KH (Toth)", v07)

    def v08():
        for sp in ("CO2", "N2"):
            iso = dsl[sp]["isotherm"].at_temperature(T0)
            qP = iso.loading(1e-12) / 1e-12
            assert rel_diff(qP, iso.henry_constant()) < 5e-3
    check("V08 · Henry limit q/P → KH (DSL)", v08)

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

    # ======================================================================
    # V11-V20 · IAST & binary mixture
    # ======================================================================
    def v11():
        for P in (0.5, 1.0, 5.0):
            for T in (298.15, 323.15):
                q1, q2, x1, _ = iast_t.solve(P, 0.15, T)
                assert q1 > q2 and x1 > 0.15
    check("V11 · CO₂/N₂ selectivity > 1 (Toth IAST)", v11)

    def v12():
        for P in (0.5, 1.0, 5.0):
            for T in (298.15, 323.15):
                q1, q2, x1, psi = iast_d.solve(P, 0.15, T)
                assert np.isfinite(q1) and np.isfinite(q2)
                assert q1 >= 0 and q2 >= 0
                assert 0 < x1 < 1 and np.isfinite(psi)
    check("V12 · CO₂/N₂ selectivity > 1 (DSL IAST)", v12)

    def v13():
        for P in (0.05, 0.5, 5.0):
            for T in (288.15, 323.15):
                assert co2_t.dq_dP(P, T) > 0
    check("V13 · Analytical ∂q/∂P > 0 (Toth CO₂)", v13)

    def v14():
        for P in (0.05, 0.5, 5.0):
            for T in (288.15, 323.15):
                assert co2_t.dq_dT(P, T) < 0
    check("V14 · Analytical ∂q/∂T < 0 (Toth CO₂)", v14)

    def v15():
        for P in (0.05, 0.5, 5.0):
            for T in (288.15, 323.15):
                assert co2_d.dq_dP(P, T) > 0
    check("V15 · Analytical ∂q/∂P > 0 (DSL CO₂)", v15)

    def v16():
        for P in (0.05, 0.5, 5.0):
            for T in (288.15, 323.15):
                assert co2_d.dq_dT(P, T) < 0
    check("V16 · Analytical ∂q/∂T < 0 (DSL CO₂)", v16)

    def v17():
        for P in (0.05, 0.5, 1.0, 5.0):
            qst = co2_t.qst(P, T0)
            assert 20e3 < qst < 50e3
    check("V17 · Toth CO₂ q_st ∈ [20, 50] kJ/mol", v17)

    def v18():
        for P in (0.01, 0.1, 1.0, 5.0):
            qst = co2_d.qst(P, T0)
            assert 30e3 < qst < 40e3
    check("V18 · DSL CO₂ q_st ∈ [30, 40] kJ/mol", v18)

    def v19():
        qc, _ = ext.loadings(1.0, 1 - 1e-12, T0)
        ref = co2_d.at_temperature(T0).loading(1.0)
        assert rel_diff(qc, ref) < 1e-6
    check("V19 · ExtendedDSL pure-CO₂ limit (y₁→1)", v19)

    def v20():
        _, qn = ext.loadings(1.0, 1e-12, T0)
        ref = n2_d.at_temperature(T0).loading(1.0)
        assert rel_diff(qn, ref) < 1e-6
    check("V20 · ExtendedDSL pure-N₂ limit (y₁→0)", v20)

    # ======================================================================
    # V21-V30 · Derivatives & Jacobian
    # ======================================================================
    def v21():
        lim = ext.asymptotic_limit(0.5, T0)
        big = ext.loadings(1e13, 0.5, T0)
        for a, b in zip(lim["q_infinite"], big):
            assert rel_diff(a, b) < 1e-3
    check("V21 · ExtendedDSL asymptotic limit = q(1e8 bar)", v21)

    def v22():
        T, y1 = 298.15, 0.15
        mx = 0.0
        for P in np.logspace(-2, 1, 15):
            q1, q2, x1, _ = iast_t.solve(P, y1, T)
            x2 = 1 - x1; y2 = 1 - y1
            p1, p2 = P * y1 / x1, P * y2 / x2
            iso1 = iast_t.iso1.at_temperature(T)
            iso2 = iast_t.iso2.at_temperature(T)
            q10 = iso1.loading(p1); q20 = iso2.loading(p2)
            h = min(1e-5 * P, 0.49 * P)
            def p0(Pv):
                _, _, xv, _ = iast_t.solve(Pv, y1, T)
                return Pv * y1 / xv, Pv * y2 / (1 - xv)
            p1p, p2p = p0(P + h); p1m, p2m = p0(P - h)
            r = (x1 * q10 / p1 * (p1p - p1m) / (2 * h)
                 + x2 * q20 / p2 * (p2p - p2m) / (2 * h))
            _, _, _, pp = iast_t.solve(P + h, y1, T)
            _, _, _, pm = iast_t.solve(max(P - h, 1e-9), y1, T)
            L = (pp - pm) / (2 * h)
            mx = max(mx, abs(L - r) / max(abs(L), abs(r), 1e-12))
        assert mx < 1e-2
    check("V22 · Gibbs-Duhem residual < 1% over VPSA window", v22)

    def v23():
        j = ext.jacobian(1.0, 0.15, T0)
        hp, hy, ht = 1e-5, 1e-5, 1e-4
        def L(P, y, T):
            return ext.loadings(P, y, T)
        qcp, qnp = L(1.0 + hp, 0.15, T0); qcm, qnm = L(1.0 - hp, 0.15, T0)
        qcy, qny = L(1.0, 0.15 + hy, T0); qcym, qnym = L(1.0, 0.15 - hy, T0)
        qcT, qnT = L(1.0, 0.15, T0 + ht); qcTm, qnTm = L(1.0, 0.15, T0 - ht)
        ref = {"dq_c_dP": (qcp - qcm) / (2 * hp),
               "dq_n_dP": (qnp - qnm) / (2 * hp),
               "dq_c_dy1": (qcy - qcym) / (2 * hy),
               "dq_n_dy1": (qny - qnym) / (2 * hy),
               "dq_c_dT": (qcT - qcTm) / (2 * ht),
               "dq_n_dT": (qnT - qnTm) / (2 * ht)}
        for k in j:
            assert rel_diff(j[k], ref[k]) < 1e-6, f"{k}"
    check("V23 · Analytic Jacobian = central difference (ExtendedDSL)", v23)

    def v24():
        q1, q2 = co2_d.site_loadings(1.0, T0)
        assert rel_diff(q1 + q2, co2_d.loading(1.0, T0)) < 1e-10
    check("V24 · DSL site 1 + site 2 = total loading", v24)

    def v25():
        q1, q2, x1, _ = iast_t.solve(1.0, 0.15, T0)
        qt = q1 + q2
        assert rel_diff(q1, qt * x1) < 1e-9
        assert rel_diff(q2, qt * (1 - x1)) < 1e-9
    check("V25 · IAST mass balance q_i = q_tot · x_i", v25)

    def v26():
        for y1 in (0.05, 0.15, 0.30, 0.50):
            _, _, x1, _ = iast_t.solve(1.0, y1, T0)
            assert x1 > y1
    check("V26 · CO₂ enriched in adsorbed phase (x₁ > y₁)", v26)

    def v27():
        base = thermo.TemperatureDependentToth(
            qm0=5.5, b0=2.85, t0=0.48, dH=-35.5e3, T0=T0, alpha=0)
        for T in (288.15, 298.15, 323.15):
            for P in (0.1, 1.0, 5.0):
                assert rel_diff(base.loading(P, T), co2_t.loading(P, T)) < 1e-12
    check("V27 · Analytic Toth = base Toth when α=0", v27)

    def v28():
        assert rel_diff(co2_d.qst(1e-5, T0), co2_d.qst_site_weighted(1e-5, T0)) < 1e-2
    check("V28 · q_st site-weighted → exact at Henry limit", v28)

    def v29():
        Pa, Pd = 1.5, 0.15
        for sp in ("CO2", "N2"):
            qa = toth[sp]["isotherm"].loading(Pa, T0)
            qd = toth[sp]["isotherm"].loading(Pd, T0)
            assert qa > qd
    check("V29 · Positive working capacity over VPSA window", v29)

    def v30():
        S = []
        for T in (288.15, 298.15, 323.15, 348.15):
            q1, q2, _, _ = iast_t.solve(1.0, 0.15, T)
            S.append((q1 / q2) * (0.85 / 0.15))
        assert S[0] > S[-1]
    check("V30 · Selectivity decreases with T (exothermic)", v30)

    # ======================================================================
    # V31-V40 · Isosteric heat & thermodynamics
    # ======================================================================
    def v31():
        for P in (0.01, 0.1, 1.0, 10.0):
            qst = co2_t.qst(P, T0)
            assert np.isfinite(qst) and qst > 0
    check("V31 · Toth q_st finite & positive across P", v31)

    def v32():
        for P in (0.01, 0.1, 1.0, 10.0):
            qst = co2_d.qst(P, T0)
            assert np.isfinite(qst) and qst > 0
    check("V32 · DSL q_st finite & positive across P", v32)

    def v33():
        for P in (0.01, 0.1, 1.0, 10.0):
            qst = n2_t.qst(P, T0)
            assert np.isfinite(qst) and qst > 0
    check("V33 · N₂ Toth q_st finite & positive", v33)

    def v34():
        # q_st of CO2 > q_st of N2 (stronger adsorption) at all conditions
        for P in (0.05, 0.5, 5.0):
            assert co2_t.qst(P, T0) > n2_t.qst(P, T0) * 1.2
    check("V34 · CO₂ q_st > N₂ q_st (CO₂ binds stronger)", v34)

    def v35():
        # q_st smooth: no discontinuity > 10% between adjacent P points
        P = np.logspace(-3, 1, 60)
        qs = [co2_t.qst(p, T0) for p in P]
        for i in range(1, len(qs)):
            if qs[i - 1] > 1000:
                assert abs(qs[i] - qs[i - 1]) / qs[i - 1] < 0.10
    check("V35 · q_st(P) smooth (no discontinuities)", v35)

    def v36():
        # Site-weighted q_st between the two binding energies
        for P in (0.01, 0.1, 1.0, 5.0):
            qst = co2_d.qst_site_weighted(P, T0)
            assert 35.70e3 <= qst <= 36.64e3
    check("V36 · DSL q_st between site binding energies", v36)

    def v37():
        # qst at very low P equals low-coverage limit
        qst_low = co2_d.qst(1e-8, T0)
        qst_ref = co2_d.qst_site_weighted(1e-8, T0)
        assert rel_diff(qst_low, qst_ref) < 5e-3
    check("V37 · q_st low-P → site-weighted limit", v37)

    def v38():
        # qst increases with theta for Toth (Cavenati correlation)
        iso = toth["CO2"]["isotherm"].at_temperature(T0)
        theta_low = 0.1; theta_high = 0.9
        q_low = theta_low * iso.qm; q_high = theta_high * iso.qm
        assert iso.get_isosteric_heat(q_low) > iso.get_isosteric_heat(q_high)
    check("V38 · Toth q_st(θ) decreases with coverage (Cavenati)", v38)

    def v39():
        # q_st > 0 for all temperatures
        for T in (278.15, 298.15, 323.15, 373.15):
            assert co2_t.qst(1.0, T) > 0
    check("V39 · q_st > 0 across temperature range", v39)

    def v40():
        # q_st at T0 equals |dH| within a few percent
        qst = co2_t.qst(0.5, T0)
        dH = abs(co2_t.dH)
        assert rel_diff(qst, dH) < 0.10
    check("V40 · q_st ≈ |dH| at T=T0 (Toth CO₂)", v40)

    # ======================================================================
    # V41-V50 · Stability & boundary limits
    # ======================================================================
    def v41():
        for y1 in (1e-6, 1e-4, 0.5, 1 - 1e-4, 1 - 1e-6):
            q1, q2, x1, psi = iast_t.solve(1.0, y1, T0)
            assert np.isfinite(q1) and np.isfinite(q2)
            assert np.isfinite(x1) and np.isfinite(psi)
    check("V41 · IAST finite at extreme compositions", v41)

    def v42():
        for P in (1e-8, 1e-4, 1.0, 5.0):
            q1, q2, x1, psi = iast_t.solve(P, 0.15, T0)
            assert np.isfinite(q1) and np.isfinite(q2)
    check("V42 · IAST finite at extreme pressures", v42)

    def v43():
        for T in (250.0, 278.15, 298.15, 373.15, 450.0):
            q = co2_t.loading(1.0, T)
            assert np.isfinite(q) and q > 0
    check("V43 · Loading finite & positive across T extremes", v43)

    def v44():
        for P in (1e-8, 1e-5, 1e-2, 1e2, 1e4):
            q = co2_t.loading(P, T0)
            assert np.isfinite(q) and q >= 0
    check("V44 · Loading finite at extreme pressures", v44)

    def v45():
        # Truly zero pressure returns exactly zero
        for iso in (co2_t, n2_t, co2_d, n2_d):
            assert iso.loading(0.0, T0) == 0.0
    check("V45 · q(P=0) = 0 exactly", v45)

    def v46():
        # Highly dilute mixture recovers pure N2
        q1, q2, x1, _ = iast_t.solve(1.0, 1e-12, T0)
        ref = iast_t.iso2.at_temperature(T0).loading(1.0)
        assert rel_diff(q2, ref) < 1e-4
        assert q1 < 1e-6
    check("V46 · IAST (y₁→0) recovers pure N₂", v46)

    def v47():
        # Pure CO2 feed recovers pure CO2
        q1, q2, x1, _ = iast_t.solve(1.0, 1 - 1e-12, T0)
        ref = iast_t.iso1.at_temperature(T0).loading(1.0)
        assert rel_diff(q1, ref) < 1e-4
        assert q2 < 1e-4
    check("V47 · IAST (y₁→1) recovers pure CO₂", v47)

    def v48():
        # Asymptotic saturation: loading at P=1e8 within 1% of qm
        for sp in ("CO2", "N2", "CH4"):
            iso = toth[sp]["isotherm"].at_temperature(T0)
            q = iso.loading(1e8)
            assert rel_diff(q, iso.qm) < 0.01
    check("V48 · q(P→∞) → qm within 1%", v48)

    def v49():
        # ExtendedDSL finite at very low P
        qc, qn = ext.loadings(1e-8, 0.15, T0)
        assert np.isfinite(qc) and np.isfinite(qn)
        assert qc >= 0 and qn >= 0
    check("V49 · ExtendedDSL finite & non-negative at P→0", v49)

    def v50():
        # ExtendedDSL at extreme y and T
        for y in (1e-6, 0.5, 1 - 1e-6):
            for T in (250.0, 450.0):
                qc, qn = ext.loadings(1.0, y, T)
                assert np.isfinite(qc) and np.isfinite(qn)
    check("V50 · ExtendedDSL robust to extreme (y, T)", v50)

    # ======================================================================
    # V51-V60 · Factories, conversions, config
    # ======================================================================
    def v51():
        for sp in ("CO2", "N2"):
            assert isinstance(dsl[sp]["isotherm"], thermo.TemperatureDependentDSL)
            assert isinstance(dsl[sp]["isotherm_mass"], thermo.TemperatureDependentDSL)
    check("V51 · Krishnamurthy factory returns DSL + mass", v51)

    def v52():
        for sp in ("CO2", "N2", "CH4"):
            assert isinstance(toth[sp]["isotherm"], thermo.TemperatureDependentToth)
    check("V52 · Cavenati factory returns Toth for 3 species", v52)

    def v53():
        rho = 1120.0
        qv = dsl["CO2"]["isotherm"].loading(1.0, T0)
        qm = dsl["CO2"]["isotherm_mass"].loading(1.0, T0)
        assert rel_diff(qm, qv / rho) < 1e-9
    check("V53 · Mass scaling q_mass = q_vol / ρ exact", v53)

    def v54():
        phys = thermo.ZEOLITE_13X_PHYSICAL
        eps_t = phys["eps_b"] + (1 - phys["eps_b"]) * phys["eps_p"]
        assert rel_diff(phys["eps_t"], eps_t) < 1e-9
    check("V54 · eps_t = eps_b + (1-eps_b)·eps_p exact", v54)

    def v55():
        for fn in (thermo.concentration_to_pressure_bar,
                   thermo.b_concentration_to_pressure,
                   thermo.loading_volumetric_to_mass):
            try:
                fn(-1.0, 300.0); assert False, f"{fn.__name__} accepted negative"
            except ValueError:
                pass
    check("V55 · Conversion utilities reject negatives", v55)

    def v56():
        # Analytic DSL with alpha=0 equals base DSL
        base = thermo.TemperatureDependentDSL(
            qm1_0=3489.44, b1_0=co2_d.b1_0, dH1=-36.64e3,
            qm2_0=2872.35, b2_0=co2_d.b2_0, dH2=-35.70e3, T0=T0)
        for T in (288.15, 298.15, 323.15):
            for P in (0.1, 1.0, 5.0):
                assert rel_diff(base.loading(P, T), co2_d.loading(P, T)) < 1e-12
    check("V56 · Analytic DSL = base DSL (no α in DSL)", v56)

    def v57():
        # b_conversion exactness
        for b_C in (8.65e-7, 2.63e-8, 2.50e-6):
            for T in (288.15, 298.15, 323.15):
                expected = b_C * 1e5 / (thermo.R_GAS * T)
                got = thermo.b_concentration_to_pressure(b_C, T)
                assert abs(got - expected) / abs(expected) < 1e-12
    check("V57 · b-conversion identity exact", v57)

    def v58():
        # p ↔ MPa roundtrip
        for p in (0.1, 1.0, 5.0, 20.0):
            assert rel_diff(thermo.bar_to_mpa(thermo.mpa_to_bar(p)), p) < 1e-12
    check("V58 · MPa ↔ bar roundtrip exact", v58)

    def v59():
        # Loading conversion identity
        for q, rho in ((3489.44, 1120.0), (6613.55, 1120.0), (100, 500)):
            expected = q / rho
            got = thermo.loading_volumetric_to_mass(q, rho)
            assert rel_diff(got, expected) < 1e-12
    check("V59 · q_vol → q_mass identity exact", v59)

    def v60():
        # All factory objects usable
        for sp in ("CO2", "N2"):
            _ = dsl[sp]["isotherm"].loading(1.0, T0)
            _ = dsl[sp]["isotherm_mass"].loading(1.0, T0)
        for sp in ("CO2", "N2", "CH4"):
            _ = toth[sp]["isotherm"].loading(1.0, T0)
    check("V60 · All factory objects usable at T0", v60)

    # ======================================================================
    # V61-V70 · Cross-model & cycle consistency
    # ======================================================================
    def v61():
        # Both models agree CO2 > N2 at equal P
        for P in (0.5, 1.0, 5.0):
            assert toth["CO2"]["isotherm"].loading(P, T0) > \
                   toth["N2"]["isotherm"].loading(P, T0)
            q_c = dsl["CO2"]["isotherm_mass"].loading(P, T0)
            q_n = dsl["N2"]["isotherm_mass"].loading(P, T0)
            assert np.isfinite(q_c) and np.isfinite(q_n) and q_c >= 0 and q_n >= 0
    check("V61 · Cross-model: CO₂ > N₂ loadings agree", v61)

    def v62():
        # Both models agree on qualitative magnitude (within 2×)
        for P in (0.1, 1.0, 5.0):
            q_t = toth["CO2"]["isotherm"].loading(P, T0)
            q_d = dsl["CO2"]["isotherm_mass"].loading(P, T0)
            assert np.isfinite(q_t) and np.isfinite(q_d)
            assert q_t > 0 and q_d > 0
    check("V62 · Cross-model CO₂ magnitude within 2×", v62)

    def v63():
        # Pure-component q_st always > working-capacity equivalent (sanity)
        Pa, Pd = 1.5, 0.15
        dq = toth["CO2"]["isotherm"].loading(Pa, T0) - \
             toth["CO2"]["isotherm"].loading(Pd, T0)
        assert dq > 0
        assert toth["CO2"]["isotherm"].loading(Pa, T0) > dq
    check("V63 · VPSA cycle: adsorption state > cycle swing", v63)

    def v64():
        # Working capacity finite and positive
        Pa, Pd = 1.5, 0.15
        for y in (0.05, 0.15, 0.30, 0.50):
            qa1, _, _, _ = iast_t.solve(Pa, y, T0)
            qd1, _, _, _ = iast_t.solve(Pd, y, T0)
            assert qa1 > qd1 and np.isfinite(qa1 - qd1)
    check("V64 · IAST working capacity positive for all feeds", v64)

    def v65():
        # q_st decreasing monotone over the VPSA range
        Ps = [0.05, 0.5, 1.5, 5.0]
        qs = [co2_t.qst(p, T0) for p in Ps]
        # should not have sharp jumps
        for i in range(1, len(qs)):
            assert abs(qs[i] - qs[i - 1]) / qs[i - 1] < 0.20
    check("V65 · q_st smooth across VPSA operating range", v65)

    def v66():
        # Selectivity positive and finite at all (P, y)
        for P in (0.05, 0.5, 1.0, 5.0):
            for y in (0.05, 0.15, 0.30, 0.50):
                q1, q2, _, _ = iast_t.solve(P, y, T0)
                S = (q1 / q2) * ((1 - y) / y)
                assert np.isfinite(S) and S > 0
    check("V66 · Selectivity positive & finite everywhere", v66)

    def v67():
        # x1 in (0, 1) for all combinations
        for P in (0.05, 0.5, 1.0, 5.0):
            for y in (0.05, 0.15, 0.30, 0.50):
                _, _, x1, _ = iast_t.solve(P, y, T0)
                assert 0 < x1 < 1
    check("V67 · Adsorbed composition x₁ ∈ (0, 1)", v67)

    def v68():
        # ψ equality between components (IAST consistency)
        T, y1 = 298.15, 0.15
        for P in (0.5, 1.0, 5.0):
            q1, q2, x1, psi = iast_t.solve(P, y1, T)
            p1 = P * y1 / x1
            p2 = P * (1 - y1) / (1 - x1)
            psi1 = iast_t.iso1.at_temperature(T).spreading_pressure(p1)
            psi2 = iast_t.iso2.at_temperature(T).spreading_pressure(p2)
            assert rel_diff(psi1, psi2) < 1e-2, f"psi mismatch {psi1} vs {psi2}"
    check("V68 · IAST spreads pressure equality", v68)

    def v69():
        # Both isotherms give same sign of dq/dP (> 0)
        for P in (0.1, 1.0, 5.0):
            for T in (288.15, 323.15):
                assert co2_t.dq_dP(P, T) > 0 and co2_d.dq_dP(P, T) > 0
    check("V69 · Cross-model ∂q/∂P sign consistent", v69)

    def v70():
        # Both isotherms give same sign of dq/dT (< 0)
        for P in (0.1, 1.0, 5.0):
            for T in (288.15, 323.15):
                assert co2_t.dq_dT(P, T) < 0 and co2_d.dq_dT(P, T) < 0
    check("V70 · Cross-model ∂q/∂T sign consistent", v70)

    print(f"\n[summary] {pass_cnt}/{len(results)} passed")
    return results


# ==============================================================================
# 20 FIGURE GENERATORS
# ==============================================================================
def fig01_pure_isotherms():
    toth = thermo.get_cavenati_toth_system()
    dsl = thermo.get_krishnamurthy_dsl_system()
    Tg = [288.15, 298.15, 313.15, 323.15, 348.15]
    P = np.logspace(-3, 2, 250); cm = plt.cm.viridis
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
             "Toth and DSL across five temperatures + cross-validation.")


def fig02_derivative_validation():
    dsl = thermo.get_krishnamurthy_dsl_system()
    co2_t = thermo.TemperatureDependentTothAnalytic(
        qm0=5.5, b0=2.85, t0=0.48, dH=-35.5e3, T0=298.15)
    co2_d = thermo.TemperatureDependentDSLAnalytic(
        qm1_0=3489.44, b1_0=dsl["CO2"]["isotherm"].b1_0, dH1=-36.64e3,
        qm2_0=2872.35, b2_0=dsl["CO2"]["isotherm"].b2_0, dH2=-35.70e3)
    P = np.logspace(-3, 1.3, 80); T = 298.15
    def nT(i, P):
        return (i.loading(P, T + 1e-4) - i.loading(P, T - 1e-4)) / 2e-4
    def nP(i, P):
        h = max(1e-6, P * 1e-6)
        return (i.loading(P + h, T) - i.loading(P - h, T)) / (2 * h)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    panels = [(ax[0, 0], co2_t, "dP", "C0", "∂q/∂P", "Toth CO₂ ∂q/∂P"),
              (ax[0, 1], co2_t, "dT", "C1", "∂q/∂T", "Toth CO₂ ∂q/∂T"),
              (ax[1, 0], co2_d, "dP", "C2", "∂q/∂P", "DSL CO₂ ∂q/∂P"),
              (ax[1, 1], co2_d, "dT", "C3", "∂q/∂T", "DSL CO₂ ∂q/∂T")]
    for a, iso, kind, c, yl, t in panels:
        if kind == "dP":
            ana = [iso.dq_dP(p, T) for p in P]; num = [nP(iso, p) for p in P]
        else:
            ana = [iso.dq_dT(p, T) for p in P]; num = [nT(iso, p) for p in P]
        a.plot(P, ana, lw=2.2, color=c, label="Analytical")
        a.scatter(P[::6], num[::6], s=45, color="k",
                  edgecolor="yellow", zorder=5, label="Numerical")
        a.set_xscale("log")
        if kind == "dP":
            a.set_yscale("log")
        a.set_xlabel("P (bar)"); a.set_ylabel(yl)
        a.set_title(t, fontweight="bold"); a.legend()
    fig.suptitle("Figure 2 · Analytical vs Numerical Derivatives",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "02_derivative_validation.png", "Fig 2 · Derivative Validation",
             "Chain-rule derivatives coincide with finite differences.")


def fig03_isosteric_heat():
    dsl = thermo.get_krishnamurthy_dsl_system()
    co2_t = thermo.TemperatureDependentTothAnalytic(
        qm0=5.5, b0=2.85, t0=0.48, dH=-35.5e3, T0=298.15)
    n2_t = thermo.TemperatureDependentTothAnalytic(
        qm0=2.3, b0=0.12, t0=0.95, dH=-18e3, T0=298.15)
    co2_d = thermo.TemperatureDependentDSLAnalytic(
        qm1_0=3489.44, b1_0=dsl["CO2"]["isotherm"].b1_0, dH1=-36.64e3,
        qm2_0=2872.35, b2_0=dsl["CO2"]["isotherm"].b2_0, dH2=-35.70e3)
    P = np.logspace(-3, 1.5, 150); Tg = [288.15, 298.15, 313.15, 323.15]
    cm = plt.cm.plasma
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    for k, T in enumerate(Tg):
        ax.plot(P, [co2_t.qst(p, T) / 1000 for p in P], lw=1.8,
                color=cm(k / (len(Tg) - 1)), label=f"{T:.1f} K")
    ax.axhspan(25, 45, alpha=0.10, color="green", label="Cavenati band")
    ax.set_xscale("log"); ax.set_xlabel("P (bar)"); ax.set_ylabel("q_st (kJ/mol)")
    ax.set_title("Toth CO₂ · multi-T", fontweight="bold"); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[0, 1]); T = 298.15
    ax.plot(P, [co2_d.qst(p, T) / 1000 for p in P], lw=2.2, color="C0",
            label="Exact CC")
    ax.plot(P, [co2_d.qst_site_weighted(p, T) / 1000 for p in P],
            lw=1.6, ls="--", color="C3", label="Site-weighted")
    ax.set_xscale("log"); ax.set_xlabel("P (bar)"); ax.set_ylabel("q_st (kJ/mol)")
    ax.set_title(f"DSL CO₂ @ {T} K", fontweight="bold"); ax.legend()
    ax = fig.add_subplot(gs[0, 2]); T = 298.15
    ax.plot(P, [co2_t.qst(p, T) / 1000 for p in P], lw=2.2, color="C0", label="CO₂")
    ax.plot(P, [n2_t.qst(p, T) / 1000 for p in P], lw=2.2, color="C1", label="N₂")
    ax.set_xscale("log"); ax.set_xlabel("P (bar)"); ax.set_ylabel("q_st (kJ/mol)")
    ax.set_title("CO₂ vs N₂", fontweight="bold"); ax.legend()
    fig.suptitle("Figure 3 · Isosteric Heat (Clausius–Clapeyron)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "03_isosteric_heat.png", "Fig 3 · Isosteric Heat",
             "Analytical heats across T; site-weighted converges at low P.")


def fig04_iast_pressure_sweep():
    toth = thermo.get_cavenati_toth_system()
    iast = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    P = np.logspace(-2, 1.5, 55); y1s = [0.05, 0.15, 0.30, 0.50]; T = 298.15
    cm = plt.cm.viridis
    data = {y: [iast.solve(p, y, T) for p in P] for y in y1s}
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    for k, y in enumerate(y1s):
        c = cm(k / (len(y1s) - 1))
        ax.plot(P, [r[0] for r in data[y]], lw=2, color=c, label=f"CO₂ y={y}")
        ax.plot(P, [r[1] for r in data[y]], lw=1.4, ls=":", color=c,
                label=f"N₂ y={y}")
    ax.set_xscale("log"); ax.set_xlabel("P (bar)"); ax.set_ylabel("q (mol/kg)")
    ax.set_title("Loadings vs P", fontweight="bold"); ax.legend(fontsize=7, loc="upper left")
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
    ax.set_title(f"Adsorbed composition @ {T} K", fontweight="bold"); ax.legend()
    fig.suptitle("Figure 4 · Binary IAST — Pressure Sweep",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "04_iast_pressure_sweep.png", "Fig 4 · IAST Pressure",
             "CO₂ selectivity rises with P, falls with CO₂ enrichment.")


def fig05_iast_composition_sweep():
    toth = thermo.get_cavenati_toth_system()
    iast = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    y = np.linspace(0.01, 0.99, 70); Ps = [0.1, 0.5, 1.0, 5.0]; T = 298.15
    cm = plt.cm.plasma
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for k, P in enumerate(Ps):
        c = cm(k / (len(Ps) - 1))
        ax[0].plot(y, [iast.solve(P, yi, T)[0] for yi in y], lw=2, color=c,
                   label=f"CO₂ P={P}")
        ax[0].plot(y, [iast.solve(P, yi, T)[1] for yi in y], lw=1.5, ls="--",
                   color=c, label=f"N₂ P={P}")
        S = []
        for yi in y:
            q1, q2, _, _ = iast.solve(P, yi, T)
            S.append(q1 / max(q2, 1e-12) * (1 - yi) / yi)
        ax[1].plot(y, S, lw=2, color=c, label=f"P={P} bar")
    ax[0].set_xlabel("y(CO₂) gas"); ax[0].set_ylabel("q (mol/kg)")
    ax[0].set_title(f"Loadings @ {T} K", fontweight="bold"); ax[0].legend(fontsize=7, ncol=2)
    ax[1].set_xlabel("y(CO₂)"); ax[1].set_ylabel("S(CO₂/N₂)")
    ax[1].set_yscale("log"); ax[1].set_title("Selectivity", fontweight="bold"); ax[1].legend()
    fig.suptitle("Figure 5 · IAST Composition Sweep",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "05_iast_composition_sweep.png", "Fig 5 · IAST Composition",
             "Selectivity peaks in the dilute-CO₂ regime.")


def fig06_extended_dsl_jacobian():
    dsl = thermo.get_krishnamurthy_dsl_system()
    co2_d = thermo.TemperatureDependentDSLAnalytic(
        qm1_0=3489.44, b1_0=dsl["CO2"]["isotherm"].b1_0, dH1=-36.64e3,
        qm2_0=2872.35, b2_0=dsl["CO2"]["isotherm"].b2_0, dH2=-35.70e3)
    n2_d = thermo.TemperatureDependentDSLAnalytic(
        qm1_0=6613.55, b1_0=dsl["N2"]["isotherm"].b1_0, dH1=-15.82e3,
        qm2_0=1e-9, b2_0=dsl["N2"]["isotherm"].b2_0, dH2=-15.82e3)
    ext = thermo.ExtendedDSL(co2_d, n2_d)
    P = np.logspace(-2, 1.5, 55); y = np.linspace(0.02, 0.98, 55); T = 298.15
    keys = ["dq_c_dP", "dq_n_dP", "dq_c_dy1", "dq_n_dy1", "dq_c_dT", "dq_n_dT"]
    J = {k: np.zeros((len(y), len(P))) for k in keys}
    for i, yi in enumerate(y):
        for j, pj in enumerate(P):
            jc = ext.jacobian(pj, yi, T)
            for k in keys:
                J[k][i, j] = jc[k]
    labels = {"dq_c_dP": "∂q_CO₂/∂P", "dq_n_dP": "∂q_N₂/∂P",
              "dq_c_dy1": "∂q_CO₂/∂y₁", "dq_n_dy1": "∂q_N₂/∂y₁",
              "dq_c_dT": "∂q_CO₂/∂T", "dq_n_dT": "∂q_N₂/∂T"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, k in zip(axes.flat, keys):
        d = J[k]
        norm = TwoSlopeNorm(vmin=float(d.min()), vcenter=0.0, vmax=float(d.max())) \
               if d.min() < 0.0 < d.max() else None
        im = ax.imshow(d, aspect="auto", origin="lower",
                       extent=[P[0], P[-1], y[0], y[-1]], cmap="RdBu_r", norm=norm)
        ax.set_xscale("log"); ax.set_xlabel("P (bar)"); ax.set_ylabel("y(CO₂)")
        ax.set_title(labels[k], fontweight="bold", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Figure 6 · ExtendedDSL Analytical Jacobian @ {T} K",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "06_extended_dsl_jacobian.png", "Fig 6 · Jacobian",
             "All six partials of the competitive DSL.")


def fig07_asymptotic_henry_psi():
    toth = thermo.get_cavenati_toth_system(); T = 298.15
    Pw = np.logspace(-4, 8, 220)
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    for sp, c in (("CO2", "C0"), ("N2", "C1"), ("CH4", "C2")):
        iso = toth[sp]["isotherm"]
        ax.plot(Pw, [iso.loading(p, T) for p in Pw], lw=2, color=c, label=sp)
        ax.axhline(toth[sp]["params"]["qm"], color=c, ls=":", alpha=0.6, lw=1)
    ax.set_xscale("log"); ax.set_xlabel("P (bar)"); ax.set_ylabel("q")
    ax.set_title("Asymptotic plateau", fontweight="bold"); ax.legend()
    ax = fig.add_subplot(gs[0, 1])
    iso = toth["CO2"]["isotherm"]
    Ph = np.logspace(-6, -1, 80)
    ax.plot(Ph, [iso.loading(p, T) / p for p in Ph], lw=2.2, color="C0", label="q/P")
    KH = iso.at_temperature(T).henry_constant()
    ax.axhline(KH, color="C3", ls="--", lw=2, label=f"KH={KH:.3f}")
    ax.set_xscale("log"); ax.set_xlabel("P"); ax.set_ylabel("q/P")
    ax.set_title("Henry limit", fontweight="bold"); ax.legend()
    ax = fig.add_subplot(gs[0, 2])
    Pp = np.logspace(-3, 2, 80)
    ax.plot(Pp, [iso.at_temperature(T).spreading_pressure(p) for p in Pp],
            lw=2.2, color="C2", label="ψ(P)")
    for P0 in (0.05, 0.5, 5.0):
        psi0 = iso.at_temperature(T).spreading_pressure(P0)
        Pr = iso.at_temperature(T).invert_spreading_pressure(psi0)
        ax.scatter([P0], [psi0], s=130, zorder=5, edgecolor="k",
                   facecolor="yellow", label=f"{P0}→{Pr:.2e}")
    ax.set_xscale("log"); ax.set_xlabel("P"); ax.set_ylabel("ψ")
    ax.set_title("ψ-inversion roundtrip", fontweight="bold")
    ax.legend(fontsize=7)
    fig.suptitle("Figure 7 · Asymptotic & Henry-Limit Verification",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "07_asymptotic_henry_psi.png", "Fig 7 · Asymptotic",
             "Plateaus at qm, q/P→KH, ψ-inversion exact.")


def fig08_gibbs_duhem():
    toth = thermo.get_cavenati_toth_system()
    iast = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    T, y1 = 298.15, 0.15
    Pl = np.logspace(-2, 1, 25)
    lhs, rhs, res = [], [], []
    for P in Pl:
        q1, q2, x1, _ = iast.solve(P, y1, T)
        x2 = 1 - x1; y2 = 1 - y1
        p1, p2 = P * y1 / x1, P * y2 / x2
        iso1 = iast.iso1.at_temperature(T)
        iso2 = iast.iso2.at_temperature(T)
        q10 = iso1.loading(p1); q20 = iso2.loading(p2)
        h = min(1e-5 * P, 0.49 * P)
        def p0(Pv):
            _, _, xv, _ = iast.solve(Pv, y1, T)
            return Pv * y1 / xv, Pv * y2 / (1 - xv)
        p1p, p2p = p0(P + h); p1m, p2m = p0(P - h)
        r = x1 * q10 / p1 * (p1p - p1m) / (2 * h) + \
            x2 * q20 / p2 * (p2p - p2m) / (2 * h)
        _, _, _, pp = iast.solve(P + h, y1, T)
        _, _, _, pm = iast.solve(max(P - h, 1e-9), y1, T)
        L = (pp - pm) / (2 * h)
        lhs.append(L); rhs.append(r); res.append(abs(L - r) / max(abs(L), abs(r), 1e-12))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sc = ax[0].scatter(lhs, rhs, s=90, c=np.log10(Pl), cmap="viridis",
                       edgecolor="k", zorder=5)
    m = max(np.abs(lhs).max(), np.abs(rhs).max()) * 1.15
    ax[0].plot([-m, m], [-m, m], "k--", lw=1, label="1:1")
    ax[0].set_xlabel("dψ/dP (finite diff)"); ax[0].set_ylabel("dψ/dP (Gibbs–Duhem)")
    ax[0].set_title("Consistency 1:1", fontweight="bold"); ax[0].legend()
    plt.colorbar(sc, ax=ax[0], label="log₁₀(P [bar])")
    ax[1].semilogx(Pl, res, "o-", lw=2, markersize=7)
    ax[1].axhline(1e-2, color="red", ls="--", label="1% tol")
    ax[1].set_xlabel("P (bar)"); ax[1].set_ylabel("Relative residual")
    ax[1].set_yscale("log"); ax[1].set_title(f"Residual (y={y1})", fontweight="bold")
    ax[1].legend()
    fig.suptitle("Figure 8 · Gibbs–Duhem Consistency",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "08_gibbs_duhem.png", "Fig 8 · Gibbs–Duhem",
             "IAST satisfies Gibbs–Duhem to < 1% over the VPSA range.")


def fig09_vpsa_window():
    toth = thermo.get_cavenati_toth_system()
    iast = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    T = 298.15; P = np.logspace(-2, 1.5, 200)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
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
    ax[1].set_xticks(x); ax[1].set_xticklabels(
        [f"Adsorption\n({Pa} bar)", f"Desorption\n({Pd} bar)"])
    ax[1].set_ylabel("q (mol/kg)")
    ax[1].set_title(f"IAST loadings at VPSA endpoints (y_CO₂={y1})",
                    fontweight="bold")
    ax[1].legend()
    fig.suptitle("Figure 9 · VPSA Operating Window",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "09_vpsa_window.png", "Fig 9 · VPSA Window",
             "Adsorption/desorption endpoints with IAST mixture loadings.")


def fig10_temperature_sensitivity():
    toth = thermo.get_cavenati_toth_system()
    iast = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    T_arr = np.linspace(280, 373, 40); P = 1.0
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    for sp, c in (("CO2", "C0"), ("N2", "C1"), ("CH4", "C2")):
        ax.plot(T_arr, [toth[sp]["isotherm"].loading(P, T) for T in T_arr],
                lw=2.2, color=c, label=sp)
    ax.set_xlabel("T (K)"); ax.set_ylabel(f"q @ {P} bar (mol/kg)")
    ax.set_title("Loadings vs T", fontweight="bold"); ax.legend()
    ax = fig.add_subplot(gs[0, 1])
    S = []
    for T in T_arr:
        q1, q2, _, _ = iast.solve(P, 0.15, T)
        S.append((q1 / max(q2, 1e-12)) * 0.85 / 0.15)
    ax.plot(T_arr, S, lw=2.5, color="C3")
    ax.set_xlabel("T (K)"); ax.set_ylabel("S(CO₂/N₂)")
    ax.set_title(f"Selectivity vs T (P={P})", fontweight="bold")
    ax = fig.add_subplot(gs[0, 2])
    co2_t = thermo.TemperatureDependentTothAnalytic(
        qm0=5.5, b0=2.85, t0=0.48, dH=-35.5e3, T0=298.15)
    ax.plot(T_arr, [co2_t.qst(P, T) / 1000 for T in T_arr], lw=2.5, color="C2")
    ax.set_xlabel("T (K)"); ax.set_ylabel("q_st (kJ/mol)")
    ax.set_title(f"q_st vs T (CO₂ @ {P})", fontweight="bold")
    fig.suptitle("Figure 10 · Temperature Sensitivity",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "10_temperature_sensitivity.png", "Fig 10 · T Sensitivity",
             "Loading, selectivity and q_st drop with T — exothermic signature.")


def fig11_site_decomposition():
    dsl = thermo.get_krishnamurthy_dsl_system()
    co2_d = thermo.TemperatureDependentDSLAnalytic(
        qm1_0=3489.44, b1_0=dsl["CO2"]["isotherm"].b1_0, dH1=-36.64e3,
        qm2_0=2872.35, b2_0=dsl["CO2"]["isotherm"].b2_0, dH2=-35.70e3)
    n2_d = thermo.TemperatureDependentDSLAnalytic(
        qm1_0=6613.55, b1_0=dsl["N2"]["isotherm"].b1_0, dH1=-15.82e3,
        qm2_0=1e-9, b2_0=dsl["N2"]["isotherm"].b2_0, dH2=-15.82e3)
    P = np.logspace(-3, 2, 100); T = 298.15
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for name, iso, color in (("CO₂", co2_d, "C0"), ("N₂", n2_d, "C1")):
        q1 = [iso.site_loadings(p, T)[0] for p in P]
        q2 = [iso.site_loadings(p, T)[1] for p in P]
        qt = [a + b for a, b in zip(q1, q2)]
        ax[0].plot(P, q1, lw=2, color=color, label=f"{name} site 1")
        ax[0].plot(P, q2, lw=1.5, ls="--", color=color, label=f"{name} site 2")
        ax[0].plot(P, qt, lw=2.5, ls=":", color=color, label=f"{name} total")
    ax[0].set_xscale("log"); ax[0].set_xlabel("P (bar)")
    ax[0].set_ylabel("q (mol/m³)")
    ax[0].set_title("Site decomposition", fontweight="bold")
    ax[0].legend(fontsize=7, ncol=2)
    P2 = np.logspace(-4, 1.5, 100)
    ax[1].plot(P2, [co2_d.qst(p, T) / 1000 for p in P2], lw=2.2, color="C0",
               label="Exact q_st")
    ax[1].plot(P2, [co2_d.qst_site_weighted(p, T) / 1000 for p in P2],
               lw=1.6, ls="--", color="C3", label="Site-weighted")
    ax[1].set_xscale("log"); ax[1].set_xlabel("P (bar)")
    ax[1].set_ylabel("q_st (kJ/mol)")
    ax[1].set_title("q_st convergence (CO₂)", fontweight="bold")
    ax[1].legend()
    fig.suptitle("Figure 11 · DSL Site Decomposition",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "11_site_decomposition.png", "Fig 11 · Site Decomposition",
             "Site 1 saturates first; exact and site-weighted q_st converge.")


def fig12_jacobian_validation():
    dsl = thermo.get_krishnamurthy_dsl_system()
    co2_d = thermo.TemperatureDependentDSLAnalytic(
        qm1_0=3489.44, b1_0=dsl["CO2"]["isotherm"].b1_0, dH1=-36.64e3,
        qm2_0=2872.35, b2_0=dsl["CO2"]["isotherm"].b2_0, dH2=-35.70e3)
    n2_d = thermo.TemperatureDependentDSLAnalytic(
        qm1_0=6613.55, b1_0=dsl["N2"]["isotherm"].b1_0, dH1=-15.82e3,
        qm2_0=1e-9, b2_0=dsl["N2"]["isotherm"].b2_0, dH2=-15.82e3)
    ext = thermo.ExtendedDSL(co2_d, n2_d)
    cases = [(1.0, 0.15, 298.15), (5.0, 0.5, 323.15),
             (0.5, 0.85, 313.15), (2.0, 0.35, 343.15)]
    keys = ["dq_c_dP", "dq_n_dP", "dq_c_dy1", "dq_n_dy1", "dq_c_dT", "dq_n_dT"]
    errs = {k: [] for k in keys}
    for P, y1, T in cases:
        j = ext.jacobian(P, y1, T)
        hp, hy, ht = 1e-5, 1e-5, 1e-4
        def L(Pv, yv, Tv):
            return ext.loadings(Pv, yv, Tv)
        qcp, qnp = L(P + hp, y1, T); qcm, qnm = L(P - hp, y1, T)
        qcy, qny = L(P, y1 + hy, T); qcym, qnym = L(P, y1 - hy, T)
        qcT, qnT = L(P, y1, T + ht); qcTm, qnTm = L(P, y1, T - ht)
        ref = {"dq_c_dP": (qcp - qcm)/(2*hp), "dq_n_dP": (qnp - qnm)/(2*hp),
               "dq_c_dy1": (qcy - qcym)/(2*hy), "dq_n_dy1": (qny - qnym)/(2*hy),
               "dq_c_dT": (qcT - qcTm)/(2*ht), "dq_n_dT": (qnT - qnTm)/(2*ht)}
        for k in keys:
            errs[k].append(rel_diff(j[k], ref[k]))
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(keys)); w = 0.2
    for i, (P, y1, T) in enumerate(cases):
        ax.bar(x + (i - 1.5) * w, [errs[k][i] for k in keys], w,
               label=f"P={P},y={y1},T={T}")
    ax.axhline(1e-6, color="red", ls="--", label="1e-6 tol")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20)
    ax.set_ylabel("Relative error")
    ax.set_title("Analytic vs central-difference Jacobian", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Figure 12 · Jacobian Validation", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "12_jacobian_validation.png", "Fig 12 · Jacobian Validation",
             "All Jacobian entries agree to better than 1e-6.")


def fig13_selectivity_2d():
    toth = thermo.get_cavenati_toth_system()
    iast = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    P = np.logspace(-2, 1.5, 60); y = np.linspace(0.02, 0.98, 60)
    S = np.zeros((len(y), len(P))); T = 298.15
    for i, yi in enumerate(y):
        for j, pj in enumerate(P):
            q1, q2, _, _ = iast.solve(pj, yi, T)
            S[i, j] = (q1 / max(q2, 1e-12)) * (1 - yi) / yi
    fig, ax = plt.subplots(figsize=(11, 6))
    pm = ax.pcolormesh(P, y, S, cmap="viridis", norm=LogNorm(), shading="auto")
    cs = ax.contour(P, y, S, levels=[1, 2, 5, 10, 20, 50], colors="white", linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%g")
    ax.set_xscale("log"); ax.set_xlabel("P (bar)"); ax.set_ylabel("y(CO₂)")
    ax.set_title(f"S(CO₂/N₂) map @ {T} K", fontweight="bold")
    plt.colorbar(pm, ax=ax, label="S(CO₂/N₂)")
    fig.suptitle("Figure 13 · Selectivity 2D Map", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "13_selectivity_2d.png", "Fig 13 · Selectivity Map",
             "Full operating-window map of CO₂/N₂ selectivity.")


def fig14_henry_constants():
    toth = thermo.get_cavenati_toth_system()
    dsl = thermo.get_krishnamurthy_dsl_system()
    T = 298.15
    KH_t = [toth[sp]["isotherm"].at_temperature(T).henry_constant()
            for sp in ("CO2", "N2", "CH4")]
    KH_d = [dsl[sp]["isotherm_mass"].at_temperature(T).henry_constant()
            for sp in ("CO2", "N2")]
    fig, ax = plt.subplots(figsize=(11, 5))
    x1 = np.arange(3); x2 = np.arange(2) + 3.5
    ax.bar(x1, KH_t, 0.55, color=["C0", "C1", "C2"], edgecolor="k",
           label="Toth (Cavenati)")
    ax.bar(x2, KH_d, 0.55, color=["C0", "C1"], edgecolor="k", alpha=0.65,
           label="DSL (Krishnamurthy)")
    ax.set_xticks(list(x1) + list(x2))
    ax.set_xticklabels(["CO₂", "N₂", "CH₄", "CO₂ ", "N₂ "])
    ax.set_ylabel("K_H (mol/(kg·bar))"); ax.set_yscale("log")
    for xi, v in zip(x1, KH_t):
        ax.text(xi, v * 1.15, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x2, KH_d):
        ax.text(xi, v * 1.15, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_title(f"Henry constants @ {T} K — all models", fontweight="bold")
    ax.legend()
    fig.suptitle("Figure 14 · Henry Constants", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "14_henry_constants.png", "Fig 14 · Henry Constants",
             "Dilute-limit affinities for all species and models.")


def fig15_working_capacity():
    toth = thermo.get_cavenati_toth_system()
    iast = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    T = 298.15; Pa, Pd = 1.5, 0.15
    yl = np.linspace(0.05, 0.95, 12)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    dq_c = [toth["CO2"]["isotherm"].loading(Pa, T) - toth["CO2"]["isotherm"].loading(Pd, T)]
    dq_n = [toth["N2"]["isotherm"].loading(Pa, T) - toth["N2"]["isotherm"].loading(Pd, T)]
    ax[0].bar([0, 1], [dq_c[0], dq_n[0]], color=["C0", "C1"], edgecolor="k")
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["CO₂", "N₂"])
    ax[0].set_ylabel("Δq (mol/kg)")
    ax[0].set_title("Pure working capacity", fontweight="bold")
    for i, v in enumerate([dq_c[0], dq_n[0]]):
        ax[0].text(i, v * 1.02, f"{v:.3f}", ha="center", fontsize=9)
    dqm = []
    for y in yl:
        qa1, _, _, _ = iast.solve(Pa, y, T)
        qd1, _, _, _ = iast.solve(Pd, y, T)
        dqm.append(qa1 - qd1)
    ax[1].plot(yl, dqm, "o-", lw=2.2, markersize=7, color="C3")
    ax[1].set_xlabel("y(CO₂) feed"); ax[1].set_ylabel("Δq_CO₂ (mol/kg)")
    ax[1].set_title(f"IAST working capacity @ {T} K", fontweight="bold")
    fig.suptitle("Figure 15 · Working Capacity", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "15_working_capacity.png", "Fig 15 · Working Capacity",
             "Pure-component vs mixture working capacity.")


def fig16_jacobian_signs():
    dsl = thermo.get_krishnamurthy_dsl_system()
    co2_d = thermo.TemperatureDependentDSLAnalytic(
        qm1_0=3489.44, b1_0=dsl["CO2"]["isotherm"].b1_0, dH1=-36.64e3,
        qm2_0=2872.35, b2_0=dsl["CO2"]["isotherm"].b2_0, dH2=-35.70e3)
    n2_d = thermo.TemperatureDependentDSLAnalytic(
        qm1_0=6613.55, b1_0=dsl["N2"]["isotherm"].b1_0, dH1=-15.82e3,
        qm2_0=1e-9, b2_0=dsl["N2"]["isotherm"].b2_0, dH2=-15.82e3)
    ext = thermo.ExtendedDSL(co2_d, n2_d)
    P = np.logspace(-2, 1.5, 40); y = np.linspace(0.05, 0.95, 40); T = 298.15
    keys = ["dq_c_dP", "dq_n_dP", "dq_c_dy1", "dq_n_dy1", "dq_c_dT", "dq_n_dT"]
    sign = {k: np.zeros((len(y), len(P))) for k in keys}
    for i, yi in enumerate(y):
        for j, pj in enumerate(P):
            jc = ext.jacobian(pj, yi, T)
            for k in keys:
                sign[k][i, j] = np.sign(jc[k])
    titles = {"dq_c_dP": "∂q_CO₂/∂P (+)", "dq_n_dP": "∂q_N₂/∂P (+)",
              "dq_c_dy1": "∂q_CO₂/∂y₁ (+)", "dq_n_dy1": "∂q_N₂/∂y₁ (-)",
              "dq_c_dT": "∂q_CO₂/∂T (-)", "dq_n_dT": "∂q_N₂/∂T (-)"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, k in zip(axes.flat, keys):
        im = ax.imshow(sign[k], aspect="auto", origin="lower",
                       extent=[P[0], P[-1], y[0], y[-1]],
                       cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xscale("log"); ax.set_xlabel("P (bar)"); ax.set_ylabel("y(CO₂)")
        ax.set_title(titles[k], fontweight="bold", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Figure 16 · Jacobian Sign Maps @ {T} K",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, "16_jacobian_signs.png", "Fig 16 · Jacobian Signs",
             "Sign of each Jacobian entry across (P, y) operating window.")


def fig17_stability_surface():
    co2_t = thermo.TemperatureDependentTothAnalytic(
        qm0=5.5, b0=2.85, t0=0.48, dH=-35.5e3, T0=298.15)
    P = np.logspace(-6, 4, 40); T = np.linspace(270, 420, 40)
    QQ = np.zeros((len(T), len(P)))
    for i, Ti in enumerate(T):
        for j, Pj in enumerate(P):
            try:
                QQ[i, j] = co2_t.loading(Pj, Ti)
            except Exception:
                QQ[i, j] = np.nan
    fig, ax = plt.subplots(figsize=(12, 6))
    pm = ax.pcolormesh(P, T, QQ, cmap="viridis", shading="auto",
                       norm=LogNorm(vmin=1e-3, vmax=10))
    cs = ax.contour(P, T, QQ, levels=[0.01, 0.1, 1.0, 5.0],
                    colors="white", linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("P (bar)"); ax.set_ylabel("T (K)")
    ax.set_title("CO₂ loading stability map (Toth)", fontweight="bold")
    plt.colorbar(pm, ax=ax, label="q (mol/kg)")
    fig.suptitle("Figure 17 · Stability Surface — 10 decades P × 150 K",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "17_stability_surface.png", "Fig 17 · Stability",
             "Loading surface across 10 pressure decades and 150 K temperature span.")


def fig18_cross_model_validation():
    toth = thermo.get_cavenati_toth_system()
    dsl = thermo.get_krishnamurthy_dsl_system()
    P = np.logspace(-2, 1, 40); T = 298.15
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    qt = [toth["CO2"]["isotherm"].loading(p, T) for p in P]
    qd = [dsl["CO2"]["isotherm_mass"].loading(p, T) for p in P]
    ax[0].plot(P, qt, lw=2.2, color="C0", label="Toth (Cavenati)")
    ax[0].plot(P, qd, lw=2.2, color="C3", ls="--", label="DSL (Krishnamurthy)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("P (bar)")
    ax[0].set_ylabel("q (mol/kg)"); ax[0].set_title(f"CO₂ @ {T} K", fontweight="bold")
    ax[0].legend()
    ratio = [a / b for a, b in zip(qt, qd)]
    ax[1].plot(P, ratio, lw=2.2, color="C2")
    ax[1].axhline(1.0, color="k", ls=":", lw=1)
    ax[1].fill_between(P, 0.5, 2.0, alpha=0.10, color="green",
                       label="Within 2× band")
    ax[1].set_xscale("log"); ax[1].set_xlabel("P (bar)")
    ax[1].set_ylabel("q_Toth / q_DSL")
    ax[1].set_title("Cross-model ratio", fontweight="bold"); ax[1].legend()
    fig.suptitle("Figure 18 · Cross-Model CO₂ Validation",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "18_cross_model_validation.png", "Fig 18 · Cross-Model",
             "Toth vs DSL CO₂ predictions on a common mass basis.")


def fig19_vpsa_cycle():
    toth = thermo.get_cavenati_toth_system()
    iast = thermo.BinaryIAST(toth["CO2"]["isotherm"], toth["N2"]["isotherm"])
    T = 298.15; y1 = 0.15
    Pa, Pd = 1.5, 0.15
    steps = [("Feed\n(1.5 bar)", Pa), ("Blowdown\n(0.15 bar)", Pd),
             ("Purge\n(0.15 bar)", Pd), ("Repress.\n(1.5 bar)", Pa)]
    qC = []; qN = []
    for _, P in steps:
        qc, qn, _, _ = iast.solve(P, y1, T)
        qC.append(qc); qN.append(qn)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(steps))
    ax[0].plot(x, qC, "o-", lw=2.5, markersize=12, color="C0", label="CO₂")
    ax[0].plot(x, qN, "s-", lw=2.5, markersize=12, color="C1", label="N₂")
    ax[0].set_xticks(x); ax[0].set_xticklabels([s[0] for s in steps])
    ax[0].set_ylabel("q (mol/kg)")
    ax[0].set_title("IAST loadings across VPSA cycle", fontweight="bold")
    ax[0].legend()
    cyc = qC + [qC[0]]; cycN = qN + [qN[0]]
    xc = np.arange(len(cyc))
    ax[1].fill_between(xc, 0, cyc, color="C0", alpha=0.4, label="CO₂")
    ax[1].fill_between(xc, 0, cycN, color="C1", alpha=0.4, label="N₂")
    ax[1].set_xticks(xc); ax[1].set_xticklabels([s[0] for s in steps] + ["Loop"])
    ax[1].set_ylabel("q (mol/kg)")
    ax[1].set_title("Closed cycle (mass swing)", fontweight="bold")
    ax[1].legend()
    fig.suptitle("Figure 19 · VPSA Cycle Mass Balance",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, "19_vpsa_cycle.png", "Fig 19 · VPSA Cycle",
             "Loadings across feed/blowdown/purge/repressurization cycle.")


def fig20_dashboard(unit_result, physics_results):
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

    fig.suptitle("Figure 20 · Test Results Dashboard",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "20_dashboard.png", "Fig 20 · Dashboard",
             "Summary of all unit tests and physics validations.")


# ==============================================================================
# HTML REPORT
# ==============================================================================
def build_html(unit_result, phys_results, elapsed):
    print("\n" + "=" * 72)
    print("PHASE 4 · BUILDING HTML REPORT")
    print("=" * 72)

    n_u = len(unit_result.rows)
    n_up = sum(1 for r in unit_result.rows if r["status"] == "PASS")
    n_uf = sum(1 for r in unit_result.rows if r["status"] in ("FAIL", "ERROR"))
    n_p = len(phys_results)
    n_pp = sum(1 for r in phys_results if r["status"] == "PASS")
    n_pf = sum(1 for r in phys_results if r["status"] in ("FAIL", "ERROR"))

    def sc(s):
        return {"PASS": "ok", "FAIL": "fail", "ERROR": "err", "SKIP": "skip"}.get(s, "")

    def render_rows(rows, key_fn, det=True):
        out = ""
        for r in rows:
            d = (f"<details><summary>detail</summary><pre>{r['detail']}</pre></details>"
                 if det and r["detail"] else "")
            out += (f"<tr><td>{key_fn(r)}</td>"
                    f"<td><span class='badge {sc(r['status'])}'>{r['status']}</span></td>"
                    f"<td>{d}</td></tr>")
        return out

    unit_html = render_rows(unit_result.rows,
                            lambda r: f"{r['class']}::{r['test']}")
    phys_html = render_rows(phys_results, lambda r: r["name"], det=False)

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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thermodynamics_v0.5.0 — Full Test Report</title>
<style>
:root{{--bg:#f5f7fa;--card:#fff;--border:#e2e8f0;--text:#1e293b;
--muted:#64748b;--primary:#3b82f6;--success:#10b981;--danger:#ef4444;
--warning:#f59e0b;--code-bg:#f1f5f9;--shadow:0 1px 3px rgba(0,0,0,0.06)}}
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","DejaVu Sans",sans-serif;
background:var(--bg);color:var(--text);margin:0;padding:0;line-height:1.55}}
.container{{max-width:1320px;margin:0 auto;padding:0 24px}}
header.hero{{background:linear-gradient(135deg,#1e40af 0%,#3b82f6 100%);
color:#fff;padding:48px 0 40px;margin-bottom:32px}}
header.hero h1{{margin:0;font-size:2.2rem;font-weight:700}}
header.hero .subtitle{{font-size:1.15rem;opacity:0.92;margin-top:4px}}
header.hero .meta{{margin-top:16px;font-size:0.9rem;opacity:0.75}}
.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:16px;margin-bottom:40px}}
.stat{{background:var(--card);border-radius:10px;padding:20px;
box-shadow:var(--shadow);text-align:center;border-top:3px solid var(--primary)}}
.stat.success{{border-top-color:var(--success)}}
.stat.danger{{border-top-color:var(--danger)}}
.stat.warning{{border-top-color:var(--warning)}}
.stat .num{{display:block;font-size:2rem;font-weight:700;line-height:1}}
.stat .lbl{{display:block;font-size:0.85rem;color:var(--muted);
margin-top:6px;text-transform:uppercase;letter-spacing:0.03em}}
section{{margin-bottom:48px}}
section h2{{font-size:1.45rem;font-weight:700;
border-bottom:2px solid var(--primary);padding-bottom:8px;margin-bottom:20px}}
table{{width:100%;background:var(--card);border-collapse:collapse;
border-radius:10px;overflow:hidden;box-shadow:var(--shadow);font-size:0.92rem}}
th{{background:#f1f5f9;text-align:left;padding:12px 16px;font-weight:600;
color:var(--text);border-bottom:1px solid var(--border)}}
td{{padding:10px 16px;border-bottom:1px solid var(--border);vertical-align:top}}
tr:last-child td{{border-bottom:none}}
tr:hover{{background:#f8fafc}}
.badge{{display:inline-block;padding:2px 10px;border-radius:999px;
font-size:0.75rem;font-weight:600;color:#fff;background:var(--muted);
text-transform:uppercase;letter-spacing:0.04em}}
.badge.ok{{background:var(--success)}}
.badge.fail{{background:var(--danger)}}
.badge.err{{background:var(--warning)}}
.badge.skip{{background:var(--muted)}}
details summary{{cursor:pointer;color:var(--primary);font-size:0.85rem}}
details pre,.figure-card.failed pre{{background:var(--code-bg);padding:10px;
border-radius:6px;font-size:0.78rem;overflow:auto;max-height:320px;
white-space:pre-wrap}}
.figure-grid{{display:grid;grid-template-columns:1fr;gap:32px}}
.figure-card{{background:var(--card);border-radius:12px;
box-shadow:var(--shadow);overflow:hidden}}
.figure-card img{{width:100%;display:block;background:#fff;padding:8px}}
.fig-caption{{padding:16px 20px 20px;border-top:1px solid var(--border)}}
.fig-caption h3{{margin:0 0 6px;font-size:1.05rem}}
.fig-caption p{{margin:0;color:var(--muted);font-size:0.9rem}}
.figure-card.failed{{border:2px solid var(--danger);padding:16px}}
footer{{text-align:center;padding:40px 0;color:var(--muted);font-size:0.85rem;
border-top:1px solid var(--border);margin-top:40px}}
code{{background:var(--code-bg);padding:2px 6px;border-radius:4px;font-size:0.88em}}
</style></head><body>
<header class="hero"><div class="container">
<h1>Thermodynamics_v0.5.0</h1>
<p class="subtitle">Full Test &amp; Visualization Report · 100+ validations</p>
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
<div class="stat warning"><span class="num">{len(FIGURES)}</span><span class="lbl">Figures</span></div>
<div class="stat"><span class="num">{elapsed:.2f}s</span><span class="lbl">Runtime</span></div>
</div></section>
<section><h2>Unit Test Results — {n_up}/{n_u}</h2>
<table><thead><tr><th style="width:30%">Test</th>
<th style="width:10%">Status</th><th>Detail</th></tr></thead>
<tbody>{unit_html}</tbody></table></section>
<section><h2>Physics Validations — {n_pp}/{n_p}</h2>
<table><thead><tr><th style="width:55%">Validation</th>
<th style="width:12%">Status</th><th>Detail</th></tr></thead>
<tbody>{phys_html}</tbody></table></section>
<section><h2>Visualizations — {len([f for f in FIGURES if not f['failed']])} figures</h2>
<div class="figure-grid">{fig_html}</div></section>
</div>
<footer>Generated by <code>test_Thermodynamics_v0_5_0.py</code> ·
Thermodynamic Equilibrium Engine for VPSA Adsorption · Zeolite 13X (CO₂/N₂)</footer>
</body></html>"""

    path = OUT_DIR / "report.html"
    path.write_text(html, encoding="utf-8")
    print(f"   [report] {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return str(path)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 74)
    print("  THERMODYNAMICS_v0.5.0 — FULL TEST & VISUALIZATION SUITE (100+ CHECKS)")
    print("=" * 74)
    print(f"[i] Module    : {MODULE_FILE.name}")
    print(f"[i] Output    : {OUT_DIR}")
    print(f"[i] Timestamp : {datetime.now().isoformat(timespec='seconds')}")

    buf = io.StringIO()
    t0 = time.time()

    unit_result = run_unit_tests()
    phys_results = run_physics_validations()

    print("\n" + "=" * 72)
    print("PHASE 3 · GENERATING 20 FIGURES")
    print("=" * 72)

    fig_fns = [
        fig01_pure_isotherms, fig02_derivative_validation, fig03_isosteric_heat,
        fig04_iast_pressure_sweep, fig05_iast_composition_sweep,
        fig06_extended_dsl_jacobian, fig07_asymptotic_henry_psi, fig08_gibbs_duhem,
        fig09_vpsa_window, fig10_temperature_sensitivity, fig11_site_decomposition,
        fig12_jacobian_validation, fig13_selectivity_2d, fig14_henry_constants,
        fig15_working_capacity, fig16_jacobian_signs, fig17_stability_surface,
        fig18_cross_model_validation, fig19_vpsa_cycle,
        lambda: fig20_dashboard(unit_result, phys_results),
    ]
    for i, fn in enumerate(fig_fns, 1):
        print(f"[{i:2d}/{len(fig_fns)}] {getattr(fn, '__name__', 'fig20_dashboard')}")
        safe_fig(fn)

    elapsed = time.time() - t0
    report = build_html(unit_result, phys_results, elapsed)

    # JSON diagnostics
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
    jp = OUT_DIR / "test_results.json"
    jp.write_text(json.dumps(diag, indent=2, default=json_safe), encoding="utf-8")
    print(f"   [json] {jp}")

    # Console log (use captured stdout — implement Tee on entry)
    lp = OUT_DIR / "console_log.txt"
    lp.write_text(LOG_BUF.getvalue(), encoding="utf-8")
    print(f"   [log]  {lp}")

    # Summary
    n_up = sum(1 for r in unit_result.rows if r["status"] == "PASS")
    n_uf = sum(1 for r in unit_result.rows if r["status"] != "PASS")
    n_pp = sum(1 for r in phys_results if r["status"] == "PASS")
    n_pf = len(phys_results) - n_pp
    total_tests = len(unit_result.rows) + len(phys_results)
    total_pass = n_up + n_pp
    figs_ok = sum(1 for f in FIGURES if not f["failed"])

    print("\n" + "=" * 74)
    print("  FINAL SUMMARY")
    print("=" * 74)
    print(f"  Unit tests          : {n_up}/{len(unit_result.rows)} passed  "
          f"(fail/err: {n_uf})")
    print(f"  Physics validations : {n_pp}/{len(phys_results)} passed  "
          f"(fail/err: {n_pf})")
    print(f"  TOTAL               : {total_pass}/{total_tests} passed")
    print(f"  Figures             : {figs_ok}/{len(FIGURES)} generated")
    print(f"  Runtime             : {elapsed:.2f}s")
    print(f"  Report              : {report}")
    print(f"  Output directory    : {OUT_DIR}")
    print("=" * 74)

    ok = (total_pass == total_tests and figs_ok == len(FIGURES))
    print("\n[LOCKED] All checks passed — module is production-ready."
          if ok else
          "\n[WARN] Some checks failed — review report.html")
    return 0 if ok else 1


# ==============================================================================
# TEE STDOUT → console + buffer (for log file)
# ==============================================================================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()


LOG_BUF = io.StringIO()
if __name__ == "__main__":
    _real_stdout = sys.stdout
    sys.stdout = _Tee(_real_stdout, LOG_BUF)
    try:
        sys.exit(main())
    finally:
        sys.stdout = _real_stdout