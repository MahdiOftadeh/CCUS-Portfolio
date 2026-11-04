"""Real, non-tautological tests for the IAST solver."""
import numpy as np
import pytest

from iast import (
    IASTSolver, Langmuir, DSL, Toth,
    LangmuirThermal, DSLThermal, TothThermal, PengRobinson,
    R_GAS,
)


# ---------------------------------------------------------------------------
# Pure isotherm tests
# ---------------------------------------------------------------------------

class TestPureIsotherms:
    def test_langmuir_henry_limit(self):
        iso = Langmuir(qm=5.5, b=2.85)
        P = 1e-10
        assert np.isclose(iso.loading(P) / P, iso.henry_constant(), rtol=1e-6)

    def test_langmuir_spreading_pressure_exact(self):
        iso = Langmuir(qm=5.5, b=2.85)
        for P in (0.1, 1.0, 10.0):
            expected = 5.5 * np.log1p(2.85 * P)
            assert np.isclose(iso.spreading_pressure(P), expected, rtol=1e-10)

    def test_toth_analytic_vs_numeric_derivative(self):
        iso = Toth(qm=5.5, b=2.85, t=0.48)
        for P in (0.05, 0.5, 1.0, 5.0):
            h = P * 1e-4
            num = (iso.loading(P + h) - iso.loading(P - h)) / (2 * h)
            ana = iso.dq_dP(P)
            assert np.isclose(num, ana, rtol=1e-5), f"P={P}"

    def test_spreading_pressure_monotonic(self):
        for iso in (Langmuir(5.5, 2.85), DSL(3, 1.5, 1.5, 0.2),
                    Toth(5.5, 2.85, 0.48)):
            P_grid = np.linspace(0.01, 20, 30)
            psi = np.array([iso.spreading_pressure(P) for P in P_grid])
            assert np.all(np.diff(psi) > 0.0)

    def test_inversion_roundtrip(self):
        for iso in (Langmuir(5.5, 2.85), DSL(3, 1.5, 1.5, 0.2),
                    Toth(5.5, 2.85, 0.48)):
            for P_target in (0.1, 1.0, 5.0):
                psi = iso.spreading_pressure(P_target)
                P_recov = iso.invert_spreading_pressure(psi)
                assert np.isclose(P_recov, P_target, rtol=1e-6)


# ---------------------------------------------------------------------------
# IAST limit tests
# ---------------------------------------------------------------------------

class TestIASTLimits:
    def test_pure_limit_recovers_isotherm(self):
        iso = DSLThermal(qm1=3.0, b1_0=1.5, Q1=30000,
                        qm2=1.5, b2_0=0.2, Q2=25000)
        solver = IASTSolver([iso, iso])
        P, T = 1.0, 298.15
        res = solver.solve(P, [1.0, 0.0], T)
        expected = iso.loading(P, T)
        assert np.isclose(res["q"][0], expected, rtol=1e-10)
        assert res["q"][1] == 0.0

    def test_identical_isotherms_give_x_equals_y(self):
        iso = TothThermal(qm=5.5, b0=2.85, t=0.48, Q=35500)
        solver = IASTSolver([iso, iso])
        res = solver.solve(1.0, [0.3, 0.7], 298.15)
        assert np.allclose(res["x"], [0.3, 0.7], atol=1e-10)
        assert np.isclose(res["q_total"], iso.loading(1.0, 298.15), rtol=1e-10)

    def test_henry_limit_selectivity(self):
        co2 = Langmuir(qm=5.5, b=2.85)
        n2 = Langmuir(qm=2.3, b=0.12)
        solver = IASTSolver([co2, n2])
        res = solver.solve(1e-8, [0.5, 0.5], 298.15)
        ratio_x = res["x"][0] / res["x"][1]
        ratio_K = (co2.henry_constant() / n2.henry_constant())
        assert np.isclose(ratio_x, ratio_K, rtol=1e-3)


# ---------------------------------------------------------------------------
# Thermodynamic consistency
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_phase_equilibrium_exact(self):
        co2 = DSLThermal(qm1=3.0, b1_0=1.5, Q1=30000,
                        qm2=1.5, b2_0=0.2, Q2=25000)
        n2 = DSLThermal(qm1=2.0, b1_0=0.1, Q1=18000,
                       qm2=0.001, b2_0=1e-9, Q2=18000)
        solver = IASTSolver([co2, n2])
        P, T = 1.0, 298.15
        y = np.array([0.15, 0.85])
        res = solver.solve(P, y, T)
        lhs = res["x"] * res["P0"]
        rhs = y * P
        assert np.allclose(lhs, rhs, rtol=1e-10)

    def test_gibbs_duhem_residual_small(self):
        co2 = TothThermal(qm=5.5, b0=2.85, t=0.48, Q=35500)
        n2 = TothThermal(qm=2.3, b0=0.12, t=0.95, Q=18000)
        solver = IASTSolver([co2, n2])
        for P in (0.5, 1.0, 5.0):
            for y_co2 in (0.1, 0.3, 0.5):
                res = solver.check_consistency(P, [y_co2, 1 - y_co2], 298.15)
                assert res["residual"] < 1e-3, f"P={P}, y={y_co2}"

    def test_qst_langmuir_equals_Q(self):
        iso = LangmuirThermal(qm=5.5, b0=2.85, Q=35500)
        for P in (0.1, 1.0, 5.0):
            qst = iso.isosteric_heat(P, 298.15, h=0.1)
            assert np.isclose(qst, 35500.0, rtol=1e-3), f"P={P}"

    def test_qst_exothermic_positive(self):
        co2 = TothThermal(qm=5.5, b0=2.85, t=0.48, Q=35500)
        for P in (0.1, 1.0, 5.0):
            assert co2.isosteric_heat(P, 298.15) > 0.0


# ---------------------------------------------------------------------------
# EOS tests
# ---------------------------------------------------------------------------

class TestEOS:
    def test_pr_pure_phi_low_pressure(self):
        eos = PengRobinson(
            Tc=[304.13], Pc=[7.3773e6], omega=[0.2236],
            names=["CO2"],
        )
        phi = eos.phi_pure(0, 1e-6, 298.15)
        assert np.isclose(phi, 1.0, atol=1e-4)

    def test_pr_mixture_matches_pure_at_limit(self):
        eos = PengRobinson(
            Tc=[304.13, 126.2], Pc=[7.3773e6, 3.3958e6],
            omega=[0.2236, 0.0372],
        )
        T, P_bar = 298.15, 1.0
        phi_mix = eos.phi_mixture(P_bar, [1.0, 0.0], T)
        phi_pure = eos.phi_pure(0, P_bar, T)
        assert np.isclose(phi_mix[0], phi_pure, rtol=1e-6)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_negative_pressure_rejected(self):
        solver = IASTSolver([Langmuir(5, 1), Langmuir(2, 1)])
        with pytest.raises(ValueError):
            solver.solve(-1.0, [0.5, 0.5], 298.15)

    def test_bad_composition_rejected(self):
        solver = IASTSolver([Langmuir(5, 1), Langmuir(2, 1)])
        with pytest.raises(ValueError):
            solver.solve(1.0, [0.5, 0.6], 298.15)

    def test_zero_pressure_returns_zeros(self):
        solver = IASTSolver([Langmuir(5, 1), Langmuir(2, 1)])
        res = solver.solve(0.0, [0.5, 0.5], 298.15)
        assert np.allclose(res["q"], 0.0)

    def test_trace_component_handled(self):
        co2 = TothThermal(qm=5.5, b0=2.85, t=0.48, Q=35500)
        n2 = TothThermal(qm=2.3, b0=0.12, t=0.95, Q=18000)
        solver = IASTSolver([co2, n2])
        res = solver.solve(1.0, [1e-8, 1.0 - 1e-8], 298.15)
        assert np.all(np.isfinite(res["q"]))
        assert res["q"].sum() > 0.0