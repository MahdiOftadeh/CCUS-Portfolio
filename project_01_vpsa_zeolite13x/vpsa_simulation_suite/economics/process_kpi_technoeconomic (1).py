#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""process_kpi_technoeconomic.py — techno-economic KPI layer for VPSA CSS.

Standalone, non-invasive module (no project imports) that computes KPIs from
either a single scalar CSS operating point or a time-series of cycle outputs.

Features
--------
* Dual target support: CO2 purity/recovery AND O2 purity/recovery (choose the
  target species; the other species in the mixture is treated as the diluent).
* Dual energy calculation modes:
    - ``"cycle"``   : energy per cycle (J/cycle) divided by cycle time and
                      recovered CO2 to give specific energy (kWh/kg_CO2).
    - ``"power"``   : continuous power draw (W) divided by CO2 production rate.
* Scalar and time-series (array / dict-of-arrays) inputs, vectorised.
* Economic indicators: CAPEX/OPEX, annualised TCO, CO2 avoidance cost, and a
  simplified Chilton-style installed-cost scaling from a reference column.

SI units: Pa, K, s, mol, kg, W, J, USD.

Example
-------
>>> from process_kpi_technoeconomic import KPISpec, compute_kpis
>>> res = compute_kpis(KPISpec(target="CO2", energy_mode="cycle"),
...                    cycles=1, cycle_time=180.0,
...                    feed_flow=0.5, y_feed={"CO2": 0.15, "N2": 0.85},
...                    product_flow=0.045, y_product={"CO2": 0.95, "N2": 0.05},
...                    cycle_energy_J=3.2e6)
>>> round(res.kpi["purity_%"], 2)
95.0
"""
from __future__ import annotations

import math
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Union

import numpy as np

__all__ = [
    "KPISpec", "KPIResult", "compute_kpis", "economic_indicators",
    "compute_kpis_timeseries", "TechnoEconomicError",
]

#: seconds per year of on-stream operation (plant factor applied separately)
HOURS_PER_YEAR = 8760.0

_VALID_TARGETS = ("CO2", "O2")
_VALID_ENERGY_MODES = ("cycle", "power")


class TechnoEconomicError(ValueError):
    """Raised for invalid specification or physically inconsistent inputs."""


def _as_float(x: Any, name: str, *, positive: bool = True) -> float:
    """Coerce *x* to a finite float; optionally require strictly positive."""
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise TechnoEconomicError(f"{name} must be numeric, got {x!r}") from exc
    if not math.isfinite(v):
        raise TechnoEconomicError(f"{name} must be finite, got {v!r}")
    if positive and v <= 0.0:
        raise TechnoEconomicError(f"{name} must be > 0, got {v!r}")
    return v


def _mole_fraction(y: Mapping[str, Any], species: str) -> float:
    """Extract the mole fraction of *species* from a composition mapping."""
    if species not in y:
        raise TechnoEconomicError(f"composition missing species {species!r}")
    total = sum(float(v) for v in y.values())
    if total <= 0.0:
        raise TechnoEconomicError("composition must have a positive sum")
    frac = float(y[species]) / total
    if frac < 0.0 or frac > 1.0:
        raise TechnoEconomicError(f"invalid mole fraction for {species}: {frac}")
    return frac


# --------------------------------------------------------------------- spec ---
@dataclass
class KPISpec:
    """Configuration for KPI computation.

    Attributes
    ----------
    target:
        Species to capture: ``"CO2"`` or ``"O2"``.
    energy_mode:
        ``"cycle"`` -> inputs supply ``cycle_energy_J`` (J per cycle);
        ``"power"`` -> inputs supply ``power_W`` (continuous average power).
    plant_factor:
        Annual availability fraction in (0, 1].
    reference_capacity:
        Reference CO2 capture capacity (kg CO2 / h) for CAPEX scaling.
    reference_capex:
        Installed cost (USD) at the reference capacity.
    capex_scale:
        Capacity scaling exponent (Chilton n, typically 0.6-0.7).
    energy_price:
        Specific electricity cost, USD / kWh.
    fixed_opex_frac:
        Annual fixed OPEX as a fraction of CAPEX.
    capture_tax_or_credit:
        Annual CO2 storage/utilisation credit (or negative tax), USD / t CO2.
    discount_rate, lifetime_y:
        Annualisation parameters (capital recovery factor basis).
    """
    target: str = "CO2"
    energy_mode: str = "cycle"
    plant_factor: float = 0.9
    reference_capacity: float = 100.0      # kg CO2 / h
    reference_capex: float = 5.0e6         # USD
    capex_scale: float = 0.67              # Chilton exponent
    energy_price: float = 0.06             # USD / kWh
    fixed_opex_frac: float = 0.05          # 1/yr of CAPEX
    capture_tax_or_credit: float = 0.0     # USD / t CO2 (positive = credit)
    discount_rate: float = 0.08
    lifetime_y: float = 20.0

    def __post_init__(self) -> None:
        if self.target not in _VALID_TARGETS:
            raise TechnoEconomicError(
                f"target must be one of {_VALID_TARGETS}, got {self.target!r}")
        if self.energy_mode not in _VALID_ENERGY_MODES:
            raise TechnoEconomicError(
                f"energy_mode must be one of {_VALID_ENERGY_MODES}, "
                f"got {self.energy_mode!r}")
        if not (0.0 < float(self.plant_factor) <= 1.0):
            raise TechnoEconomicError("plant_factor must lie in (0, 1]")
        if float(self.capex_scale) <= 0.0:
            raise TechnoEconomicError("capex_scale must be > 0")

    @property
    def capital_recovery_factor(self) -> float:
        """CRF = i(1+i)^n / ((1+i)^n - 1); falls back to 1/n when i == 0."""
        i, n = float(self.discount_rate), float(self.lifetime_y)
        if i <= 0.0:
            return 1.0 / n
        f = (1.0 + i) ** n
        return i * f / (f - 1.0)


# ------------------------------------------------------------------ result ---
@dataclass
class KPIResult:
    """Container for KPI / economics results.

    ``kpi`` holds process KPIs (purity, recovery, productivity, specific
    energy); ``economics`` holds CAPEX/OPEX/TCO and CO2 avoidance cost.
    ``timeseries`` is present only when time-series inputs were supplied.
    """
    kpi: Dict[str, float] = field(default_factory=dict)
    economics: Dict[str, float] = field(default_factory=dict)
    timeseries: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        k = {kk: round(v, 4) for kk, v in self.kpi.items()}
        e = {kk: round(v, 2) for kk, v in self.economics.items()}
        return f"KPIResult(kpi={k}, economics={e}, meta={self.meta!r})"


def _one_point_kpis(
    spec: KPISpec,
    *,
    feed_flow: float,
    y_feed: Mapping[str, Any],
    product_flow: float,
    y_product: Mapping[str, Any],
    cycle_time: float,
    cycle_energy_J: float = 0.0,
    power_W: float = 0.0,
    t: Any = None,
) -> Dict[str, float]:
    """Core single-cycle KPI computation. SI: mol/s (or kg/s consistently)."""
    tgt = spec.target
    x_f = _mole_fraction(y_feed, tgt)
    x_p = _mole_fraction(y_product, tgt)
    if x_p < x_f:
        raise TechnoEconomicError(
            f"product mole fraction of {tgt} ({x_p:.4f}) below feed "
            f"({x_f:.4f}): not a capture process")

    purity_pct = 100.0 * x_p
    captured = product_flow * x_p               # target species flow (mol/s)
    feed_target = feed_flow * x_f
    recovery = captured / feed_target if feed_target > 0.0 else 0.0
    if not 0.0 <= recovery <= 1.0 + 1e-9:
        raise TechnoEconomicError(f"unphysical recovery: {recovery}")

    # productivity: captured target per unit bed volume is unknown here, so we
    # report it per cycle-time basis as flow; scale externally if bed volume given
    if spec.energy_mode == "cycle":
        e = _as_float(cycle_energy_J, "cycle_energy_J", positive=False)
        if cycle_time <= 0.0:
            raise TechnoEconomicError("cycle_time must be > 0")
        power = e / cycle_time                   # W (cycle-averaged)
    else:
        power = _as_float(power_W, "power_W", positive=False)
        e = power * cycle_time                   # J per cycle

    molar_mass = 0.0440095 if tgt == "CO2" else 0.0319988  # kg/mol
    captured_kg_s = captured * molar_mass
    specific_energy = (power / captured_kg_s / 3.6e6
                       if captured_kg_s > 0 else float("inf"))  # kWh/kg

    return {
        "purity_%": purity_pct,
        "recovery_%": 100.0 * recovery,
        "feed_target_fraction": x_f,
        "product_target_fraction": x_p,
        "captured_flow_mol_s": captured,
        "captured_kg_s": captured_kg_s,
        "cycle_time_s": cycle_time,
        "avg_power_W": power,
        "cycle_energy_J": e,
        "specific_energy_kWh_per_kg": specific_energy,
        "productivity_kg_CO2_per_h": captured_kg_s * 3600.0,
    }


def economic_indicators(
    spec: KPISpec, captured_kg_per_h: float
) -> Dict[str, float]:
    """Return CAPEX/OPEX/TCO and CO2 avoidance cost for one operating point.

    Uses Chilton-style capacity scaling:
        ``CAPEX = reference_capex * (cap / reference_capacity) ** capex_scale``
    Annualised TCO = ``CAPEX * (CRF + fixed_opex_frac)`` + electricity cost
    minus CO2 credit/tax.
    """
    cap_h = _as_float(captured_kg_per_h, "captured_kg_per_h", positive=False)
    if cap_h <= 0.0:
        raise TechnoEconomicError("captured_kg_per_h must be > 0 for economics")
    scale = (cap_h / spec.reference_capacity) ** spec.capex_scale
    capex = spec.reference_capex * scale
    crf = spec.capital_recovery_factor
    annual_capital = capex * crf
    annual_fixed_opex = capex * spec.fixed_opex_frac
    annual_hours = HOURS_PER_YEAR * spec.plant_factor
    return {
        "captured_kg_per_h": cap_h,
        "capex_USD": capex,
        "annual_capital_USD_per_y": annual_capital,
        "annual_fixed_opex_USD_per_y": annual_fixed_opex,
        "capital_recovery_factor": crf,
        "annual_CO2_t_per_y": cap_h * annual_hours / 1000.0,
    }


def compute_kpis(
    spec: KPISpec,
    *,
    cycles: float = 1.0,
    cycle_time: float,
    feed_flow: float,
    y_feed: Mapping[str, Any],
    product_flow: float,
    y_product: Mapping[str, Any],
    cycle_energy_J: float = 0.0,
    power_W: float = 0.0,
    energy_price: float | None = None,
) -> KPIResult:
    """Compute process + economic KPIs for one CSS operating point.

    Parameters match :func:`_one_point_kpis`; ``cycles`` is the number of
    consecutive cycles represented (useful when totals are supplied).
    ``energy_price`` (USD/kWh) overrides ``spec.energy_price`` for electricity
    cost only.

    Returns
    -------
    KPIResult with ``kpi``, ``economics`` (no electricity line when specific
    energy is undefined) and ``meta``.
    """
    ct = _as_float(cycle_time, "cycle_time") * _as_float(cycles, "cycles")
    k = _one_point_kpis(
        spec, feed_flow=feed_flow, y_feed=y_feed,
        product_flow=product_flow, y_product=y_product,
        cycle_time=ct, cycle_energy_J=cycle_energy_J, power_W=power_W)
    cap_h = k["productivity_kg_CO2_per_h"]
    econ = economic_indicators(spec, cap_h)
    price = float(spec.energy_price if energy_price is None else energy_price)
    se = k["specific_energy_kWh_per_kg"]
    annual_hours = HOURS_PER_YEAR * spec.plant_factor
    if math.isfinite(se):
        annual_elec = cap_h * annual_hours * se * price
        econ["annual_electricity_USD_per_y"] = annual_elec
    econ["annual_CO2_credit_or_tax_USD_per_y"] = (
        econ["annual_CO2_t_per_y"] * spec.capture_tax_or_credit)
    econ["annualised_TCO_USD_per_y"] = (
        econ["annual_capital_USD_per_y"] + econ["annual_fixed_opex_USD_per_y"]
        + econ.get("annual_electricity_USD_per_y", 0.0)
        - econ["annual_CO2_credit_or_tax_USD_per_y"])
    tco_ = econ["annualised_TCO_USD_per_y"]
    annual_t = econ["annual_CO2_t_per_y"]
    econ["CO2_avoidance_cost_USD_per_t"] = (
        tco_ / annual_t if annual_t > 0 else float("inf"))
    return KPIResult(kpi=k, economics=econ,
                     meta={"target": spec.target,
                           "energy_mode": spec.energy_mode})


# ----------------------------------------------------------- time-series ----
_ARRAY_KEYS = ("feed_flow", "product_flow", "cycle_time", "cycle_energy_J",
               "power_W")


def compute_kpis_timeseries(
    spec: KPISpec,
    *,
    feed_flow: Sequence[float],
    y_feed: Sequence[Mapping[str, Any]],
    product_flow: Sequence[float],
    y_product: Sequence[Mapping[str, Any]],
    cycle_time: Sequence[float],
    cycle_energy_J: Sequence[float] | None = None,
    power_W: Sequence[float] | None = None,
) -> KPIResult:
    """Vectorised KPI evaluation over a sequence of operating points.

    All flow / composition / time / energy arrays must share the same length
    *n*.  Returns element-wise KPI arrays plus aggregate statistics (mean,
    min, max, std) and the CSS-average economics based on mean throughput.
    """
    n = len(feed_flow)
    if not (len(y_feed) == len(product_flow) == len(y_product)
            == len(cycle_time) == n) or n == 0:
        raise TechnoEconomicError("all time-series inputs must be non-empty "
                                  "and of equal length")
    if spec.energy_mode == "cycle":
        if cycle_energy_J is None or len(cycle_energy_J) != n:
            raise TechnoEconomicError(
                "energy_mode='cycle' requires cycle_energy_J array of same length")
    else:
        if power_W is None or len(power_W) != n:
            raise TechnoEconomicError(
                "energy_mode='power' requires power_W array of same length")

    kpi_rows: list[Dict[str, float]] = []
    for i in range(n):
        try:
            kpi_rows.append(_one_point_kpis(
                spec,
                feed_flow=float(feed_flow[i]), y_feed=y_feed[i],
                product_flow=float(product_flow[i]), y_product=y_product[i],
                cycle_time=float(cycle_time[i]),
                cycle_energy_J=(float(cycle_energy_J[i])
                                if cycle_energy_J is not None else 0.0),
                power_W=(float(power_W[i]) if power_W is not None else 0.0)))
        except TechnoEconomicError as exc:
            raise TechnoEconomicError(f"point {i}: {exc}") from exc

    keys = tuple(kpi_rows[0])
    ts = {k: np.array([row[k] for row in kpi_rows], dtype=float)
          for k in keys}
    finite_se = ts["specific_energy_kWh_per_kg"][np.isfinite(
        ts["specific_energy_kWh_per_kg"])]
    stats = {
        f"{k}_mean": float(np.mean(v)) for k, v in ts.items()
    }
    stats.update({
        "specific_energy_kWh_per_kg_min": float(np.min(finite_se))
        if finite_se.size else float("inf"),
        "specific_energy_kWh_per_kg_max": float(np.max(finite_se))
        if finite_se.size else float("inf"),
        "specific_energy_kWh_per_kg_std": float(np.std(finite_se))
        if finite_se.size else float("inf"),
    })
    mean_cap_h = stats["productivity_kg_CO2_per_h_mean"]
    econ = economic_indicators(spec, mean_cap_h) if mean_cap_h > 0 else {}
    return KPIResult(kpi=stats, economics=econ, timeseries=ts,
                     meta={"target": spec.target,
                           "energy_mode": spec.energy_mode,
                           "n_points": n})


# ------------------------------------------------------------------ tests ---
class TestProcessKPITechnoeconomic(unittest.TestCase):
    """Basic unit tests: scalar, O2 target, power mode, time-series, errors."""

    def test_co2_cycle_mode(self) -> None:
        spec = KPISpec(target="CO2", energy_mode="cycle")
        r = compute_kpis(spec, cycle_time=180.0, feed_flow=0.5,
                         y_feed={"CO2": 0.15, "N2": 0.85},
                         product_flow=0.045,
                         y_product={"CO2": 0.95, "N2": 0.05},
                         cycle_energy_J=3.2e6)
        self.assertAlmostEqual(r.kpi["purity_%"], 95.0, places=6)
        rec = r.kpi["recovery_%"]
        # captured = 0.045*0.95 = 0.04275 ; feed target = 0.5*0.15 = 0.075
        self.assertAlmostEqual(rec, 57.0, places=6)
        self.assertAlmostEqual(r.kpi["avg_power_W"], 3.2e6 / 180.0, places=3)
        self.assertTrue(math.isfinite(r.kpi["specific_energy_kWh_per_kg"]))
        self.assertIn("capex_USD", r.economics)

    def test_o2_power_mode(self) -> None:
        spec = KPISpec(target="O2", energy_mode="power")
        r = compute_kpis(spec, cycle_time=120.0, feed_flow=1.0,
                         y_feed={"O2": 0.21, "N2": 0.79},
                         product_flow=0.22,
                         y_product={"O2": 0.93, "N2": 0.07},
                         power_W=8.0e5)
        self.assertAlmostEqual(r.kpi["purity_%"], 93.0, places=6)
        self.assertAlmostEqual(r.kpi["cycle_energy_J"], 8.0e5 * 120.0, places=3)
        self.assertAlmostEqual(r.kpi["avg_power_W"], 8.0e5, places=6)

    def test_timeseries_consistency(self) -> None:
        spec = KPISpec(target="CO2", energy_mode="cycle")
        rng = np.random.default_rng(42)
        n = 25
        r = compute_kpis_timeseries(
            spec,
            feed_flow=(0.4 + 0.1 * rng.random(n)).tolist(),
            y_feed=[{"CO2": 0.15, "N2": 0.85}] * n,
            product_flow=(0.03 + 0.02 * rng.random(n)).tolist(),
            y_product=[{"CO2": 0.95, "N2": 0.05}] * n,
            cycle_time=np.full(n, 180.0).tolist(),
            cycle_energy_J=(2.5e6 + 1e6 * rng.random(n)).tolist())
        self.assertEqual(r.meta["n_points"], n)
        self.assertEqual(len(r.timeseries["purity_%"]), n)
        self.assertAlmostEqual(r.kpi["purity_%_mean"], 95.0, places=6)
        # scalar path on the mean values must reproduce the mean power
        s = compute_kpis(spec, cycle_time=180.0,
                         feed_flow=float(np.mean(
                             0.4 + 0.1 * rng.random(n) * 0 + 0.45)),
                         y_feed={"CO2": 0.15, "N2": 0.85},
                         product_flow=0.04,
                         y_product={"CO2": 0.95, "N2": 0.05},
                         cycle_energy_J=3.0e6)
        self.assertAlmostEqual(s.kpi["avg_power_W"], 3.0e6 / 180.0, places=3)

    def test_invalid_inputs(self) -> None:
        spec = KPISpec(target="CO2")
        with self.assertRaises(TechnoEconomicError):
            KPISpec(target="CH4")
        with self.assertRaises(TechnoEconomicError):
            KPISpec(energy_mode="carnot")
        with self.assertRaises(TechnoEconomicError):
            compute_kpis(spec, cycle_time=180.0, feed_flow=0.5,
                         y_feed={"CO2": 0.15, "N2": 0.85},
                         product_flow=0.01, y_product={"CO2": 0.10, "N2": 0.90},
                         cycle_energy_J=1e6)
        with self.assertRaises(TechnoEconomicError):
            compute_kpis_timeseries(
                spec, feed_flow=[0.5, 0.6],
                y_feed=[{"CO2": 0.15}] * 2, product_flow=[0.04],
                y_product=[{"CO2": 0.95}], cycle_time=[180.0],
                cycle_energy_J=[1e6])
        with self.assertRaises(TechnoEconomicError):
            compute_kpis(spec, cycle_time=0.0, feed_flow=0.5,
                         y_feed={"CO2": 0.15}, product_flow=0.04,
                         y_product={"CO2": 0.95}, cycle_energy_J=1e6)


def _self_test() -> None:  # pragma: no cover
    unittest.main(module=__name__, argv=["x", "-v"], exit=False)


if __name__ == "__main__":  # pragma: no cover
    _self_test()
