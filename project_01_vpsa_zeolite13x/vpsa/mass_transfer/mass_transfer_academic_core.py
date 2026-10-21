"""
mass_transfer_academic_core.py
==============================

A research-grade, strictly academic, open-source core module for classical
adsorption mass-transfer analysis.

Design goals
------------
1. Research Software Engineering standards: frozen/slotted dataclasses, explicit
   ``numpy.typing.NDArray`` annotations, custom exception hierarchy, and strict
   validation (positivity, bounds, NaN/Inf guards) enforced in ``__post_init__``.
2. Classical analytical benchmarking: Crank's (1975) exact series solution for
   diffusion into a sphere with a constant surface boundary condition, compared
   against the Glueckauf (1955) Linear Driving Force (LDF) approximation with
   an explicit truncation-error analysis.
3. Dimensionless-group diagnostics: mass-transfer Biot number
   ``Bi_m = k_film * R_p / (D_pore / tau)`` to discriminate external-film
   control from intraparticle-diffusion control.
4. Transparent series-resistance breakdown (film / macropore / micropore) with
   a percentage report and an analytical diagnostic log.
5. Verifiable strict mass conservation in closed gas<->solid exchange steps,
   checked with explicit floating-point tolerances.

What this module deliberately does NOT contain (scope guarantee)
----------------------------------------------------------------
* No proprietary, calibrated, or tuned industrial kinetics or constants.
* No PSA/VPSA balance-of-plant, valve, compressor, or flowsheet models.
* No dynamic file discovery or imports; fully self-contained, NumPy-only.
* No non-linear PDE or multicomponent (Maxwell-Stefan) solvers. All results
  are closed-form analytical expressions from the standard textbook literature.

Units
-----
Strict SI: length [m], time [s], diffusivity [m^2 s^-1], film coefficient
[m s^-1], rate constants [s^-1], moles [mol]. The gas constant is retained for
dimensional reference only.

References (BibTeX)
-------------------
.. code-block:: bibtex

    @article{glueckauf1955,
      author  = {Glueckauf, E.},
      title   = {Theory of chromatography. Part 10. -- Formulae for diffusion
                 into spheres and their application to chromatography},
      journal = {Transactions of the Faraday Society},
      volume  = {51},
      pages   = {1540--1551},
      year    = {1955},
      doi     = {10.1039/TF9555101540}
    }

    @book{crank1975,
      author    = {Crank, J.},
      title     = {The Mathematics of Diffusion},
      edition   = {2},
      publisher = {Oxford University Press},
      address   = {Oxford},
      year      = {1975}
    }

    @book{ruthven1984,
      author    = {Ruthven, D. M.},
      title     = {Principles of Adsorption and Adsorption Processes},
      publisher = {John Wiley \& Sons},
      address   = {New York},
      year      = {1984}
    }
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DimensionMismatchError",
    "PhysicalInconsistencyError",
    "AcademicParticleParameters",
    "AcademicTransportParameters",
    "CrankSphereSolution",
    "glueckauf_ldf_rate",
    "fractional_uptake_ldf",
    "truncation_error",
    "compute_mass_transfer_biot",
    "classify_biot_regime",
    "SeriesResistanceBreakdown",
    "conservative_exchange",
    "run_verification_report",
]

# Universal gas constant [J mol^-1 K^-1] (dimensional reference only).
R_GAS: Final[float] = 8.314462618

# Floating-point tolerances used for conservation / consistency checks.
_CONS_ATOL: Final[float] = 1.0e-12
_CONS_RTOL: Final[float] = 1.0e-12


# ==========================================================================
# 1. Exception hierarchy
# ==========================================================================
class DimensionMismatchError(ValueError):
    """Raised when array shapes or physical dimensional arguments disagree."""


class PhysicalInconsistencyError(ValueError):
    """Raised when physically required orderings or bounds are violated."""


class _Validator:
    """Static validation helpers with strict NaN/Inf and bounds checking."""

    @staticmethod
    def positive(x: float, name: str) -> float:
        v = float(x)
        if not math.isfinite(v) or v <= 0.0:
            raise PhysicalInconsistencyError(
                f"{name} must be positive and finite (got {x!r})."
            )
        return v

    @staticmethod
    def non_negative(x: float, name: str) -> float:
        v = float(x)
        if not math.isfinite(v) or v < 0.0:
            raise PhysicalInconsistencyError(
                f"{name} must be non-negative and finite (got {x!r})."
            )
        return v

    @staticmethod
    def in_bounds(x: float, name: str, lo: float, hi: float) -> float:
        v = float(x)
        if not math.isfinite(v) or not (lo <= v <= hi):
            raise PhysicalInconsistencyError(
                f"{name} must lie in [{lo}, {hi}] (got {x!r})."
            )
        return v

    @staticmethod
    def array(x: Sequence[float] | NDArray[np.float64],
              n: int | None = None,
              name: str = "array") -> NDArray[np.float64]:
        a = np.asarray(x, dtype=np.float64)
        if a.ndim != 1:
            raise DimensionMismatchError(
                f"{name} must be a 1-D array (got ndim={a.ndim})."
            )
        if n is not None and a.shape != (n,):
            raise DimensionMismatchError(
                f"{name} must have shape ({n},); got {a.shape}."
            )
        if not np.all(np.isfinite(a)):
            raise PhysicalInconsistencyError(f"{name} contains NaN or Inf.")
        return a


# ==========================================================================
# 2. Parameter dataclasses (frozen + slotted; no tuned values)
# ==========================================================================
@dataclass(frozen=True, slots=True)
class AcademicParticleParameters:
    """
    Neutral geometric/transport parameter skeleton for a bidispersed pellet.

    Every default is a NEUTRAL PLACEHOLDER, not a calibrated or literature-tuned
    value. The user must supply and justify their own values from primary
    sources (validation enforces positivity and ordering, not realism).

    Attributes
    ----------
    particle_radius : float
        Pellet (macroparticle) radius ``R_p`` [m].
    crystal_radius : float
        Zeolite crystal (microparticle) radius ``r_c`` [m]; must satisfy
        ``r_c < R_p`` (bidispersed geometry).
    pore_diffusivity : float
        Effective macropore diffusivity ``D_p`` [m^2 s^-1].
    tortuosity : float
        Pellet tortuosity ``tau`` [dimensionless]; must be >= 1.
    porosity : float
        Pellet (intercrystalline void) porosity ``epsilon_p`` [dimensionless],
        bounded to (0, 1].
    """

    particle_radius: float = 1.0e-3
    crystal_radius: float = 1.0e-6
    pore_diffusivity: float = 1.0e-9
    tortuosity: float = 2.0
    porosity: float = 0.4

    def __post_init__(self) -> None:
        v = _Validator
        v.positive(self.particle_radius, "particle_radius")
        v.positive(self.crystal_radius, "crystal_radius")
        v.positive(self.pore_diffusivity, "pore_diffusivity")
        if v.positive(self.tortuosity, "tortuosity") < 1.0:
            raise PhysicalInconsistencyError("tortuosity must be >= 1.")
        v.in_bounds(self.porosity, "porosity", 0.0 + 1e-12, 1.0)
        if self.crystal_radius >= self.particle_radius:
            raise PhysicalInconsistencyError(
                "crystal_radius must be smaller than particle_radius "
                "(bidispersed pellet geometry assumed)."
            )

    @property
    def effective_pore_diffusivity(self) -> float:
        """Tortuosity-corrected pore diffusivity ``D_p / tau`` [m^2 s^-1]."""
        return self.pore_diffusivity / self.tortuosity


@dataclass(frozen=True, slots=True)
class AcademicTransportParameters:
    """
    Neutral transport-rate parameter skeleton [all rate constants in s^-1].

    The user supplies film / macropore / micropore first-order constants
    directly; no correlations, no temperature/pressure/loading corrections,
    and no calibration are performed (see :class:`SeriesResistanceBreakdown`).

    Attributes
    ----------
    film_coefficient : float
        External gas-film mass-transfer coefficient ``k_f`` [m s^-1].
    film_rate_constant : float
        Film LDF rate constant ``k_film`` [s^-1].
    macropore_rate_constant : float
        Macropore LDF rate constant ``k_macro`` [s^-1].
    micropore_rate_constant : float
        Micropore LDF rate constant ``k_micro`` [s^-1].
    """

    film_coefficient: float = 1.0e-2
    film_rate_constant: float = 1.0
    macropore_rate_constant: float = 1.0
    micropore_rate_constant: float = 1.0

    def __post_init__(self) -> None:
        v = _Validator
        v.positive(self.film_coefficient, "film_coefficient")
        v.positive(self.film_rate_constant, "film_rate_constant")
        v.positive(self.macropore_rate_constant, "macropore_rate_constant")
        v.positive(self.micropore_rate_constant, "micropore_rate_constant")


# ==========================================================================
# 3. Crank exact series vs Glueckauf LDF benchmark
# ==========================================================================
class CrankSphereSolution:
    """
    Crank's (1975, section 6.2) exact analytical series for uptake by a sphere
    of radius ``R`` with a constant surface concentration (constant ``D``):

        M_t / M_inf = 1 - (6 / pi^2) * sum_{n=1}^{inf} (1/n^2)
                      * exp(-n^2 * pi^2 * D * t / R^2)

    References
    ----------
    Crank, J. (1975) *The Mathematics of Diffusion*, 2nd ed., Oxford.
    """

    #: Hard upper bound on retained series terms (safety guard).
    MAX_TERMS: Final[int] = 100_000

    @staticmethod
    def fractional_uptake(t: Sequence[float] | NDArray[np.float64],
                          diffusivity: float,
                          radius: float,
                          n_terms: int = 200) -> NDArray[np.float64]:
        """
        Evaluate the truncated Crank series for an array of times.

        Parameters
        ----------
        t : array-like
            Times [s]; must be finite and non-negative.
        diffusivity : float
            Constant effective diffusivity ``D`` [m^2 s^-1].
        radius : float
            Sphere radius ``R`` [m].
        n_terms : int
            Number of retained series terms ``N``. The relative truncation
            error can be quantified with :meth:`truncation_error`.

        Returns
        -------
        NDArray[np.float64]
            Fractional uptake ``M_t / M_inf`` at each time (dimensionless).
        """
        t_arr = _Validator.array(t, name="t")
        if t_arr.size and np.any(t_arr < 0.0):
            raise PhysicalInconsistencyError("t must be non-negative.")
        D = _Validator.positive(diffusivity, "diffusivity")
        Rp = _Validator.positive(radius, "radius")
        if not (1 <= int(n_terms) <= CrankSphereSolution.MAX_TERMS):
            raise PhysicalInconsistencyError(
                f"n_terms must be in [1, {CrankSphereSolution.MAX_TERMS}]."
            )

        lam = np.pi ** 2 * D / Rp ** 2                      # [s^-1]
        n = np.arange(1, n_terms + 1, dtype=np.float64)      # (N,)
        # exp(-n^2 * lam * t): (N, T); compute log-space-safe via broadcasting.
        exponent = -np.outer(n ** 2, lam * t_arr)            # (N, T)
        series = np.sum(np.exp(exponent) / n[:, None] ** 2, axis=0)
        return 1.0 - (6.0 / np.pi ** 2) * series

    @staticmethod
    def recommended_terms(t: Sequence[float] | NDArray[np.float64],
                          diffusivity: float,
                          radius: float,
                          tolerance: float = 1.0e-10) -> int:
        """
        Smallest term count whose largest late-time residual is below tolerance.

        For long times the series is dominated by the n=1 term; the criterion is
        that the first omitted term at the largest time is below ``tolerance``.
        """
        t_arr = _Validator.array(t, name="t")
        lam = np.pi ** 2 * _Validator.positive(diffusivity, "diffusivity") \
            / _Validator.positive(radius, "radius") ** 2
        tol = _Validator.in_bounds(tolerance, "tolerance", 1e-16, 1e-2)
        t_max = float(t_arr.max()) if t_arr.size else 0.0
        n = 1
        while n < CrankSphereSolution.MAX_TERMS:
            omitted = math.exp(-(n + 1) ** 2 * lam * t_max) / (n + 1) ** 2
            if omitted * (6.0 / np.pi ** 2) < tol:
                return n + 1
            n += 1
        return CrankSphereSolution.MAX_TERMS


def glueckauf_ldf_rate(diffusivity: float, radius: float,
                       shape_factor: float = 15.0) -> float:
    """
    Classic Glueckauf (1955) constant-diffusivity spherical LDF constant:

        k_LDF = shape_factor * D / R^2     [s^-1]

    The shape factor 15 applies to spheres; the derivation assumes a linear
    isotherm, constant D, and small fractional uptake corrections of
    third order in time (Glueckauf 1955; Ruthven 1984, Ch. 4).
    """
    sf = _Validator.positive(shape_factor, "shape_factor")
    D = _Validator.positive(diffusivity, "diffusivity")
    Rp = _Validator.positive(radius, "radius")
    return sf * D / Rp ** 2


def fractional_uptake_ldf(t: Sequence[float] | NDArray[np.float64],
                          k_ldf: float) -> NDArray[np.float64]:
    """
    Analytical Glueckauf LDF uptake curve: ``F(t) = 1 - exp(-k_LDF * t)``.

    This is the first-order response of the LDF rate law
    ``dq/dt = k_LDF * (q_sat - q)`` with constant ``q_sat``.
    """
    t_arr = _Validator.array(t, name="t")
    if t_arr.size and np.any(t_arr < 0.0):
        raise PhysicalInconsistencyError("t must be non-negative.")
    k = _Validator.positive(k_ldf, "k_ldf")
    return 1.0 - np.exp(-k * t_arr)


def truncation_error(t: Sequence[float] | NDArray[np.float64],
                     diffusivity: float,
                     radius: float,
                     n_terms: int = 200) -> dict[str, float]:
    """
    Compare the Glueckauf LDF approximation (``k = 15 D / R^2``) against the
    truncated Crank exact series and report relative errors.

    Returns a dictionary with the LDF rate constant, maximum absolute relative
    deviation ``max |F_LDF - F_Crank|``, the RMS deviation, and the maximum
    relative-to-Crank error over the supplied time grid.
    """
    t_arr = _Validator.array(t, name="t")
    k = glueckauf_ldf_rate(diffusivity, radius)
    f_crank = CrankSphereSolution.fractional_uptake(t_arr, diffusivity, radius,
                                                    n_terms=n_terms)
    f_ldf = fractional_uptake_ldf(t_arr, k)
    abs_dev = np.abs(f_ldf - f_crank)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_dev = np.where(f_crank > 1e-12, abs_dev / f_crank, 0.0)
    return {
        "k_ldf_s^-1": k,
        "max_abs_deviation": float(abs_dev.max()) if abs_dev.size else 0.0,
        "rms_deviation": float(np.sqrt(np.mean(abs_dev ** 2))) if abs_dev.size else 0.0,
        "max_relative_error": float(rel_dev.max()) if rel_dev.size else 0.0,
        "final_crank": float(f_crank[-1]) if f_crank.size else 0.0,
        "final_ldf": float(f_ldf[-1]) if f_ldf.size else 0.0,
    }


# ==========================================================================
# 4. Dimensionless group analysis: mass-transfer Biot number
# ==========================================================================
def compute_mass_transfer_biot(film_coefficient: float,
                               particle_radius: float,
                               pore_diffusivity: float,
                               tortuosity: float = 1.0) -> float:
    """
    Mass-transfer Biot number for a porous adsorbent pellet:

        Bi_m = k_film * R_p / (D_pore / tau)

    ``D_pore / tau`` is the tortuosity-corrected intraparticle diffusivity.
    ``Bi_m`` compares the external film resistance with the internal
    diffusion resistance (Ruthven 1984, Ch. 4; Crank 1975, Ch. 6).
    """
    kf = _Validator.positive(film_coefficient, "film_coefficient")
    Rp = _Validator.positive(particle_radius, "particle_radius")
    D = _Validator.positive(pore_diffusivity, "pore_diffusivity")
    tau = _Validator.positive(tortuosity, "tortuosity")
    if tau < 1.0:
        raise PhysicalInconsistencyError("tortuosity must be >= 1.")
    return kf * Rp / (D / tau)


def classify_biot_regime(bi: float) -> str:
    """
    Classify the rate-controlling regime from the mass-transfer Biot number.

    Textbook ordering-of-magnitude thresholds (deliberately coarse and
    dimensionless, not tuned to any system):

    * ``Bi_m << 1`` (here < 0.1): external film diffusion control.
    * ``0.1 <= Bi_m <= 10``: mixed (comparable film and internal resistances).
    * ``Bi_m >> 10``: intraparticle (pore/crystal) diffusion control.
    """
    b = _Validator.positive(bi, "bi")
    if b < 0.1:
        return ("external film diffusion control "
                "(film resistance dominates: Bi_m << 1)")
    if b <= 10.0:
        return ("mixed external/internal control "
                "(comparable film and intraparticle resistances)")
    return ("intraparticle diffusion control "
            "(internal resistance dominates: Bi_m >> 1)")


# ==========================================================================
# 5. Series resistance breakdown with percentage report
# ==========================================================================
class SeriesResistanceBreakdown:
    """
    Transparent three-resistance-in-series overall LDF coefficient.

        k_overall = 1 / (1/k_film + 1/k_macro + 1/k_micro)

    All rate constants are user-supplied [s^-1]; the class computes the
    resistance shares, the overall constant, and an analytical diagnostic log
    naming the dominant resistance. No correlations or calibration are used
    (Ruthven 1984, Ch. 4; Glueckauf 1955).
    """

    def __init__(self, transport: AcademicTransportParameters) -> None:
        if not isinstance(transport, AcademicTransportParameters):
            raise DimensionMismatchError(
                "transport must be an AcademicTransportParameters instance."
            )
        self._p = transport

    @property
    def resistances(self) -> dict[str, float]:
        """Per-unit-loading resistances [s] in series."""
        p = self._p
        return {
            "film": 1.0 / p.film_rate_constant,
            "macropore": 1.0 / p.macropore_rate_constant,
            "micropore": 1.0 / p.micropore_rate_constant,
        }

    @property
    def k_overall(self) -> float:
        """Overall LDF rate constant [s^-1]."""
        total = sum(self.resistances.values())
        if not math.isfinite(total) or total <= 0.0:
            raise ArithmeticError("total series resistance must be positive.")
        return 1.0 / total

    def resistance_percentages(self) -> dict[str, float]:
        """Percentage contribution of each resistance to the total."""
        res = self.resistances
        total = sum(res.values())
        return {name: 100.0 * value / total for name, value in res.items()}

    def diagnostic_log(self) -> str:
        """Analytical diagnostic report naming the controlling resistance."""
        pct = self.resistance_percentages()
        dominant = max(pct, key=pct.__getitem__)          # type: ignore[arg-type]
        lines = [
            "Series resistance diagnostic (analytical, no correlations):",
            *[f"  {name:<10s}: {share:6.2f} %" for name, share in pct.items()],
            f"  k_overall  : {self.k_overall:.6e} s^-1",
            f"  controlling step: {dominant} diffusion "
            f"({pct[dominant]:.2f} % of total resistance)",
        ]
        if pct[dominant] < 40.0:
            lines.append("  note: no single step clearly dominates; "
                         "a genuinely mixed-control regime is indicated.")
        return "\n".join(lines)

    def rate(self, q: float, q_sat: float) -> float:
        """LDF rate ``dq/dt = k_overall * (q_sat - q)`` [loading units / s]."""
        q_v = _Validator.non_negative(q, "q")
        q_sat_v = _Validator.non_negative(q_sat, "q_sat")
        return self.k_overall * (q_sat_v - q_v)


# ==========================================================================
# 6. Strict mass conservation in closed exchange steps
# ==========================================================================
def conservative_exchange(gas_moles: Sequence[float] | NDArray[np.float64],
                          solid_moles: Sequence[float] | NDArray[np.float64],
                          uptake: Sequence[float] | NDArray[np.float64],
                          dt: float) -> tuple[NDArray[np.float64],
                                              NDArray[np.float64]]:
    """
    Explicit, conservatively time-limited gas<->solid mole exchange.

    For each cell, the attempted transfer is ``delta = uptake * dt`` (positive
    = adsorption, gas -> solid). The transfer is clipped to the donor
    inventory so that neither reservoir can go negative. Total moles
    ``gas + solid`` are exactly conserved, verified with explicit floating
    point tolerances (``atol = rtol = 1e-12``).

    Raises
    ------
    DimensionMismatchError
        If input arrays have inconsistent shapes.
    PhysicalInconsistencyError
        On negative inventories, NaN/Inf inputs, or invalid ``dt``.
    ArithmeticError
        If conservation or non-negativity fails after the exchange.
    """
    g = _Validator.array(gas_moles, name="gas_moles").copy()
    s = _Validator.array(solid_moles, g.shape[0], "solid_moles").copy()
    u = _Validator.array(uptake, g.shape[0], "uptake")
    dt_v = _Validator.positive(dt, "dt")
    if np.any(g < 0.0) or np.any(s < 0.0):
        raise PhysicalInconsistencyError("inventories must be non-negative.")

    total_before = g + s
    delta = u * dt_v
    delta = np.where(delta > 0.0, np.minimum(delta, s),
                     np.maximum(delta, -g))
    g2 = g - delta
    s2 = s + delta

    if np.any(g2 < -_CONS_ATOL) or np.any(s2 < -_CONS_ATOL):
        raise ArithmeticError("negative inventory after exchange.")
    if not np.allclose(total_before, g2 + s2, rtol=_CONS_RTOL,
                       atol=_CONS_ATOL):
        raise ArithmeticError("strict mass conservation failed.")
    return g2, s2


# ==========================================================================
# 7. Self-contained verification report
# ==========================================================================
def run_verification_report() -> None:
    """Run all self-tests and print a clean tabular verification report."""

    def banner(title: str) -> None:
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)

    banner("MASS TRANSFER ACADEMIC CORE -- VERIFICATION REPORT (academic only)")

    # --- Test 1: mass conservation ----------------------------------------
    g0 = np.array([1.0, 0.5, 0.0])
    s0 = np.array([2.0, 0.2, 0.3])
    rates = np.array([5.0, 2.0, -1.0])
    g1, s1 = conservative_exchange(g0, s0, rates, dt=0.1)
    cons_ok = bool(np.allclose(g0 + s0, g1 + s1, rtol=_CONS_RTOL,
                               atol=_CONS_ATOL))
    nonneg_ok = bool(np.all(g1 >= 0.0) and np.all(s1 >= 0.0))
    exact_ok = bool(abs(g1[0] - 0.5) < 1e-12 and abs(s1[0] - 2.5) < 1e-12)
    assert cons_ok and nonneg_ok and exact_ok, "conservation tests failed"
    print("[PASS] conservative_exchange: strict mass conservation, "
          "non-negativity, exact clipped transfer.")

    # --- Test 2: Glueckauf LDF constant ------------------------------------
    k = glueckauf_ldf_rate(diffusivity=1.0e-9, radius=1.0e-6)
    assert abs(k - 15.0e3) <= 1e-9 * k, "Glueckauf k mismatch"
    print(f"[PASS] glueckauf_ldf_rate: k = {k:.6e} s^-1 (expected 1.5e+04).")

    # --- Test 3: Crank series vs LDF benchmark ------------------------------
    D, Rp = 1.0e-10, 1.0e-3
    t = np.linspace(0.0, 6.0 * Rp ** 2 / D, 400)   # 0 .. 6 characteristic times
    n_rec = CrankSphereSolution.recommended_terms(t, D, Rp, tolerance=1e-12)
    f_crank = CrankSphereSolution.fractional_uptake(t, D, Rp, n_terms=n_rec)
    err = truncation_error(t, D, Rp, n_terms=n_rec)
    assert np.all(f_crank >= 0.0) and np.all(f_crank <= 1.0 + 1e-12)
    assert abs(f_crank[-1] - 1.0) < 1e-9, "Crank series did not converge to 1"
    # The LDF cannot reproduce the parabolic short-time behaviour F -> 0 with
    # zero slope (its derivative at t=0 is k_LDF), so the largest deviation
    # occurs at t=0 and equals exactly 6/pi^2. This is asserted analytically.
    assert abs(err["max_abs_deviation"] - 6.0 / np.pi ** 2) < 5e-2, (
        "LDF vs Crank deviation outside the expected analytical band")
    interior = slice(1, None)
    max_dev_interior = float(np.abs(
        fractional_uptake_ldf(t, err["k_ldf_s^-1"])[interior]
        - f_crank[interior]).max())
    assert max_dev_interior < 0.10, "unexpected LDF deviation at t > 0"
    assert 0.0 < err["rms_deviation"] < 0.05
    print(f"[PASS] Crank series: N={n_rec} terms, F(t_end)={f_crank[-1]:.12f}")
    print(f"       LDF vs Crank: max|dev|={err['max_abs_deviation']:.4e} "
          f"(at t=0, theoretical 6/pi^2={6/np.pi**2:.4e}), "
          f"max|dev|(t>0)={max_dev_interior:.4e}, "
          f"rms={err['rms_deviation']:.4e}")

    # --- Test 4: Biot number regimes ----------------------------------------
    bi_film_ctl = compute_mass_transfer_biot(1e-5, 1e-3, 1e-9, tau=2.0)  # << 1
    bi_int_ctl = compute_mass_transfer_biot(1e-1, 1e-3, 1e-12, tau=2.0)  # >> 10
    assert classify_biot_regime(bi_film_ctl).startswith("external film")
    assert classify_biot_regime(bi_int_ctl).startswith("intraparticle")
    print(f"[PASS] Biot diagnostics: Bi_m(film-limited)={bi_film_ctl:.3e} -> "
          f"film control; Bi_m(pore-limited)={bi_int_ctl:.3e} -> "
          f"intraparticle control.")

    # --- Test 5: series resistance breakdown --------------------------------
    params = AcademicTransportParameters(
        film_coefficient=1.0e-2,
        film_rate_constant=0.10,
        macropore_rate_constant=0.05,
        micropore_rate_constant=0.01,
    )
    brk = SeriesResistanceBreakdown(params)
    k_expected = 1.0 / (10.0 + 20.0 + 100.0)
    assert abs(brk.k_overall - k_expected) < 1e-15
    pct = brk.resistance_percentages()
    assert abs(sum(pct.values()) - 100.0) < 1e-9
    print("[PASS] SeriesResistanceBreakdown: k_overall = "
          f"{brk.k_overall:.6e} s^-1; shares = "
          + ", ".join(f"{k_}={v:.2f}%" for k_, v in pct.items()))
    print(brk.diagnostic_log())

    # --- Test 6: validation & exception guards -------------------------------
    for bad_call in (
        lambda: glueckauf_ldf_rate(-1e-9, 1e-6),
        lambda: compute_mass_transfer_biot(float("nan"), 1e-3, 1e-9),
        lambda: AcademicParticleParameters(crystal_radius=2e-3),
        lambda: AcademicParticleParameters(tortuosity=0.5),
        lambda: CrankSphereSolution.fractional_uptake(
            np.array([1.0]), 1e-10, 1e-3, n_terms=0),
    ):
        try:
            bad_call()
        except (PhysicalInconsistencyError, DimensionMismatchError):
            pass
        else:  # pragma: no cover
            raise AssertionError("validation guard did not fire")
    print("[PASS] Validation guards: NaN/Inf, positivity, bounds, ordering "
          "and shape checks all raise PhysicalInconsistencyError / "
          "DimensionMismatchError.")

    # --- Summary table --------------------------------------------------------
    banner("SUMMARY")
    rows = [
        ("conservative_exchange conservation", cons_ok),
        ("conservative_exchange non-negativity", nonneg_ok),
        ("conservative_exchange exact clipping", exact_ok),
        ("glueckauf_ldf_rate analytic value", abs(k - 15e3) <= 1e-9 * k),
        ("Crank series convergence to 1", abs(f_crank[-1] - 1.0) < 1e-9),
        ("LDF vs Crank deviation matches 6/pi^2 bound",
         abs(err["max_abs_deviation"] - 6.0 / np.pi ** 2) < 5e-2),
        ("Biot regime classification", bi_film_ctl < 0.1 < bi_int_ctl),
        ("series k_overall analytic match",
         abs(brk.k_overall - k_expected) < 1e-15),
        ("resistance percentages sum to 100", abs(sum(pct.values()) - 100) < 1e-9),
    ]
    w = max(len(r[0]) for r in rows)
    for name, ok in rows:
        print(f"  {name:<{w}s} : {'PASS' if ok else 'FAIL'}")
    print("\nAll self-tests passed. Strictly academic scope: analytical, "
          "single-component, isothermal, constant-D, no tuned constants.")


if __name__ == "__main__":
    run_verification_report()
