"""N-component IAST solver with robust log-psi bracketing."""
import numpy as np
from scipy import optimize


class IASTInputError(ValueError):
    """Invalid input to the IAST solver."""


class IASTConvergenceError(RuntimeError):
    """Solver failed to bracket or converge."""


class IASTSolver:
    """N-component IAST equilibrium solver.

    Given total pressure P, gas-phase mole fractions y, and temperature T,
    computes:
        q_i  : adsorbed-phase loading of component i [mol/kg]
        x_i  : adsorbed-phase mole fraction
        psi  : common reduced spreading pressure [mol/kg]
    """

    def __init__(self, isotherms, eos=None, pressure_unit="bar"):
        self.isotherms = list(isotherms)
        self.N = len(self.isotherms)
        if self.N < 2:
            raise IASTInputError("IAST needs at least two isotherms")
        self.eos = eos
        self.pressure_unit = str(pressure_unit).lower()
        if self.pressure_unit not in ("bar", "pa", "atm", "mpa"):
            raise IASTInputError(f"Unknown pressure unit: {pressure_unit}")
        self._p0_cache = {}

    # ---- unit conversion ----------------------------------------------------
    def _to_bar(self, P):
        factor = {"bar": 1.0, "pa": 1e-5, "atm": 1.01325, "mpa": 10.0}
        return P * factor[self.pressure_unit]

    # ---- fugacity -----------------------------------------------------------
    def _phi_mix(self, P_bar, y, T):
        if self.eos is None:
            return np.ones(self.N)
        return self.eos.phi_mixture(P_bar, y, T)

    def _phi_pure(self, i, P_bar, T):
        if self.eos is None:
            return 1.0
        return self.eos.phi_pure(int(i), P_bar, T)

    # ---- P0 cache -----------------------------------------------------------
    def _P0(self, i, psi, T):
        key = (int(i), round(float(psi), 15), round(float(T), 8))
        if key in self._p0_cache:
            return self._p0_cache[key]
        iso = self.isotherms[int(i)]
        if hasattr(iso, "at_temperature"): iso = iso.at_temperature(T)
        val = iso.invert_spreading_pressure(psi)
        self._p0_cache[key] = val
        return val

    def clear_cache(self):
        self._p0_cache.clear()

    # ---- main solve ---------------------------------------------------------
    def solve(self, P, y, T):
        P_bar = self._to_bar(float(P))
        y = np.asarray(y, dtype=float)
        T = float(T)

        if P_bar < 0.0:
            raise IASTInputError("Pressure must be non-negative")
        if T <= 0.0:
            raise IASTInputError("Temperature must be positive")
        if y.shape != (self.N,):
            raise IASTInputError(f"y must have shape ({self.N},)")
        if np.any(y < 0.0):
            raise IASTInputError("y must be non-negative")
        if not np.isclose(y.sum(), 1.0, atol=1e-10):
            raise IASTInputError(f"y must sum to 1 (got {y.sum()})")

        if P_bar == 0.0:
            return self._result(np.zeros(self.N), y.copy(), 0.0,
                                P0=np.zeros(self.N),
                                phi_mix=np.ones(self.N),
                                phi_pure=np.ones(self.N))

        active = np.flatnonzero(y > 0.0)
        if active.size == 0:
            raise IASTInputError("At least one component must have y > 0")

        if active.size == 1:
            i = int(active[0])
            iso = self.isotherms[i].at_temperature(T)
            q = np.zeros(self.N)
            q[i] = iso.loading(P_bar)
            x = np.zeros(self.N)
            x[i] = 1.0
            psi = iso.spreading_pressure(P_bar)
            P0 = np.zeros(self.N)
            P0[i] = P_bar
            return self._result(q, x, psi, P0=P0,
                                phi_mix=np.ones(self.N),
                                phi_pure=np.ones(self.N))

        phi_mix = self._phi_mix(P_bar, y, T)
        f = y * P_bar * phi_mix

        def residual(log_psi):
            psi = np.exp(log_psi)
            s = 0.0
            for i in active:
                try:
                    P0_i = self._P0(int(i), psi, T)
                except (RuntimeError, ValueError):
                    return -1e10
                if P0_i <= 0.0:
                    return 1e6
                phi0_i = self._phi_pure(int(i), P0_i, T)
                s += f[i] / (P0_i * phi0_i)
            return np.log(max(s, 1e-300))

        lo = -700.0
        hi = np.log(max(P_bar, 1e-12)) + 10.0
        for _ in range(400):
            if residual(lo) > 0.0:
                break
            lo -= 5.0
            if lo < -745.0:
                raise IASTConvergenceError("Cannot bracket psi lower")
        for _ in range(400):
            if residual(hi) < 0.0:
                break
            hi += 5.0
            if hi > 700.0:
                raise IASTConvergenceError("Cannot bracket psi upper")

        log_psi = optimize.brentq(
            residual, lo, hi, xtol=1e-13, rtol=1e-13, maxiter=500,
        )
        psi = float(np.exp(log_psi))

        P0 = np.zeros(self.N)
        phi_pure = np.ones(self.N)
        x = np.zeros(self.N)
        for i in active:
            P0[i] = self._P0(int(i), psi, T)
            phi_pure[i] = self._phi_pure(int(i), P0[i], T)
            if P0[i] > 0.0:
                x[i] = f[i] / (P0[i] * phi_pure[i])

        if x.sum() <= 0.0:
            raise IASTConvergenceError("Failed to compute x (all zeros)")
        x /= x.sum()

        q0 = np.zeros(self.N)
        for i in active:
            iso = self.isotherms[int(i)]
            if hasattr(iso, "at_temperature"): iso = iso.at_temperature(T)
            q0[i] = iso.loading(P0[i])
        if np.any(q0[active] <= 0.0):
            raise IASTConvergenceError("Non-positive pure-component loading")

        q_mix = 1.0 / np.sum(x[active] / q0[active])
        q = x * q_mix

        return self._result(q, x, psi, P0=P0,
                            phi_mix=phi_mix, phi_pure=phi_pure)

    @staticmethod
    def _result(q, x, psi, P0=None, phi_mix=None, phi_pure=None):
        return {
            "q": q,
            "x": x,
            "psi": psi,
            "P0": P0,
            "phi_mix": phi_mix,
            "phi_pure": phi_pure,
            "q_total": float(q.sum()),
        }

    # ---- Jacobian -----------------------------------------------------------
    def jacobian(self, P, y, T, h_P=1e-4, h_y=1e-4, h_T=1e-2):
        """Numerical Jacobian of q with respect to P, y_i, T."""
        y = np.asarray(y, dtype=float)
        base = self.solve(P, y, T)["q"]

        dq_dP = (self.solve(P + h_P, y, T)["q"] - base) / h_P
        dq_dT = (self.solve(P, y, T + h_T)["q"] - base) / h_T

        dq_dy = np.zeros((self.N, self.N))
        for j in range(self.N - 1):
            yy = y.copy()
            yy[j] += h_y
            yy[-1] -= h_y
            yy = np.clip(yy, 0.0, 1.0)
            yy /= yy.sum()
            dq_dy[:, j] = (self.solve(P, yy, T)["q"] - base) / h_y

        return {"dq_dP": dq_dP, "dq_dy": dq_dy, "dq_dT": dq_dT}

    # ---- Gibbs-Duhem check --------------------------------------------------
    def check_consistency(self, P, y, T, tol=1e-3):
        """Check IAST thermodynamic consistency.

        Verifies that every component's spreading pressure evaluated at
        the solver's P0_i equals the equilibrium spreading pressure
        psi_target. This is the fundamental IAST equilibrium condition.
        """
        y = np.asarray(y, dtype=float)
        res = self.solve(P, y, T)
        psi_target = res["psi"]
        P0 = res["P0"]
        if psi_target <= 0.0:
            return {"residual": 0.0, "consistent": True}
        active = np.flatnonzero(y > 0)
        residuals = []
        for i in active:
            if P0[i] <= 0.0:
                continue
            iso = self.isotherms[int(i)]
            if hasattr(iso, "at_temperature"):
                iso = iso.at_temperature(T)
            psi_i = iso.spreading_pressure(P0[i])
            residuals.append(abs(psi_i - psi_target) / abs(psi_target))
        max_residual = max(residuals) if residuals else 0.0
        return {"residual": float(max_residual),
                "consistent": bool(max_residual < tol)}