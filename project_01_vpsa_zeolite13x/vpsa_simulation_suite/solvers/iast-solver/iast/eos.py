"""Peng-Robinson EOS for fugacity coefficients.

Units:
    Input: P in bar, T in K.
    Internally converts to Pa.
"""
import numpy as np

R_GAS = 8.314462618  # J/(mol.K)


class PengRobinson:
    """Peng-Robinson EOS with van der Waals mixing rules."""

    def __init__(self, Tc, Pc, omega, names=None):
        self.Tc = np.asarray(Tc, dtype=float)
        self.Pc = np.asarray(Pc, dtype=float)  # Pa
        self.omega = np.asarray(omega, dtype=float)
        self.N = len(Tc)
        if not (len(Pc) == len(omega) == self.N):
            raise ValueError("Tc, Pc, omega must have the same length")
        self.names = list(names) if names else [f"c{i}" for i in range(self.N)]

        self.kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2
        self.a_c = 0.45724 * R_GAS ** 2 * self.Tc ** 2 / self.Pc
        self.b = 0.07780 * R_GAS * self.Tc / self.Pc
        self.kij = np.zeros((self.N, self.N))

    def set_binary_interaction(self, kij):
        kij = np.asarray(kij, dtype=float)
        if kij.shape != (self.N, self.N):
            raise ValueError(f"kij must be {self.N}x{self.N}")
        self.kij = kij

    def _a_pure(self, T):
        Tr = T / self.Tc
        alpha = (1.0 + self.kappa * (1.0 - np.sqrt(Tr))) ** 2
        return self.a_c * alpha

    def _a_mix(self, T, y):
        a_i = self._a_pure(T)
        a_ij = np.sqrt(np.outer(a_i, a_i)) * (1.0 - self.kij)
        return float(y @ a_ij @ y)

    def _b_mix(self, y):
        return float(y @ self.b)

    def _Z(self, A, B):
        coeffs = [1.0, -(1.0 - B), A - 3.0 * B ** 2 - 2.0 * B,
                  -(A * B - B ** 2 - B ** 3)]
        roots = np.roots(coeffs)
        real = roots[np.abs(roots.imag) < 1e-10].real
        if real.size == 0:
            return 1.0
        return float(np.max(real))

    def _ln_phi(self, Z, A, B, b_i, b_mix, a_i, a_mix, kij, y, i):
        a_ij = np.sqrt(np.outer(a_i, a_i)) * (1.0 - kij)
        sum_aij_yj = float(a_ij[i] @ y)
        term_A = A / (2.0 * np.sqrt(2.0) * B)
        term_log = np.log((Z + (1.0 + np.sqrt(2.0)) * B) /
                          (Z + (1.0 - np.sqrt(2.0)) * B))
        return (b_i / b_mix * (Z - 1.0)
                - np.log(Z - B)
                - term_A * (2.0 * sum_aij_yj / a_mix - b_i / b_mix) * term_log)

    def phi_pure(self, i, P_bar, T):
        """Fugacity coefficient of pure component i at P_bar, T."""
        P = P_bar * 1e5
        a = self._a_pure(T)[i]
        b = self.b[i]
        A = a * P / (R_GAS * T) ** 2
        B = b * P / (R_GAS * T)
        Z = self._Z(A, B)
        term_A = A / (2.0 * np.sqrt(2.0) * B)
        term_log = np.log((Z + (1.0 + np.sqrt(2.0)) * B) /
                          (Z + (1.0 - np.sqrt(2.0)) * B))
        ln_phi = (Z - 1.0) - np.log(Z - B) - term_A * term_log
        return float(np.exp(ln_phi))

    def phi_mixture(self, P_bar, y, T):
        """Fugacity coefficients of the mixture."""
        y = np.asarray(y, dtype=float)
        P = P_bar * 1e5
        a_mix = self._a_mix(T, y)
        b_mix = self._b_mix(y)
        A = a_mix * P / (R_GAS * T) ** 2
        B = b_mix * P / (R_GAS * T)
        Z = self._Z(A, B)
        a_i = self._a_pure(T)
        phi = np.zeros(self.N)
        for i in range(self.N):
            phi[i] = np.exp(self._ln_phi(
                Z, A, B, self.b[i], b_mix, a_i, a_mix, self.kij, y, i,
            ))
        return phi