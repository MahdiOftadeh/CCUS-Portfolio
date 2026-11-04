"""CO2/N2 separation on Zeolite 13X - IAST demo.

Parameters from Cavenati, Grande & Rodrigues (2004),
Ind. Eng. Chem. Res. 43, 5399-5406, Table 5.
Toth isotherm, q in mol/kg, P in bar, T in K.
"""
import numpy as np
from iast import IASTSolver, TothThermal


CO2 = TothThermal(qm=5.50, b0=2.85, t=0.480, Q=35500.0)
N2  = TothThermal(qm=2.30, b0=0.12, t=0.950, Q=18000.0)


def main():
    solver = IASTSolver([CO2, N2])
    T = 298.15

    print(f"{'P(bar)':>8} {'y_CO2':>6} {'q_CO2':>8} {'q_N2':>8}"
          f" {'x_CO2':>8} {'psi':>8}")
    print("-" * 60)

    for P in (0.1, 1.0, 5.0, 10.0):
        for y_co2 in (0.05, 0.15, 0.50):
            res = solver.solve(P, [y_co2, 1.0 - y_co2], T)
            print(f"{P:>8.2f} {y_co2:>6.2f}"
                  f" {res['q'][0]:>8.4f} {res['q'][1]:>8.4f}"
                  f" {res['x'][0]:>8.4f} {res['psi']:>8.4f}")

    print("\nGibbs-Duhem residual (should be < 1e-3):")
    for P in (0.1, 1.0, 5.0):
        r = solver.check_consistency(P, [0.15, 0.85], T)
        print(f"  P={P:>4.1f} bar: residual = {r['residual']:.2e}"
              f" -> {'OK' if r['consistent'] else 'FAIL'}")


if __name__ == "__main__":
    main()