
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import json
import os

R = 8.314462618; T = 298.15; L = 1.0; D = 0.05; A = np.pi * D**2 / 4; eps = 0.37; rho = 1130.
Nz = 10; dz = L / Nz; z = (np.arange(Nz) + 0.5) * dz; Mbed = rho * (1 - eps) * A * L

# 13X dual-site Langmuir parameters
qm = np.array([[2.425, 1.05], [2.425, 1.05]])
b = np.array([[3.25e-5, 0.12e-5], [3.25e-5, 0.12e-5]])
k = np.array([0.18, 0.25])
Plo, Pint, Peq, Phi = 0.05e5, 0.4e5, 0.95e5, 1.5e5

def qstar(y, P):
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        if y.shape[0] != 2: raise ValueError("1D array must have length 2")
        y = y[:, None]
    elif y.ndim == 2 and y.shape[0] != 2 and y.shape[1] == 2:
        y = y.T
    p = np.maximum(y, 0) * P
    q = np.zeros_like(p)
    for s in range(2):
        den = 1.0 + np.sum(b[s, :, None] * p, axis=0)
        q += qm[s, :, None] * b[s, :, None] * p / den
    return q

def run_stage_sim(yfeed, feed_u, times_dict, rinse_u=0.04, label="Stage", cycles=5, adaptive_rinse=False, rinse_stop_thresh=0.05):
    # times_dict has: press, ads, rinse, pe, blow, evac
    CT = sum(times_dict.values())
    
    def one_cycle(state, keep_profiles=False):
        total = np.zeros(8)
        prof = {}
        
        # Adaptive rinse: if requested, we can adjust rinse time or velocity based on the front
        rinse_t = times_dict['rinse']
        current_rinse_u = rinse_u
        
        if adaptive_rinse:
            # Check CO2 mole fraction at the last 10% (last cell, z index -1)
            y_curr = state[:2*Nz].reshape(2, Nz)
            y_CO2_top = y_curr[0, -1] / max(np.sum(y_curr[:, -1]), 1e-12)
            # If front has already reached top, decrease rinse to avoid pushing CO2 out
            # If top is very clean (y_CO2_top < thresh), can rinse moderately
            if y_CO2_top > rinse_stop_thresh:
                rinse_t = max(5.0, times_dict['rinse'] * 0.3)
                current_rinse_u = max(0.01, rinse_u * 0.4)
            else:
                rinse_t = times_dict['rinse'] * 0.8
                current_rinse_u = rinse_u * 0.6
                
        stages = [
            ('press', times_dict['press'], Plo, Phi, 'press', yfeed, 0.0),
            ('ads', times_dict['ads'], Phi, Phi, 'inlet', yfeed, feed_u),
            ('rinse', rinse_t, Phi, Phi, 'inlet', np.array([1.0, 0.0]), current_rinse_u),
            ('pe', times_dict['pe'], Phi, Peq, 'pe', None, 0.0),
            ('blow', times_dict['blow'], Peq, Pint, 'vent', None, 0.0),
            ('evac', times_dict['evac'], Pint, Plo, 'vent', None, 0.0)
        ]
        
        for name, duration, p0, p1, mode, iny, u in stages:
            if duration <= 0:
                continue
            dp = (p1 - p0) / duration
            def rhs(t, X):
                s = X[:4*Nz].reshape(4, Nz)
                y = np.clip(s[:2], 0, 1)
                sum_y = np.maximum(y.sum(axis=0), 1e-12)
                y = y / sum_y
                q = s[2:]
                P = p0 + dp * t
                c = P / (R * T)
                dc = dp / (R * T)
                dq = (qstar(y, P) - q) * k[:, None]
                solid = (1 - eps) * rho * dq.sum(axis=0)
                F = np.zeros(Nz + 1)
                if mode == 'inlet':
                    F[0] = u * c
                    for j in range(Nz):
                        F[j+1] = F[j] - dz * (eps * dc + solid[j])
                elif mode in ('press', 'vent'):
                    F[-1] = 0.0
                    for j in range(Nz - 1, -1, -1):
                        F[j] = F[j+1] + dz * (eps * dc + solid[j])
                elif mode == 'pe':
                    F[0] = 0.0
                    for j in range(Nz):
                        F[j+1] = F[j] - dz * (eps * dc + solid[j])
                
                yfcell = y.T
                yf = np.empty((Nz + 1, 2))
                for f in range(Nz + 1):
                    if f == 0:
                        yf[f] = iny if mode in ('inlet', 'press') and F[f] >= 0 and iny is not None else yfcell[0]
                    elif f == Nz:
                        yf[f] = yfcell[-1]
                    else:
                        yf[f] = yfcell[f-1] if F[f] >= 0 else yfcell[f]
                
                Fi = F[:, None] * yf
                dflux = (Fi[1:] - Fi[:-1]) / dz
                dF = (F[1:] - F[:-1]) / dz
                dydt = (-dflux + y.T * (dF + solid)[:, None] - (1 - eps) * rho * dq.T) / (eps * c)
                dstate = np.vstack([dydt.T, dq]).reshape(-1)
                
                feed = np.zeros(2); heavy = np.zeros(2); waste = np.zeros(2); rin = 0.0; work = 0.0
                if name in ('press', 'ads') and F[0] > 0:
                    feed = F[0] * yf[0]
                if name == 'rinse' and F[0] > 0:
                    rin = F[0] * yf[0, 0]
                if name in ('blow', 'evac') and F[0] < 0:
                    heavy = -Fi[0]
                if name in ('ads', 'rinse') and F[-1] > 0:
                    waste = Fi[-1]
                if name == 'evac' and F[0] < 0:
                    work = (-F[0]) * R * T * np.log(max(Phi / P, 1.0)) / 0.75
                
                return np.r_[dstate, feed, heavy, rin, waste, work]

            X0 = np.r_[state, np.zeros(8)]
            sol = solve_ivp(rhs, (0, duration), X0, method='Radau', rtol=1e-3, atol=1e-5, max_step=duration/8)
            if not sol.success:
                raise RuntimeError(f'{label} {name}: {sol.message}')
            state = sol.y[:4*Nz, -1]
            total += sol.y[4*Nz:, -1]
            if keep_profiles:
                prof[name] = state.reshape(4, Nz).copy()
        return state, total, prof

    Pinit = Plo
    y0 = np.tile(yfeed[:, None], (1, Nz))
    q0 = np.repeat(qstar(yfeed, Pinit), Nz, axis=1)
    state = np.vstack([y0, q0]).reshape(-1)
    rows = []
    lastprof = {}
    for cyc in range(1, cycles + 1):
        old = state.copy()
        state, acc, pr = one_cycle(state, True)
        err = float(np.max(np.abs(state - old)))
        feed = acc[:2]; product = acc[2:4]; rinse = acc[4]; waste = acc[5:7]; work = acc[7]
        gross = product[0]; net = max(gross - rinse, 0.0)
        purity = 100.0 * gross / product.sum() if product.sum() > 0 else 0.0
        recovery = 100.0 * net / feed[0] if feed[0] > 0 else 0.0
        mass = net * 0.04401 * A / 1000.0  # kg per cycle
        sec = (work * A / 3.6e6) / mass if mass > 0 else 0.0
        prod_rate = mass / (Mbed * CT / 3600.0)
        rows.append(dict(cycle=cyc, css_error=err, feed_CO2_mol=feed[0], feed_N2_mol=feed[1],
                         product_CO2_gross_mol=gross, product_N2_mol=product[1], rinse_CO2_mol=rinse,
                         product_CO2_net_mol=net, waste_CO2_mol=waste[0], waste_N2_mol=waste[1],
                         work_J=work, purity_pct=purity, recovery_pct=recovery,
                         productivity_kgCO2_kgads_h=prod_rate, specific_energy_kWh_tCO2=sec))
        lastprof = pr
    return pd.DataFrame(rows), state, lastprof

# Quick run test
times = {'press': 20.0, 'ads': 60.0, 'rinse': 30.0, 'pe': 15.0, 'blow': 20.0, 'evac': 65.0}
df1, s1, p1 = run_stage_sim(np.array([0.15, 0.85]), 0.08, times, rinse_u=0.04, label="S1", cycles=4)
print("Stage 1 test done:")
print(df1.iloc[-1][['purity_pct', 'recovery_pct', 'specific_energy_kWh_tCO2', 'feed_CO2_mol', 'product_CO2_net_mol']])
