#!/usr/bin/env python3
"""Refined 1D finite-volume 2-stage VPSA cycle model with vector isotherm."""
import os, json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)

R = 8.314462618; T = 298.15; L = 1.0; D = 0.05; A = np.pi * D**2 / 4; eps = 0.37; rho = 1130.
Nz = 10; dz = L / Nz; z = (np.arange(Nz) + 0.5) * dz; Mbed = rho * (1 - eps) * A * L

# 13X legacy project constants split into 2 equal sites
qm = np.array([[2.425, 1.05], [2.425, 1.05]])  # (sites, components)
b = np.array([[3.25e-5, 0.12e-5], [3.25e-5, 0.12e-5]])  # (sites, components)
k = np.array([0.18, 0.25])
Plo, Pint, Peq, Phi = 0.05e5, 0.4e5, 0.95e5, 1.5e5
times = {'press': 20.0, 'ads': 60.0, 'rinse': 30.0, 'pe': 15.0, 'blow': 20.0, 'evac': 65.0}
CT = sum(times.values())

def qstar(y, P):
    """Vectorized isotherm supporting (2,), (2, Nz), or (Nz, 2) arrays."""
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

# Quick verification of qstar with spatial 1D vectors
test_p = qstar(np.vstack([np.full(Nz, 0.15), np.full(Nz, 0.85)]), Phi)
assert test_p.shape == (2, Nz)

def run_stage(yfeed, feed_u, label, cycles=6):
    def one_cycle(state, keep_profiles=False):
        total = np.zeros(8); prof = {}
        stages = [
            ('press', times['press'], Plo, Phi, 'press', yfeed, 0.0),
            ('ads', times['ads'], Phi, Phi, 'inlet', yfeed, feed_u),
            ('rinse', times['rinse'], Phi, Phi, 'inlet', np.array([1.0, 0.0]), 0.04),
            ('pe', times['pe'], Phi, Peq, 'pe', None, 0.0),
            ('blow', times['blow'], Peq, Pint, 'vent', None, 0.0),
            ('evac', times['evac'], Pint, Plo, 'vent', None, 0.0)
        ]
        for name, duration, p0, p1, mode, iny, u in stages:
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
            sol = solve_ivp(rhs, (0, duration), X0, method='Radau', rtol=1e-4, atol=1e-7, max_step=duration/10)
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
        mass = net * 0.04401 / 1000.0
        sec = (work / 3.6e6) / mass if mass > 0 else 0.0
        prod_rate = mass / (Mbed * CT / 3600.0)
        rows.append(dict(cycle=cyc, css_error=err, feed_CO2_mol=feed[0], feed_N2_mol=feed[1],
                         product_CO2_gross_mol=gross, product_N2_mol=product[1], rinse_CO2_mol=rinse,
                         product_CO2_net_mol=net, purity_pct=purity, recovery_pct=recovery,
                         productivity_kgCO2_kgads_h=prod_rate, specific_energy_kWh_tCO2=sec))
        lastprof = pr
        print(f"[{label}] Cycle {cyc}: err={err:.3e}, purity={purity:.2f}%, rec={recovery:.2f}%")
    return pd.DataFrame(rows), state, lastprof

print("Starting Stage 1...")
y1 = np.array([0.15, 0.85])
df1, s1, p1 = run_stage(y1, 0.08, 'Stage 1', cycles=6)
r1 = df1.iloc[-1]
prod1 = np.array([r1.product_CO2_gross_mol, r1.product_N2_mol])
y2 = prod1 / prod1.sum() if prod1.sum() > 0 else y1
F2 = prod1.sum() / CT
u2 = F2 * R * T / (Phi * A)
print(f"Starting Stage 2 with Feed CO2={y2[0]*100:.2f}%, velocity={u2:.4f} m/s...")
df2, s2, p2 = run_stage(y2, u2, 'Stage 2', cycles=6)
r2 = df2.iloc[-1]

def summary_metrics(row, bedmass):
    return {
        'purity_pct': float(row.purity_pct),
        'recovery_pct': float(row.recovery_pct),
        'productivity_kg_CO2_per_kg_ads_h': float(row.productivity_kgCO2_kgads_h),
        'specific_energy_kWh_per_tonne_CO2': float(row.specific_energy_kWh_tCO2),
        'feed_CO2_mol_per_cycle': float(row.feed_CO2_mol),
        'product_CO2_net_mol_per_cycle': float(row.product_CO2_net_mol),
        'bed_adsorbent_mass_kg': float(bedmass)
    }

overall_rec = 100.0 * max(float(r2.product_CO2_net_mol), 0.0) / max(float(r1.feed_CO2_mol), 1e-30)
overall_sec = float(r1.specific_energy_kWh_tCO2) + float(r2.specific_energy_kWh_tCO2)
overall_prod = float(r2.product_CO2_net_mol) * 0.04401 / (2 * Mbed * CT / 3600.0)

summary = {
    'model': 'Rigorous 1-D conservative finite-volume MOL with vector Dual-Site Langmuir isotherm',
    'parameters': {
        'Nz': Nz, 'L_m': L, 'D_m': D, 'eps': eps, 'rho_s_kg_m3': rho,
        'P_high_bar': Phi / 1e5, 'P_low_bar': Plo / 1e5, 'T_K': T, 'cycle_time_s': CT
    },
    'stage1': summary_metrics(r1, Mbed),
    'stage2': {**summary_metrics(r2, Mbed), 'feed_CO2_mole_fraction': float(y2[0]), 'feed_velocity_m_s': float(u2)},
    'overall': {
        'purity_pct': float(r2.purity_pct),
        'recovery_pct': overall_rec,
        'productivity_kg_CO2_per_kg_ads_h': overall_prod,
        'specific_energy_kWh_per_tonne_CO2': overall_sec
    },
    'checks': {
        'stage1_purity_in_0_100': bool(0 <= r1.purity_pct <= 100),
        'stage1_recovery_in_0_100': bool(0 <= r1.recovery_pct <= 100),
        'stage2_purity_in_0_100': bool(0 <= r2.purity_pct <= 100),
        'stage2_recovery_in_0_100': bool(0 <= r2.recovery_pct <= 100),
        'overall_recovery_in_0_100': bool(0 <= overall_rec <= 100),
        'meets_95pct_stage2_purity': bool(r2.purity_pct >= 95.0)
    }
}

json_path = OUT / 'robust_dual_stage_summary.json'
csv_path = OUT / 'robust_dual_stage_summary.csv'
json_path.write_text(json.dumps(summary, indent=2) + chr(10))

csv_rows = [
    {'stage': 'Stage 1', **summary['stage1']},
    {'stage': 'Stage 2', **summary['stage2']},
    {'stage': 'Overall', **summary['overall']}
]
pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

# Plot dashboard
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(z, p1['ads'][0], 'b-o', label='y_CO2 (after AD)')
axes[0, 0].plot(z, p1['rinse'][0], 'g-s', label='y_CO2 (after RI)')
axes[0, 0].plot(z, p1['evac'][0], 'r-^', label='y_CO2 (after EVAC)')
axes[0, 0].set(title='Stage 1 Axial Gas Profiles', xlabel='Bed Length z (m)', ylabel='CO2 Mole Fraction')
axes[0, 0].grid(True); axes[0, 0].legend()

axes[0, 1].plot(z, p2['ads'][0], 'b-o', label='y_CO2 (after AD)')
axes[0, 1].plot(z, p2['rinse'][0], 'g-s', label='y_CO2 (after RI)')
axes[0, 1].plot(z, p2['evac'][0], 'r-^', label='y_CO2 (after EVAC)')
axes[0, 1].set(title='Stage 2 Axial Gas Profiles', xlabel='Bed Length z (m)', ylabel='CO2 Mole Fraction')
axes[0, 1].grid(True); axes[0, 1].legend()

axes[1, 0].plot(df1.cycle, df1.purity_pct, 'b-o', label='Stage 1 Purity (%)')
axes[1, 0].plot(df1.cycle, df1.recovery_pct, 'r--s', label='Stage 1 Recovery (%)')
axes[1, 0].set(title='Stage 1 Cyclic Convergence', xlabel='Cycle Number', ylabel='Percentage (%)')
axes[1, 0].set_ylim(0, 105); axes[1, 0].grid(True); axes[1, 0].legend()

axes[1, 1].plot(df2.cycle, df2.purity_pct, 'b-o', label='Stage 2 Purity (%)')
axes[1, 1].plot(df2.cycle, df2.recovery_pct, 'r--s', label='Stage 2 Recovery (%)')
axes[1, 1].set(title='Stage 2 Cyclic Convergence', xlabel='Cycle Number', ylabel='Percentage (%)')
axes[1, 1].set_ylim(0, 105); axes[1, 1].grid(True); axes[1, 1].legend()

fig.tight_layout()
png_path = OUT / 'robust_dual_stage_dashboard.png'
fig.savefig(png_path, dpi=160)
plt.close(fig)

print("EXECUTION COMPLETE.")
print(json.dumps(summary, indent=2))
