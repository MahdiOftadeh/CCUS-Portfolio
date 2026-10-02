"""
Complete 4-Step VPSA Cycle Simulation on Zeolite 13X
Author: Mahdi Oftadeh / Chemical Engineering
Target: CO2 Capture from Flue Gas (15% CO2 / 85% N2)
Directory Architecture: project_01_vpsa_zeolite13x/ (src/, data/, results/)
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------------------------------------------------
# 1. مسیردهی پوشه‌ها بر اساس ساختار استاندارد
# -------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for d in [SRC_DIR, DATA_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# -------------------------------------------------------------
# 2. ثوابت فیزیکی و ویژگی‌های سیستم بستر و جاذب (زئولیت 13X)
# -------------------------------------------------------------
R_gas = 8.314462   # J / (mol * K)
T = 298.15         # K

# مشخصات ستون
L_bed = 1.0        # طول بستر (متر)
D_bed = 0.05       # قطر بستر (متر)
A_bed = np.pi * (D_bed / 2)**2
V_bed = A_bed * L_bed
eps_bed = 0.37     # تخلخل بستر
rho_s = 1130.0     # چگالی اسکلت جامد (kg/m3)

# مش‌بندی بستر (Method of Lines)
N = 20
dz = L_bed / N
z_nodes = np.linspace(dz / 2, L_bed - dz / 2, N)

# پارامترهای ایزوترم دوجزیی لانگمویر و سینتیک LDF
# جزء ۱: CO2 | جزء ۲: N2
q_max_CO2 = 4.85   # mol/kg
b_CO2 = 3.25       # bar^-1
k_ldf_CO2 = 0.18   # 1/s

q_max_N2 = 2.10    # mol/kg
b_N2 = 0.12        # bar^-1
k_ldf_N2 = 0.25    # 1/s

# سطوح فشار عملیاتی
P_high = 1.5e5     # 1.50 bar (جذب)
P_low = 0.10e5     # 0.10 bar (خلأ)

# مشخصات گاز خوراک
y_CO2_feed = 0.15
y_N2_feed = 0.85

# زمان‌بندی ۴ مرحله چرخه
t_press = 15.0     # تراکم با خوراک (PR)
t_ads = 45.0       # جذب در فشار بالا (AD)
t_blow = 20.0      # تخلیه و افت فشار (BD)
t_purge = 40.0     # شستشو در خلأ (PU)
t_cycle = t_press + t_ads + t_blow + t_purge

# سرعت‌های بینابینی گاز (m/s)
u_press = 0.12     # هم‌جهت (0 -> L)
u_ads = 0.20       # هم‌جهت (0 -> L)
u_blow = -0.15     # پادجریان (L -> 0)
u_purge = -0.08    # پادجریان (L -> 0)

# ذخیره پارامترها در data/
params = {
    "L_bed_m": L_bed, "D_bed_m": D_bed, "eps_bed": eps_bed, "rho_s_kg_m3": rho_s,
    "T_K": T, "P_high_bar": P_high / 1e5, "P_low_bar": P_low / 1e5,
    "y_CO2_feed": y_CO2_feed, "y_N2_feed": y_N2_feed,
    "t_press_s": t_press, "t_ads_s": t_ads, "t_blow_s": t_blow, "t_purge_s": t_purge,
    "t_cycle_s": t_cycle, "N_nodes": N
}
with open(os.path.join(DATA_DIR, "vpsa_cycle_parameters.json"), "w", encoding="utf-8") as f:
    json.dump(params, f, indent=4)

# -------------------------------------------------------------
# 3. توابع تعادل ترمودینامیکی و معادلات دیفرانسیل انتقال جرم
# -------------------------------------------------------------
def extended_langmuir(c_CO2, c_N2, T_k=T):
    p_CO2 = np.maximum(c_CO2 * R_gas * T_k / 1e5, 0.0)
    p_N2 = np.maximum(c_N2 * R_gas * T_k / 1e5, 0.0)
    denom = 1.0 + b_CO2 * p_CO2 + b_N2 * p_N2
    q_CO2_star = (q_max_CO2 * b_CO2 * p_CO2) / denom
    q_N2_star = (q_max_N2 * b_N2 * p_N2) / denom
    return q_CO2_star, q_N2_star

def pack_state(c_CO2, c_N2, q_CO2, q_N2):
    return np.concatenate([c_CO2, c_N2, q_CO2, q_N2])

def unpack_state(y_vec):
    return y_vec[0:N], y_vec[N:2*N], y_vec[2*N:3*N], y_vec[3*N:4*N]

def odes_stage(t, y, u, c_in_CO2, c_in_N2, P_curr):
    c_CO2, c_N2, q_CO2, q_N2 = unpack_state(y)
    
    q_star_CO2, q_star_N2 = extended_langmuir(c_CO2, c_N2)
    dq_CO2_dt = k_ldf_CO2 * (q_star_CO2 - q_CO2)
    dq_N2_dt = k_ldf_N2 * (q_star_N2 - q_N2)
    
    dc_CO2_dz = np.zeros(N)
    dc_N2_dz = np.zeros(N)
    
    if u >= 0:  # جریان مستقیم (0 -> L)
        dc_CO2_dz[0] = (c_CO2[0] - c_in_CO2) / dz
        dc_N2_dz[0] = (c_N2[0] - c_in_N2) / dz
        dc_CO2_dz[1:] = (c_CO2[1:] - c_CO2[:-1]) / dz
        dc_N2_dz[1:] = (c_N2[1:] - c_N2[:-1]) / dz
    else:       # جریان معکوس (L -> 0)
        dc_CO2_dz[-1] = (c_in_CO2 - c_CO2[-1]) / dz
        dc_N2_dz[-1] = (c_in_N2 - c_N2[-1]) / dz
        dc_CO2_dz[:-1] = (c_CO2[1:] - c_CO2[:-1]) / dz
        dc_N2_dz[:-1] = (c_N2[1:] - c_N2[:-1]) / dz
        
    dc_CO2_dt = (- u * dc_CO2_dz - ((1.0 - eps_bed) * rho_s / eps_bed) * dq_CO2_dt)
    dc_N2_dt = (- u * dc_N2_dz - ((1.0 - eps_bed) * rho_s / eps_bed) * dq_N2_dt)
    
    return pack_state(dc_CO2_dt, dc_N2_dt, dq_CO2_dt, dq_N2_dt)

# -------------------------------------------------------------
# 4. حلقه همگرایی به حالت پایدار چرخه‌ای (CSS)
# -------------------------------------------------------------
c_tot_init = P_low / (R_gas * T)
c_CO2_init = np.full(N, 0.001 * c_tot_init)
c_N2_init = np.full(N, 0.999 * c_tot_init)
q_CO2_init, q_N2_init = extended_langmuir(c_CO2_init, c_N2_init)
y_state = pack_state(c_CO2_init, c_N2_init, q_CO2_init, q_N2_init)

c_tot_feed = P_high / (R_gas * T)
c_in_CO2_feed = y_CO2_feed * c_tot_feed
c_in_N2_feed = y_N2_feed * c_tot_feed

c_tot_low = P_low / (R_gas * T)
c_in_CO2_purge = 0.0
c_in_N2_purge = c_tot_low

cycle_history = []
n_cycles = 10

for cycle_idx in range(1, n_cycles + 1):
    sol_s1 = solve_ivp(odes_stage, [0, t_press], y_state, args=(u_press, c_in_CO2_feed, c_in_N2_feed, P_high), method='RK23', rtol=1e-3, atol=1e-4)
    sol_s2 = solve_ivp(odes_stage, [0, t_ads], sol_s1.y[:, -1], args=(u_ads, c_in_CO2_feed, c_in_N2_feed, P_high), method='RK23', rtol=1e-3, atol=1e-4)
    sol_s3 = solve_ivp(odes_stage, [0, t_blow], sol_s2.y[:, -1], args=(u_blow, 0.0, 0.05 * c_tot_low, P_low), method='RK23', rtol=1e-3, atol=1e-4)
    sol_s4 = solve_ivp(odes_stage, [0, t_purge], sol_s3.y[:, -1], args=(u_purge, c_in_CO2_purge, c_in_N2_purge, P_low), method='RK23', rtol=1e-3, atol=1e-4)
    
    y_after_s4 = sol_s4.y[:, -1]
    diff = np.linalg.norm(y_after_s4 - y_state) / (np.linalg.norm(y_state) + 1e-8)
    cycle_history.append({
        'cycle': cycle_idx,
        'relative_change': float(diff),
        'q_CO2_avg_end': float(np.mean(y_after_s4[2*N:3*N])),
        'q_N2_avg_end': float(np.mean(y_after_s4[3*N:4*N]))
    })
    y_state = y_after_s4
    if diff < 0.01 and cycle_idx >= 4:
        break

# تحلیل دقیق چرخه پایدار نهایی
t_eval_s1 = np.linspace(0, t_press, 25)
t_eval_s2 = np.linspace(0, t_ads, 35)
t_eval_s3 = np.linspace(0, t_blow, 25)
t_eval_s4 = np.linspace(0, t_purge, 35)

sol_css_s1 = solve_ivp(odes_stage, [0, t_press], y_state, args=(u_press, c_in_CO2_feed, c_in_N2_feed, P_high), t_eval=t_eval_s1, method='RK23')
sol_css_s2 = solve_ivp(odes_stage, [0, t_ads], sol_css_s1.y[:, -1], args=(u_ads, c_in_CO2_feed, c_in_N2_feed, P_high), t_eval=t_eval_s2, method='RK23')
sol_css_s3 = solve_ivp(odes_stage, [0, t_blow], sol_css_s2.y[:, -1], args=(u_blow, 0.0, 0.05 * c_tot_low, P_low), t_eval=t_eval_s3, method='RK23')
sol_css_s4 = solve_ivp(odes_stage, [0, t_purge], sol_css_s3.y[:, -1], args=(u_purge, c_in_CO2_purge, c_in_N2_purge, P_low), t_eval=t_eval_s4, method='RK23')

# -------------------------------------------------------------
# 5. محاسبه شاخص‌های کلیدی عملکرد (KPIs)
# -------------------------------------------------------------
n_feed_s1_CO2 = abs(u_press) * A_bed * eps_bed * c_in_CO2_feed * t_press
n_feed_s2_CO2 = abs(u_ads) * A_bed * eps_bed * c_in_CO2_feed * t_ads
n_feed_CO2_total = n_feed_s1_CO2 + n_feed_s2_CO2

n_feed_s1_N2 = abs(u_press) * A_bed * eps_bed * c_in_N2_feed * t_press
n_feed_s2_N2 = abs(u_ads) * A_bed * eps_bed * c_in_N2_feed * t_ads
n_feed_N2_total = n_feed_s1_N2 + n_feed_s2_N2
n_feed_total = n_feed_CO2_total + n_feed_N2_total

c_CO2_s2_exit = sol_css_s2.y[N-1, :]
c_N2_s2_exit = sol_css_s2.y[2*N-1, :]
n_prod_light_CO2 = float(np.trapz(abs(u_ads) * A_bed * eps_bed * c_CO2_s2_exit, t_eval_s2))
n_prod_light_N2 = float(np.trapz(abs(u_ads) * A_bed * eps_bed * c_N2_s2_exit, t_eval_s2))
n_purge_in_N2 = float(abs(u_purge) * A_bed * eps_bed * c_in_N2_purge * t_purge)

c_CO2_s3_exit = sol_css_s3.y[0, :]
c_N2_s3_exit = sol_css_s3.y[N, :]
n_heavy_s3_CO2 = float(np.trapz(abs(u_blow) * A_bed * eps_bed * c_CO2_s3_exit, t_eval_s3))
n_heavy_s3_N2 = float(np.trapz(abs(u_blow) * A_bed * eps_bed * c_N2_s3_exit, t_eval_s3))

c_CO2_s4_exit = sol_css_s4.y[0, :]
c_N2_s4_exit = sol_css_s4.y[N, :]
n_heavy_s4_CO2 = float(np.trapz(abs(u_purge) * A_bed * eps_bed * c_CO2_s4_exit, t_eval_s4))
n_heavy_s4_N2 = float(np.trapz(abs(u_purge) * A_bed * eps_bed * c_N2_s4_exit, t_eval_s4))

n_heavy_CO2_total = n_heavy_s3_CO2 + n_heavy_s4_CO2
n_heavy_N2_total = n_heavy_s3_N2 + n_heavy_s4_N2
n_heavy_total = n_heavy_CO2_total + n_heavy_N2_total

co2_purity = float((n_heavy_CO2_total / n_heavy_total) * 100.0)
co2_recovery = float((n_heavy_CO2_total / n_feed_CO2_total) * 100.0)
m_adsorbent_kg = float(rho_s * (1.0 - eps_bed) * V_bed)
productivity = float((n_heavy_CO2_total * 44.01e-3) / (m_adsorbent_kg * (t_cycle / 3600.0)))

# مصرف انرژی کمپرسور و پمپ خلأ
gamma = 1.4
eta_eff = 0.75
W_comp_kJ = float((n_feed_total * R_gas * T * (gamma / (gamma - 1)) * ((P_high / 1.0e5)**((gamma-1)/gamma) - 1.0) / eta_eff) / 1000.0)
W_vac_kJ = float((n_heavy_total * R_gas * T * (gamma / (gamma - 1)) * ((1.0e5 / P_low)**((gamma-1)/gamma) - 1.0) / eta_eff) / 1000.0)
specific_energy_MJ_ton = float(((W_comp_kJ + W_vac_kJ) / (n_heavy_CO2_total * 44.01e-3)) * 1000.0 / 1000.0)
specific_energy_kWh_ton = float(specific_energy_MJ_ton / 3.6)

# -------------------------------------------------------------
# 6. ذخیره‌سازی خروجی‌ها و رسم نمودارها
# -------------------------------------------------------------
pd.DataFrame(cycle_history).to_csv(os.path.join(DATA_DIR, "css_convergence.csv"), index=False)
df_axial = pd.DataFrame({
    'z_position_m': z_nodes,
    'stage1_q_CO2_mol_kg': sol_css_s1.y[2*N:3*N, -1],
    'stage2_q_CO2_mol_kg': sol_css_s2.y[2*N:3*N, -1],
    'stage3_q_CO2_mol_kg': sol_css_s3.y[2*N:3*N, -1],
    'stage4_q_CO2_mol_kg': sol_css_s4.y[2*N:3*N, -1],
})
df_axial.to_csv(os.path.join(DATA_DIR, "axial_profiles_steady_state.csv"), index=False)

results_summary = {
    "CO2_Purity_percent": round(co2_purity, 2),
    "CO2_Recovery_percent": round(co2_recovery, 2),
    "Productivity_kg_kgads_h": round(productivity, 4),
    "Specific_Energy_kWh_tonCO2": round(specific_energy_kWh_ton, 2),
    "Cycle_Time_s": round(t_cycle, 1),
    "CSS_Cycles_Count": len(cycle_history)
}
with open(os.path.join(RESULTS_DIR, "vpsa_metrics_summary.json"), "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=4)

# رسم شکل‌ها در results/
plt.figure(figsize=(7, 4.5))
plt.plot(z_nodes, sol_css_s1.y[2*N:3*N, 0], 'b--', label='q_CO2 (t = 0 s)')
plt.plot(z_nodes, sol_css_s1.y[2*N:3*N, -1], 'b-', lw=2, label='q_CO2 (t = 15 s)')
plt.plot(z_nodes, sol_css_s1.y[3*N:4*N, -1], 'g-', lw=1.5, label='q_N2 (t = 15 s)')
plt.title("Stage 1: Feed Pressurization Axial Loading", fontweight='bold')
plt.xlabel("Bed Position z (m)")
plt.ylabel("Solid Loading q (mol/kg)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "01_stage1_pressurization.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
for idx, ti in enumerate([0, 10, 20, len(t_eval_s2)-1]):
    plt.plot(z_nodes, sol_css_s2.y[2*N:3*N, ti], label=f'CO2 t={t_eval_s2[ti]:.0f}s', lw=1.8)
plt.title("Stage 2: Adsorption & Mass Transfer Zone (MTZ)", fontweight='bold')
plt.xlabel("Bed Position z (m)")
plt.ylabel("Solid Loading q (mol/kg)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "02_stage2_adsorption.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.plot(z_nodes, sol_css_s3.y[2*N:3*N, 0], 'r--', label='q_CO2 Start (High-P)', lw=2)
plt.plot(z_nodes, sol_css_s3.y[2*N:3*N, -1], 'r-', label='q_CO2 End (Low-P)', lw=2.5)
plt.title("Stage 3: Counter-Current Blowdown Profile", fontweight='bold')
plt.xlabel("Bed Position z (m)")
plt.ylabel("Solid Loading q (mol/kg)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "03_stage3_blowdown.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.plot(z_nodes, sol_css_s4.y[2*N:3*N, 0], 'purple', linestyle='--', label='q_CO2 Start Purge', lw=2)
plt.plot(z_nodes, sol_css_s4.y[2*N:3*N, -1], 'purple', linestyle='-', label='q_CO2 End Purge', lw=2.5)
plt.title("Stage 4: Vacuum Purge Bed Regeneration", fontweight='bold')
plt.xlabel("Bed Position z (m)")
plt.ylabel("Solid Loading q (mol/kg)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "04_stage4_purge.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.plot([c['cycle'] for c in cycle_history], [c['relative_change'] for c in cycle_history], 'go-', lw=2)
plt.axhline(0.01, color='r', linestyle=':', label='Convergence Threshold (<1%)')
plt.yscale('log')
plt.title("CSS Convergence History", fontweight='bold')
plt.xlabel("Cycle Number")
plt.ylabel("Relative Change")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "05_css_convergence.png"), dpi=200)
plt.close()

# رسم داشبورد ۴ تایی
fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
axs[0, 0].step([0, t_press, t_press+t_ads, t_press+t_ads+t_blow, t_cycle],
               [P_low/1e5, P_high/1e5, P_high/1e5, P_low/1e5, P_low/1e5], where='post', color='#1f77b4', lw=2.5)
axs[0, 0].set_title("1. Column Pressure Cycle (bar)", fontweight='bold')
axs[0, 0].grid(True, alpha=0.3)

axs[0, 1].plot(z_nodes, sol_css_s1.y[2*N:3*N, -1], label='Pressurization')
axs[0, 1].plot(z_nodes, sol_css_s2.y[2*N:3*N, -1], label='Adsorption')
axs[0, 1].plot(z_nodes, sol_css_s3.y[2*N:3*N, -1], label='Blowdown')
axs[0, 1].plot(z_nodes, sol_css_s4.y[2*N:3*N, -1], label='Purge')
axs[0, 1].set_title("2. CO2 Axial Loading per Stage", fontweight='bold')
axs[0, 1].legend(fontsize=8)
axs[0, 1].grid(True, alpha=0.3)

bars = ['Feed In', 'Light Out', 'Heavy Out', 'Purge In']
axs[1, 0].bar(np.arange(4) - 0.175, [n_feed_CO2_total, n_prod_light_CO2, n_heavy_CO2_total, 0], 0.35, label='CO2', color='#d62728')
axs[1, 0].bar(np.arange(4) + 0.175, [n_feed_N2_total, n_prod_light_N2, n_heavy_N2_total, n_purge_in_N2], 0.35, label='N2', color='#1f77b4')
axs[1, 0].set_title("3. Molar Balances per Cycle", fontweight='bold')
axs[1, 0].set_xticks(range(4))
axs[1, 0].set_xticklabels(bars, fontsize=9)
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

axs[1, 1].axis('off')
kpi_box = f"""===== VPSA Performance KPIs =====
• CO2 Purity:      {co2_purity:.2f} %
• CO2 Recovery:    {co2_recovery:.2f} %
• Productivity:    {productivity:.3f} kg/(kg_ads*h)
• Specific Energy: {specific_energy_kWh_ton:.1f} kWh / ton CO2
• Net Cycle Time:  {t_cycle:.1f} s
• Adsorbent:       Zeolite 13X
"""
axs[1, 1].text(0.1, 0.5, kpi_box, fontsize=11, family='monospace', verticalalignment='center',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#ced4da', lw=1.5))

plt.suptitle("4-Stage VPSA Complete Process Dashboard", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "06_overall_vpsa_dashboard.png"), dpi=200)
plt.close()
