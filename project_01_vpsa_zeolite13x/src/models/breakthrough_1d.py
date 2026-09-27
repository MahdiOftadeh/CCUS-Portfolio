import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==========================================
# 1. System Parameters
# ==========================================
L = 0.5            # Bed length (m)
N_nodes = 50       # Number of spatial nodes
dz = L / (N_nodes - 1)
z = np.linspace(0, L, N_nodes)

epsilon_b = 0.37   # Bed voidage
rho_p = 1130.0     # Particle density (kg/m3)
u_superficial = 0.05  # Superficial velocity (m/s)
u = u_superficial / epsilon_b  # Interstitial velocity (m/s)
D_ax = 1e-4        # Axial dispersion coefficient (m2/s)

# Kinetic parameters (LDF)
k_LDF_CO2 = 0.169  # (1/s)
k_LDF_N2 = 0.05    # (1/s)

# Equilibrium parameters (Langmuir for Zeolite 13X at 298 K)
qm_CO2 = 3.2       # mol/kg
b_CO2 = 1.8        # 1/bar
qm_N2 = 0.8        # mol/kg
b_N2 = 0.05        # 1/bar

# Feed conditions
P_total = 1.01325  # bar
T = 298.15         # K
R_gas = 8.314e-5   # m3.bar / (mol.K)
c_total = P_total / (R_gas * T)  # total gas concentration (mol/m3)

y_feed_CO2 = 0.15
y_feed_N2 = 0.85
c_feed_CO2 = y_feed_CO2 * c_total
c_feed_N2 = y_feed_N2 * c_total

# ==========================================
# 2. Model Equations (ODEs)
# ==========================================
def isotherm(c_CO2, c_N2):
    p_CO2 = (c_CO2 / c_total) * P_total
    p_N2 = (c_N2 / c_total) * P_total
    denom = 1.0 + b_CO2 * p_CO2 + b_N2 * p_N2
    q_star_CO2 = (qm_CO2 * b_CO2 * p_CO2) / denom
    q_star_N2 = (qm_N2 * b_N2 * p_N2) / denom
    return q_star_CO2, q_star_N2

def model_ode(t, y):
    # State vector unpack:
    # y = [c_CO2 (N), c_N2 (N), q_CO2 (N), q_N2 (N)]
    c_CO2 = y[0:N_nodes]
    c_N2 = y[N_nodes:2*N_nodes]
    q_CO2 = y[2*N_nodes:3*N_nodes]
    q_N2 = y[3*N_nodes:4*N_nodes]
    
    q_star_CO2, q_star_N2 = isotherm(c_CO2, c_N2)
    
    dq_CO2_dt = k_LDF_CO2 * (q_star_CO2 - q_CO2)
    dq_N2_dt = k_LDF_N2 * (q_star_N2 - q_N2)
    
    dc_CO2_dt = np.zeros(N_nodes)
    dc_N2_dt = np.zeros(N_nodes)
    
    coeff = (1.0 - epsilon_b) * rho_p / epsilon_b
    
    # Interior nodes (Upwind for advection + Central for dispersion)
    for i in range(1, N_nodes - 1):
        adv_CO2 = -u * (c_CO2[i] - c_CO2[i-1]) / dz
        disp_CO2 = D_ax * (c_CO2[i+1] - 2.0*c_CO2[i] + c_CO2[i-1]) / (dz**2)
        dc_CO2_dt[i] = adv_CO2 + disp_CO2 - coeff * dq_CO2_dt[i]
        
        adv_N2 = -u * (c_N2[i] - c_N2[i-1]) / dz
        disp_N2 = D_ax * (c_N2[i+1] - 2.0*c_N2[i] + c_N2[i-1]) / (dz**2)
        dc_N2_dt[i] = adv_N2 + disp_N2 - coeff * dq_N2_dt[i]
        
    # Boundary conditions:
    # Node 0: Danckwerts inlet
    dc_CO2_dt[0] = (u / dz) * (c_feed_CO2 - c_CO2[0]) - coeff * dq_CO2_dt[0]
    dc_N2_dt[0] = (u / dz) * (c_feed_N2 - c_N2[0]) - coeff * dq_N2_dt[0]
    
    # Node N-1: Zero Neumann outlet
    dc_CO2_dt[-1] = -u * (c_CO2[-1] - c_CO2[-2]) / dz - coeff * dq_CO2_dt[-1]
    dc_N2_dt[-1] = -u * (c_N2[-1] - c_N2[-2]) / dz - coeff * dq_N2_dt[-1]
    
    return np.concatenate([dc_CO2_dt, dc_N2_dt, dq_CO2_dt, dq_N2_dt])

# ==========================================
# 3. Execution & Integration
# ==========================================
# Initial condition: Bed saturated with pure N2
y0 = np.zeros(4 * N_nodes)
y0[N_nodes:2*N_nodes] = c_feed_N2  # c_N2(z, 0)
# Equilibrium solid phase for initial N2
q_star_init_CO2, q_star_init_N2 = isotherm(np.zeros(N_nodes), np.full(N_nodes, c_feed_N2))
y0[3*N_nodes:4*N_nodes] = q_star_init_N2

t_span = (0.0, 600.0)  # 600 seconds simulation
t_eval = np.linspace(0.0, 600.0, 300)

sol = solve_ivp(model_ode, t_span, y0, t_eval=t_eval, method='BDF')

# ==========================================
# 4. Plotting Breakthrough Curves
# ==========================================
c_out_CO2 = sol.y[N_nodes - 1, :]
c_out_N2 = sol.y[2 * N_nodes - 1, :]

plt.figure(figsize=(8, 5))
plt.plot(sol.t, c_out_CO2 / c_feed_CO2, label="CO2 (Outlet / Feed)", color="crimson", linewidth=2)
plt.plot(sol.t, c_out_N2 / c_feed_N2, label="N2 (Outlet / Feed)", color="navy", linestyle="--", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Normalized Outlet Concentration (C / C_feed)")
plt.title("1D Breakthrough Curve: CO2/N2 on Zeolite 13X")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("breakthrough_curve.png", dpi=300)
print("Simulation complete. Output saved as breakthrough_curve.png")
