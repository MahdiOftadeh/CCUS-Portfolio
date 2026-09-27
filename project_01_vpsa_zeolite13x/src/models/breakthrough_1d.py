import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Parameters ---
L = 0.5              # Bed length (m)
nodes = 50           # Number of spatial nodes
dz = L / (nodes - 1)
u = 0.05             # Superficial velocity (m/s)
epsilon_b = 0.37     # Bed porosity
rho_p = 1130         # Particle density (kg/m3)
D_ax = 1e-4          # Axial dispersion (m2/s)

# Adsorption parameters (Langmuir + LDF)
k_LDF_CO2 = 0.169    # 1/s
k_LDF_N2 = 0.05      # 1/s
qm_CO2 = 3.2         # mol/kg
b_CO2 = 1.8          # 1/bar
qm_N2 = 0.8          # mol/kg
b_N2 = 0.05          # 1/bar

# Feed conditions
P = 1.01325          # bar
T = 298.15           # K
y_CO2_feed = 0.15
y_N2_feed = 0.85

# --- Model Definition ---
def breakthrough_model(t, y):
    c_CO2 = y[0:nodes]
    c_N2 = y[nodes:2*nodes]
    q_CO2 = y[2*nodes:3*nodes]
    q_N2 = y[3*nodes:4*nodes]
    
    dcdt_CO2 = np.zeros(nodes)
    dcdt_N2 = np.zeros(nodes)
    dqdt_CO2 = np.zeros(nodes)
    dqdt_N2 = np.zeros(nodes)
    
    # Mass transfer (LDF)
    q_eq_CO2 = qm_CO2 * b_CO2 * c_CO2 / (1 + b_CO2 * c_CO2)
    q_eq_N2 = qm_N2 * b_N2 * c_N2 / (1 + b_N2 * c_N2)
    
    dqdt_CO2 = k_LDF_CO2 * (q_eq_CO2 - q_CO2)
    dqdt_N2 = k_LDF_N2 * (q_eq_N2 - q_N2)
    
    # Gas phase mass balance (Advection + Dispersion)
    factor = ((1 - epsilon_b) / epsilon_b) * rho_p
    
    for i in range(1, nodes - 1):
        dcdt_CO2[i] = -u * (c_CO2[i] - c_CO2[i-1])/dz + D_ax * (c_CO2[i+1] - 2*c_CO2[i] + c_CO2[i-1])/dz**2 - factor * dqdt_CO2[i]
        dcdt_N2[i] = -u * (c_N2[i] - c_N2[i-1])/dz + D_ax * (c_N2[i+1] - 2*c_N2[i] + c_N2[i-1])/dz**2 - factor * dqdt_N2[i]
        
    # Boundary Conditions
    dcdt_CO2[0] = (c_CO2[0] - y_CO2_feed) * 100 
    dcdt_CO2[-1] = (c_CO2[-1] - c_CO2[-2]) / dz
    
    dcdt_N2[0] = (c_N2[0] - y_N2_feed) * 100
    dcdt_N2[-1] = (c_N2[-1] - c_N2[-2]) / dz
    
    return np.concatenate([dcdt_CO2, dcdt_N2, dqdt_CO2, dqdt_N2])

# --- Simulation ---
y0 = np.zeros(4 * nodes)
t_span = (0.0, 3500.0)  # Extended time for full breakthrough
t_eval = np.linspace(0.0, 3500.0, 500)

solution = solve_ivp(breakthrough_model, t_span, y0, method='BDF', t_eval=t_eval)

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[nodes-1] / y_CO2_feed, 'r-', linewidth=2, label='CO2 (Outlet / Feed)')
plt.plot(solution.t, solution.y[2*nodes-1] / y_N2_feed, 'b--', linewidth=2, label='N2 (Outlet / Feed)')
plt.title('1D Breakthrough Curve: CO2/N2 on Zeolite 13X')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Outlet Concentration (C / C_feed)')
plt.legend()
plt.grid(True, linestyle=':')
plt.savefig('breakthrough_curve.png')
print("Simulation complete. Output saved as breakthrough_curve01.png")
