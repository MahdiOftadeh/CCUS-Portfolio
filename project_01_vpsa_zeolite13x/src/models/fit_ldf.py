import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA_PATH = os.path.join("data", "ldf_data.csv")
OUTPUT_PLOT = os.path.join("data", "ldf_fit_validation.png")

c_bulk = 40.0
R_p = 0.001
D_eff = 2.0e-8

k_ldf_theory = 15.0 * D_eff / (R_p * R_p)

print("Reading data...")
try:
    df = pd.read_csv(DATA_PATH, comment="%", sep=None, engine="python", header=None)
    t_data = df.iloc[:, 0].to_numpy()
    c_avg_data = df.iloc[:, 1].to_numpy()
except Exception:
    df = pd.read_csv(DATA_PATH)
    t_data = df.iloc[:, 0].to_numpy()
    c_avg_data = df.iloc[:, 1].to_numpy()

def ldf_model(t, k):
    return c_bulk * (1.0 - np.exp(-k * t))

popt, pcov = curve_fit(ldf_model, t_data, c_avg_data, p0=[k_ldf_theory])
k_ldf_fitted = float(popt[0])

ss_res = np.sum((c_avg_data - ldf_model(t_data, k_ldf_fitted))**2)
ss_tot = np.sum((c_avg_data - np.mean(c_avg_data))**2)
r2_score = float(1.0 - (ss_res / ss_tot))
deviation = float(abs(k_ldf_fitted - k_ldf_theory) / k_ldf_theory * 100.0)

print("---------------------------------------------")
print("Theoretical k_LDF (Glueckauf):", round(k_ldf_theory, 4))
print("Fitted k_LDF from COMSOL:     ", round(k_ldf_fitted, 4))
print("Deviation percentage:         ", round(deviation, 2))
print("R-squared:                    ", round(r2_score, 5))
print("---------------------------------------------")

plt.figure(figsize=(7, 5), dpi=300)
plt.plot(t_data, c_avg_data, "k.", label="COMSOL", markersize=4, alpha=0.7)
plt.plot(t_data, ldf_model(t_data, k_ldf_fitted), "r-", linewidth=2, label="LDF Fit")
plt.plot(t_data, ldf_model(t_data, k_ldf_theory), "b--", linewidth=1.5, label="Glueckauf Theory")

plt.title("COMSOL Micro-Scale Diffusion vs LDF Model")
plt.xlabel("Time (s)")
plt.ylabel("Average Concentration")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_PLOT)
print("Plot saved successfully.")
