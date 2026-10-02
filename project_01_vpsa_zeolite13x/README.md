# VPSA Zeolite 13X — Research Artifacts & Simulation Pipeline

A clean, GitHub-ready repository bundle containing simulation source code, convergence data, parametric studies, results, and dashboards for dual-stage Vacuum Pressure Swing Adsorption (VPSA) carbon capture using Zeolite 13X.

---

## 📌 Persian Summary / خلاصه‌ی فارسی
این مخزن شامل تمام کدهای شبیه‌سازی، داده‌های همگرایی به حالت پایا (CSS)، مطالعات پارامتریک و شکل‌های خروجی فرآیند جذب نوسانی فشار در خلاء (VPSA) دو مرحله‌ای با جاذب زئولیت 13X برای جداسازی و جذب کربن ($CO_2$) است. تمام فایل‌های مراحل مختلف (Phase 3، Phase 4 و بهینه‌سازی تصحیح‌شده Robust) به‌همراه فایل‌های اصلی و تصاویر به‌صورت ساختاریافته در این پکیج قرار داده شده‌اند تا امکان بررسی و بازتولید محاسبات فراهم باشد.

---

## 🎯 Reported Target Operating Point (DOE / NETL Benchmarks)

According to the supplied optimization summary and target report, the optimal dual-stage configuration yields the following reported values:

| Parameter / Metric | Reported Value | Notes |
| :--- | :---: | :--- |
| **Recycle Fraction** ($R_f$) | `0.75` | Split fraction of tail gas recycled to Stage 1 |
| **Rinse Ratio** ($R_{rinse}$) | `0.833` | Ratio of rinse flow to feed flow ($\sim 11.75 / 14.1$) |
| **$CO_2$ Product Purity** | `95.13%` (95.125%) | Exceeds DOE/NETL min target (≥ 95.0%) |
| **Overall $CO_2$ Recovery** | `91.65%` (91.649%) | Exceeds DOE/NETL min target (≥ 90.0%) |
| **Stage 2 Recovery** | `92.00%` | Stage 2 capture efficiency |
| **Product $CO_2$ Molar Flow** | `13.75 mol/min` (13.747) | At feed total 100 mol/min (15% $CO_2$) |

> **⚠️ Important Notice / Verification:**  
> These metrics are **reported values** from model output summaries. Readers and researchers must verify them against [`vpsa_corrected/results/final_optimized_target.json`](vpsa_corrected/results/final_optimized_target.json) and related scripts before using them for design or benchmarking.

---

## ⚙️ Model Scope, Idealizations & Industrial Limitations

Per the supplied technical summary, the underlying simulation models feature certain fundamental assumptions:
1. **Mechanical & Compressor Efficiency:** Isentropic and mechanical efficiencies of blowers, vacuum pumps, and compressors are **not included** in the ideal thermodynamic equations.
2. **Column Pressure Drop:** Momentum balance assumes negligible axial pressure drop (Ergun equation not coupled in simplified solvers).
3. **Heat Transfer & Mass Transfer:** Some subroutines assume isothermal or simplified LDF (Linear Driving Force) kinetics.
4. **Industrial Scale-up:** The reported performance represents idealized bed dynamics; real-world industrial units will require safety margins, valve switching dynamics, parasitic energy calculations, and dynamic pressure drop considerations.

---

## 📁 Repository Layout & Provenance Map

```text
.
├── README.md                                  # Repository overview & documentation
├── .gitignore                                 # Standard Python gitignore
├── test_vpsa_fast.py                          # Fast validation test script
├── assets/                                    # Uploaded figures and screenshots
│   ├── Screenshot 2026-09-26 184153.png
│   └── breakthrough_curve.png
├── archives/                                  # Intact legacy archives
│   └── project_01_vpsa_zeolite13x.zip
├── project_01_vpsa_zeolite13x/                # Base simulation tree (Phase 1-4)
│   ├── src/                                   # Base simulation scripts
│   ├── data/                                  # CSS convergence & axial profiles
│   ├── results/                               # Base stage figures & summary metrics
│   ├── phase3/                                # Phase 3 cycle code and parameter sweep
│   └── phase4/                                # Phase 4 CO2 rinse & dual-stage studies
├── project_01_vpsa_zeolite13x_phase3/         # Retained legacy Phase 3 snapshot (provenance)
│   ├── data/
│   ├── figures/
│   ├── metrics/
│   └── run_vpsa_cycle.py
└── vpsa_corrected/                            # Corrected & robust optimization pipeline
    ├── src/                                   # Robust dual-stage & corrected VPSA solvers
    ├── results/                               # Pareto sweep, rinse study, final target JSON
    └── run_vpsa_optimization.py               # Main optimization entry point
```

*Note on Provenance:* Duplicate or legacy files (such as `project_01_vpsa_zeolite13x_phase3/` and `archives/project_01_vpsa_zeolite13x.zip`) are intentionally preserved to ensure full reproducibility and audit trail across project development stages.

---

## 🧪 Reproduction & Execution Guidelines

> **Notice:** No `requirements.txt` was included among the raw files. Furthermore, **no scripts have been executed or validated as part of this packaging process**.

To set up an environment and inspect the models:

1. **Create an isolated Python environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scriptsctivate
   pip install numpy scipy matplotlib pandas
   ```

2. **Inspect and run the quick test:**
   ```bash
   python test_vpsa_fast.py
   ```

3. **Run robust dual-stage optimization / sweep:**
   ```bash
   python vpsa_corrected/run_vpsa_optimization.py
   ```

---

## 🚀 Git Commands for Publishing to GitHub

To push this repository to your GitHub account:

```bash
# 1. Initialize git repository
git init

# 2. Add all files
git add README.md .gitignore assets archives project_01_vpsa_zeolite13x project_01_vpsa_zeolite13x_phase3 vpsa_corrected test_vpsa_fast.py

# 3. Commit files
git commit -m "feat: Initial commit of VPSA Zeolite 13X research repository"

# 4. Set main branch and remote
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git

# 5. Push to GitHub
git push -u origin main
```
