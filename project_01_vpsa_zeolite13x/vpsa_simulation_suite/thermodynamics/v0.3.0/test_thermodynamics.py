#!/usr/bin/env python3
"""Self-contained engineering/test showcase for Thermodynamics_module_0_2.

Place this file beside the target module (or pass its path as argv[1]). The
engineering showcase is deliberately best-effort: a demo/plot failure is
reported, while the process exit status is determined by the core unit tests.
"""
from __future__ import annotations
import importlib.util
import math
import os
import sys
import unittest
from pathlib import Path

PREFERRED = "thermodynamics.py"
FALLBACK = "thermodynamics.py"
REQUIRED = (
    "TothIsotherm", "DualSiteLangmuirIsotherm", "TemperatureDependentToth",
    "TemperatureDependentDSL", "BinaryIAST", "CAVENATI_13X_TOTH",
    "KRISHNAMURTHY_13X_DSL", "ZEOLITE_13X_PHYSICAL",
)

def load_target() -> tuple[object, Path]:
    candidates = [Path(sys.argv[1])] if len(sys.argv) > 1 else []
    here = Path(__file__).resolve().parent
    candidates.extend((here / PREFERRED, here / FALLBACK))
    # Also permit running from a different working directory.
    candidates.extend((Path.cwd() / PREFERRED, Path.cwd() / FALLBACK))
    seen = set()
    for path in candidates:
        path = path.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            spec = importlib.util.spec_from_file_location("thermodynamics_target", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            missing = [name for name in REQUIRED if not hasattr(mod, name)]
            if missing:
                raise ImportError(f"{path} is missing required API: {', '.join(missing)}")
            # Bind the requested public API explicitly in this script's namespace.
            globals().update({name: getattr(mod, name) for name in REQUIRED})
            return mod, path
    raise FileNotFoundError("Could not find target module: " + " or ".join((PREFERRED, FALLBACK)))

def line(char="=", width=92):
    print(char * width)

def table(title, headers, rows, formats=None):
    print("\n" + title)
    if not rows:
        print("  (no rows)")
        return
    rendered = []
    for row in rows:
        vals=[]
        for i, value in enumerate(row):
            if formats and i < len(formats) and callable(formats[i]):
                try: value = formats[i](value)
                except Exception: pass
            vals.append(str(value))
        rendered.append(vals)
    widths=[max(len(str(h)), *(len(r[i]) for r in rendered)) for i,h in enumerate(headers)]
    fmt=" | ".join("{{:<{}}}".format(w) for w in widths)
    print(fmt.format(*headers)); print("-+-".join("-"*w for w in widths))
    for row in rendered: print(fmt.format(*row))

def run_tests(mod):
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for cls_name in ("TestThermodynamicEngine", "TestPhase1ReferenceDatabase"):
        cls = getattr(mod, cls_name, None)
        if cls is None:
            raise AttributeError(f"Target module does not define {cls_name}")
        suite.addTests(loader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result

def showcase(mod):
    T = 298.15
    print("\nENGINEERING SHOWCASE — Zeolite 13X / CO2-N2")
    line()
    # Literature-backed Cavenati mass-based models, pressure in bar / loading mol/kg.
    systems = mod.get_cavenati_toth_system()
    co2 = systems["CO2"]["isotherm"]
    n2 = systems["N2"]["isotherm"]
    pressures = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    rows=[]
    for pressure in pressures:
        rows.append((pressure, co2.loading(pressure,T), n2.loading(pressure,T),
                     co2.dq_dP(pressure,T), n2.dq_dP(pressure,T)))
    table("Pure-component equilibrium at 298.15 K (q: mol/kg; dq/dP: mol/kg/bar)",
          ["P [bar]", "CO2 q", "N2 q", "CO2 dq/dP", "N2 dq/dP"], rows,
          [lambda x:f"{x:.3f}", lambda x:f"{x:.6f}", lambda x:f"{x:.6f}", lambda x:f"{x:.6f}", lambda x:f"{x:.6f}"])

    # Binary IAST at specified flue gas.
    iast = mod.BinaryIAST(co2, n2)
    Pmix, yco2 = 1.0, 0.15
    qco2, qn2, xco2, psi = iast.solve(P=Pmix, y1=yco2, T=T)
    selectivity = (xco2/(1.0-xco2))/(yco2/(1.0-yco2)) if 0 < xco2 < 1 else float("nan")
    table("Binary IAST: 15% CO2 / 85% N2, 1.0 bar, 298.15 K",
          ["qCO2 [mol/kg]", "qN2 [mol/kg]", "xCO2 adsorbed", "psi [mol/kg]", "selectivity"],
          [(qco2,qn2,xco2,psi,selectivity)], [lambda x:f"{x:.7g}"]*5)

    # Isosteric heat from source Cavenati heat-loading relation (q in mol/kg).
    heat_rows=[]
    heat_curves={}
    for species, iso in (("CO2",co2), ("N2",n2)):
        qm=float(systems[species]["params"]["qm"])
        qs=[qm*f for f in (0.0,0.1,0.2,0.4,0.6,0.8,0.95)]
        heats=[iso.at_temperature(T).get_isosteric_heat(q) for q in qs]
        heat_curves[species]=(qs,heats)
        heat_rows.extend((species,q,h) for q,h in zip(qs,heats))
    table("Loading-dependent isosteric heat Qst (source correlation; kJ/mol)",
          ["Species","q [mol/kg]","Qst [kJ/mol]"], heat_rows,
          [str,lambda x:f"{x:.4f}",lambda x:f"{x/1000:.4f}"])

    # Spreading-pressure inversion round-trip for both pure gases.
    round_rows=[]
    for species, iso in (("CO2",co2),("N2",n2)):
        for p in (0.1,1.0,5.0,10.0):
            ps=iso.spreading_pressure(p,T)
            recovered=iso.invert_spreading_pressure(ps,T,P_guess=p)
            err=abs(recovered-p)/max(p,1e-15)
            round_rows.append((species,p,ps,recovered,err))
    table("Spreading pressure and P -> psi -> inverse(psi) -> P verification",
          ["Species","P [bar]","psi [mol/kg]","P recovered [bar]","relative error"],round_rows,
          [str,lambda x:f"{x:.4f}",lambda x:f"{x:.7g}",lambda x:f"{x:.8f}",lambda x:f"{x:.2e}"])

    # Full six-entry numerical thermodynamic Jacobian.
    jac = iast.thermodynamic_jacobian(P=Pmix,y1=yco2,T=T)
    jac_order=("dq1_dP","dq1_dy","dq1_dT","dq2_dP","dq2_dy","dq2_dT")
    units={"dq1_dP":"mol kg^-1 bar^-1","dq1_dy":"mol kg^-1","dq1_dT":"mol kg^-1 K^-1"}
    jac_rows=[]
    for key in jac_order:
        if key not in jac: raise KeyError(f"Missing Jacobian element {key}")
        val=float(jac[key])
        if not math.isfinite(val): raise ValueError(f"Non-finite Jacobian element {key}={val}")
        unit = ("mol kg^-1 bar^-1" if key.endswith("_dP") else
                "mol kg^-1" if key.endswith("_dy") else "mol kg^-1 K^-1")
        jac_rows.append((key, val, unit))
    table("Complete IAST thermodynamic Jacobian at P=1 bar, yCO2=0.15, T=298.15 K",
          ["Derivative","Value","Units"],jac_rows,[str,lambda x:f"{x:.8g}",str])

    # Add reference database / physical parameter snapshot.
    phys=mod.ZEOLITE_13X_PHYSICAL
    print("\nReference data snapshot:")
    print(f"  Cavenati species: {', '.join(systems.keys())}; T0={mod.CAVENATI_13X_TOTH['T0']:.2f} K")
    print(f"  Krishnamurthy DSL species: {', '.join(mod.KRISHNAMURTHY_13X_DSL.keys())}")
    print("  Zeolite 13X: " + ", ".join(f"{k}={v:g}" for k,v in phys.items()))

    # Plot is optional; numerical demonstration above remains useful without matplotlib.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, axes=plt.subplots(1,3,figsize=(15,4.6))
        pgrid=np.geomspace(0.01,10,160)
        axes[0].plot(pgrid,[co2.loading(p,T) for p in pgrid],label="CO2")
        axes[0].plot(pgrid,[n2.loading(p,T) for p in pgrid],label="N2")
        axes[0].set(xlabel="Pressure [bar]",ylabel="Loading [mol/kg]",title="Pure isotherms")
        axes[0].set_xscale("log"); axes[0].grid(True,alpha=.3); axes[0].legend()
        for sp,(qs,hs) in heat_curves.items(): axes[1].plot(qs,np.array(hs)/1000,"o-",label=sp)
        axes[1].set(xlabel="Loading [mol/kg]",ylabel="Qst [kJ/mol]",title="Isosteric heat")
        axes[1].grid(True,alpha=.3); axes[1].legend()
        axes[2].bar(["CO2","N2"],[qco2,qn2],color=["#2878b5","#8ab6d6"])
        axes[2].set(ylabel="Loading [mol/kg]",title="IAST mixture loading")
        axes[2].grid(True,axis="y",alpha=.3)
        fig.suptitle("Thermodynamics Showcase — Zeolite 13X, 298.15 K")
        fig.tight_layout()
        out=Path(__file__).resolve().parent/"thermodynamics_showcase_v3.png"
        fig.savefig(out,dpi=160,bbox_inches="tight"); plt.close(fig)
        print(f"\nPlot saved: {out}")
    except Exception as exc:
        print(f"\n[SHOWCASE NOTE] Plot generation skipped: {type(exc).__name__}: {exc}")

def main() -> int:
    line()
    print("THERMODYNAMICS MODULE — TEST & ENGINEERING SHOWCASE v3")
    try:
        mod,path=load_target()
        print(f"Target module: {path}")
    except Exception as exc:
        print(f"[FATAL] Cannot load required target module: {type(exc).__name__}: {exc}")
        return 2
    line()
    try:
        result=run_tests(mod)
    except Exception as exc:
        print(f"[FATAL] Could not run required unittest suites: {type(exc).__name__}: {exc}")
        return 2
    print("\nCore unit-test result: " + ("PASS" if result.wasSuccessful() else "FAIL"))
    if not result.wasSuccessful():
        failed = [case.id() for case, _ in result.failures]
        errored = [case.id() for case, _ in result.errors]
        if failed: print("  Assertion failures: " + "; ".join(failed))
        if errored: print("  Test errors: " + "; ".join(errored))
    try:
        showcase(mod)
    except Exception as exc:
        print(f"\n[SHOWCASE NOTE] Engineering demo encountered an issue: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
    # Explicit contract: showcase issues do not mask passing core tests.
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    raise SystemExit(main())
