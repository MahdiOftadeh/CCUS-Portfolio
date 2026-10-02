"""Conservative, well-mixed two-stage VPSA accounting model.

This is a corrected, runnable replacement for the phase-4 two-stage scaffold,
not a spatially resolved/validated design model. One well-mixed gas volume and
one adsorbed inventory per component are used per bed. Unlike the legacy
bookkeeping, each stream is integrated from component inventory balances;
there are no guessed outlet fractions or clipped mole fractions.
Run: python src/run_corrected_dual_stage_vpsa.py [--output DIR]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

R=8.314462; T=298.15; P_HI=1.5e5; P_LO=0.05e5
AREA=np.pi*(0.05/2)**2; VOID=AREA*1.0*0.37; MASS=1130*(1-0.37)*AREA
# Extended Langmuir and LDF coefficients, retaining the source-project values.
QMAX=np.array([4.85,2.10]); B=np.array([3.25e-5,0.12e-5]); K=np.array([0.18,0.25])
CYCLE=210.0; T_ADS=60.0; T_VAC=65.0; T_PRESS=20.0
N_HI=P_HI*VOID/(R*T); N_LO=P_LO*VOID/(R*T)
# Baseline stage-2 CO2 rinse rate uses the related 1-D solver ratio 0.04/0.08
# (rinse/feed superficial velocity). This reduced model treats it as a pure-CO2
# co-feed during stage-2 adsorption; it is not a separately resolved rinse step.
RINSE_BASELINE_FRACTION_STAGE2=0.5
RINSE_SWEEP_RATIOS=np.linspace(0.75,0.95,9)

class InvalidState(RuntimeError): pass

def check_y(y, tol=1e-9):
    y=np.asarray(y,float)
    if not np.all(np.isfinite(y)) or np.any(y < -tol) or np.any(y > 1+tol) or abs(float(y.sum())-1)>tol:
        raise InvalidState(f"invalid gas mole fractions: {y}")

def qstar(y, pressure):
    check_y(y)
    partial=np.asarray(y)*pressure
    return QMAX*B*partial/(1+np.dot(B,partial))

def integrate(fun, span, state):
    sol=solve_ivp(fun,span,state,method='LSODA',rtol=2e-8,atol=1e-11,max_step=1.0)
    if not sol.success or not np.all(np.isfinite(sol.y[:,-1])):
        raise RuntimeError('ODE integration failed: '+sol.message)
    return sol.y[:,-1]

def one_cycle(y0,q0,yfeed, flow, rinse_flow=0.0):
    """Cycle with component order [CO2,N2]. All stream amounts are mol/cycle."""
    check_y(y0); check_y(yfeed)
    if not np.isfinite(rinse_flow) or rinse_flow<0: raise ValueError('rinse_flow must be finite and nonnegative')
    q0=np.asarray(q0,float)
    if q0.shape!=(2,) or np.any(q0<0): raise InvalidState('invalid starting adsorbed inventory')
    # Pressurize the gas void from low to high using feed of known composition.
    press_in=(N_HI-N_LO)*yfeed
    y=yfeed.copy(); q=q0.copy(); feed=press_in.copy(); rinse_input=np.zeros(2); waste=np.zeros(2); prod=np.zeros(2)
    # Feed adsorption at constant pressure. Outlet flow is derived from the
    # total-gas balance: Fout=Fin-M*sum(dq/dt), never assigned by a heuristic.
    # State vector: y_CO2, q_CO2, q_N2, cumulative outlet CO2, outlet N2.
    F=flow
    def ads_rhs(t,x):
        yy=np.array([x[0],1-x[0]])
        if yy[0] < -1e-8 or yy[0] > 1+1e-8: raise InvalidState(f'adsorption y={yy[0]}')
        qq=x[1:3]; dq=K*(qstar(yy,P_HI)-qq)
        rin=rinse_flow
        fin=F+rin
        fout=fin-MASS*dq.sum()
        if fout < -1e-10: raise InvalidState(f'negative derived adsorption outlet {fout:g} mol/s; increase feed flow')
        fout=max(fout,0.0)
        dy=(F*yfeed+rin*np.array([1.0,0.0])-fout*yy-MASS*dq)/N_HI
        return [dy[0],dq[0],dq[1],fout*yy[0],fout*yy[1]]
    x=integrate(ads_rhs,(0,T_ADS),[y[0],q[0],q[1],0,0])
    y=np.array([x[0],1-x[0]]); q=x[1:3]; waste += x[3:5]
    feed += F*T_ADS*yfeed
    rinse_input[0] += rinse_flow*T_ADS
    # Instantaneous pressure letdown: removed void gas is an accounted product.
    prod += (N_HI-N_LO)*y
    # Evacuation at constant low pressure: Fout=-M*sum(dq/dt), and each
    # component outlet is from the mixed-gas composition. This conserves both.
    def vac_rhs(t,x):
        yy=np.array([x[0],1-x[0]])
        if yy[0] < -1e-8 or yy[0] > 1+1e-8: raise InvalidState(f'vacuum y={yy[0]}')
        qq=x[1:3]; dq=K*(qstar(yy,P_LO)-qq)
        fout=-MASS*dq.sum()
        if fout < -1e-9: raise InvalidState(f'vacuum outlet negative ({fout:g}); non-desorbing operating state')
        fout=max(fout,0.)
        dy=(-fout*yy-MASS*dq)/N_LO
        return [dy[0],dq[0],dq[1],fout*yy[0],fout*yy[1]]
    x=integrate(vac_rhs,(0,T_VAC),[y[0],q[0],q[1],0,0])
    y=np.array([x[0],1-x[0]]); q=x[1:3]; prod += x[3:5]
    check_y(y)
    return {'y':y,'q':q,'feed':feed,'rinse_input':rinse_input,'waste':waste,'product':prod}

def converge(feed_y, flow, rinse_flow=0.0, max_cycles=250, tol=2e-7):
    y=np.array([0.5,0.5]); q=qstar(y,P_LO).copy()
    history=[]; last=None
    for cycle in range(1,max_cycles+1):
        r=one_cycle(y,q,feed_y,flow,rinse_flow)
        # Cyclic state includes gas composition and adsorbed amounts.
        err=max(float(np.max(np.abs(r['y']-y))),float(np.max(np.abs(r['q']-q))))
        history.append(err); y,q=r['y'],r['q']
        if err<tol:
            # Recompute recorded streams from the converged initial state.
            last=one_cycle(y,q,feed_y,flow,rinse_flow)
            # Numerical fixed-point residual and balance evaluated on repeated cycle.
            residual=max(float(np.max(np.abs(last['y']-y))),float(np.max(np.abs(last['q']-q))))
            if residual<tol*2: return last,cycle,residual
    raise RuntimeError(f'cycle did not converge in {max_cycles}; last state error={history[-1]:.3g}')

def audit(r, feed_y, atol=2e-8):
    # Component cycle balance: feed - waste - product equals inventory change.
    check_y(r['y'])
    for name,v in [('feed',r['feed']),('waste',r['waste']),('product',r['product'])]:
        if np.any(~np.isfinite(v)) or np.any(v < -atol): raise InvalidState(f'invalid {name} stream {v}')
    rin=r.get('rinse_input',np.zeros(2))
    if np.any(~np.isfinite(rin)) or np.any(rin < -atol): raise InvalidState(f'invalid rinse input {rin}')
    residual=r['feed']+rin-r['waste']-r['product']
    # Explicit derived balance accounting, including equal start/end inventories
    # at CSS. Component balance should close to integration tolerance.
    scale=max(float(np.max(r['feed'])),1e-8)
    if np.max(np.abs(residual))>atol*scale:
        raise InvalidState(f'component conservation failed: residual={residual}, feed={r["feed"]}')
    return residual

def run(feed_co2=0.15, recycle_fraction=0.99, rinse_ratio_stage2=0.0):
    if not 0<feed_co2<1: raise ValueError('feed CO2 fraction must be strictly between 0 and 1')
    if not 0<=recycle_fraction<=1: raise ValueError('recycle_fraction must be in [0, 1]')
    if not np.isfinite(rinse_ratio_stage2) or rinse_ratio_stage2<0: raise ValueError('rinse_ratio_stage2 must be finite and nonnegative')
    fresh_y=np.array([feed_co2,1-feed_co2])
    # Preserve the source-project fresh-feed superficial velocity. Recycle is
    # an additional feed, so Stage 1's actual inlet flow is fresh + recycle.
    fresh_flow=0.08*AREA*P_HI/(R*T)
    fresh_cycle=fresh_flow*(T_PRESS+T_ADS)*fresh_y+(N_HI-N_LO)*fresh_y
    recycle=np.zeros(2)
    last=None
    for outer in range(1,101):
        combined=fresh_cycle+recycle
        total_flow=combined.sum()/(T_PRESS+T_ADS+(N_HI-N_LO)/fresh_flow)
        # Equivalent cycle-average feed formulation while retaining the source
        # superficial velocity for adsorption; pressurization uses this same
        # combined-feed composition. Keep stream accounting explicit per cycle.
        feed1=combined/combined.sum()
        F1=(combined.sum()-(N_HI-N_LO))/T_ADS
        if F1<=0: raise InvalidState('combined stage-1 feed flow must be positive')
        s1,n1,e1=converge(feed1,F1)
        b1=audit(s1,feed1)
        if s1['product'].sum()<=0: raise InvalidState('stage 1 produced no product')
        feed2=s1['product']/s1['product'].sum()
        F2=s1['product'].sum()/CYCLE
        rinse_flow_stage2=rinse_ratio_stage2*RINSE_BASELINE_FRACTION_STAGE2*F2
        s2,n2,e2=converge(feed2,F2,rinse_flow_stage2)
        b2=audit(s2,feed2)
        updated=recycle_fraction*s2['waste']
        err=float(np.max(np.abs(updated-recycle)))
        scale=max(float(np.max(fresh_cycle)),1e-12)
        last=(s1,n1,e1,b1,feed1,F1,s2,n2,e2,b2,feed2,F2,rinse_flow_stage2,updated,err)
        if err < 2e-7*scale:
            break
        # Damped Picard iteration stabilizes high recycle fractions.
        recycle=0.5*recycle+0.5*updated
    else:
        raise RuntimeError(f'recycle loop failed to converge; final stream error={err:g}')
    (s1,n1,e1,b1,feed1,F1,s2,n2,e2,b2,feed2,F2,rinse_flow_stage2,recycle,outer_err)=last
    def metrics(s):
        p=s['product']; ft=s['feed']; rin=s.get('rinse_input',np.zeros(2)); net_co2=p[0]-rin[0]
        return {'product_co2_mol_cycle':float(p[0]),'product_n2_mol_cycle':float(p[1]),
                'rinse_co2_input_mol_cycle':float(rin[0]),'net_product_co2_mol_cycle':float(net_co2),
                'product_total_mol_cycle':float(p.sum()),
                'product_co2_purity_pct':float(100*p[0]/p.sum()) if p.sum()>0 else None,
                'co2_recovery_pct':float(100*net_co2/ft[0]) if ft[0]>0 else None,
                'feed_component_mol_cycle':ft.tolist(),'waste_component_mol_cycle':s['waste'].tolist(),
                'component_balance_residual_mol':(ft+rin-s['waste']-p).tolist(),
                'final_gas_y_co2':float(s['y'][0]),'final_adsorbed_mol_per_kg':[float(v) for v in s['q']]}
    return {'model':'two-stage well-mixed conservative cyclic model (reduced-order; not spatially resolved)',
            'assumptions':['isothermal, well-mixed gas in each bed','stage-2 CO2 rinse is approximated as a pure-CO2 co-feed during adsorption at a rate scaled from the baseline stage-2 feed','stage-2 waste is recycled to stage 1 at the specified recycle fraction','stage-1 inlet is fresh feed plus recycled stage-2 waste, with mixed component fractions and consistent total flow','instantaneous gas-void letdown; vacuum outlet held at low-pressure void inventory'],
            'cycle_time_s':CYCLE,
            'stage1':{**metrics(s1),'feed_y_co2':float(feed1[0]),'feed_molar_flow_mol_s':float(F1),'fresh_feed_component_mol_cycle':fresh_cycle.tolist(),'recycle_feed_component_mol_cycle':recycle.tolist(),'cycles_to_css':n1,'css_residual':e1},
            'stage2':{**metrics(s2),'feed_y_co2':float(feed2[0]),'feed_molar_flow_mol_s':float(F2),'rinse_ratio_stage2':float(rinse_ratio_stage2),'rinse_flow_mol_s':float(rinse_flow_stage2),'rinse_baseline_fraction_of_feed':RINSE_BASELINE_FRACTION_STAGE2,'cycles_to_css':n2,'css_residual':e2},
            'recycle_fraction':float(recycle_fraction),'recycle_outer_iterations':outer,
            'recycle_outer_residual_mol_cycle':float(outer_err),
            'validation':{'all_mole_fractions_in_unit_interval':True,'component_conservation_each_stage':True,
                          'stage1_combined_feed_components_exact':bool(np.allclose(feed1*sum(fresh_cycle+recycle),fresh_cycle+recycle,rtol=0,atol=1e-14)),
                          'stage_interface_component_exact':bool(np.allclose(feed2,s1['product']/s1['product'].sum(),rtol=0,atol=1e-14)),
                          'cycle_converged_each_stage':True,'recycle_loop_converged':bool(outer_err<2e-7*max(float(np.max(fresh_cycle)),1e-12))}}

def run_rinse_sweep_stage2(recycle_fraction=0.99, ratios=RINSE_SWEEP_RATIOS, feed_co2=0.15, output=None):
    """Run stage-2 rinse-ratio sweep and save stage-2 purity/recovery CSV."""
    import csv
    output=Path(output) if output else Path(__file__).resolve().parents[1]/'results'/'rinse_ratio_sweep_stage2_with_recycle099.csv'
    output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for ratio in ratios:
        result=run(feed_co2=feed_co2,recycle_fraction=recycle_fraction,rinse_ratio_stage2=float(ratio))
        st=result['stage2']
        rows.append({'rinse_ratio_stage2':float(ratio),'recycle_fraction':float(recycle_fraction),
                     'rinse_flow_mol_s':st['rinse_flow_mol_s'],
                     'purity_pct':st['product_co2_purity_pct'],'recovery_pct':st['co2_recovery_pct'],
                     'net_product_co2_mol_cycle':st['net_product_co2_mol_cycle'],
                     'cycles_to_css':st['cycles_to_css'],'css_residual':st['css_residual']})
        print(f"rinse ratio {ratio:.3f}: purity={st['product_co2_purity_pct']:.3f}%, recovery={st['co2_recovery_pct']:.3f}%")
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return output,rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['run','sweep_rinse_stage2'],default='run')
    ap.add_argument('--feed-co2',type=float,default=.15)
    ap.add_argument('--recycle',type=float,default=.99)
    ap.add_argument('--rinse-ratio-stage2',type=float,default=0.0)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    a=ap.parse_args()
    if a.mode=='sweep_rinse_stage2':
        path,rows=run_rinse_sweep_stage2(a.recycle,feed_co2=a.feed_co2,
                                         output=a.output/'rinse_ratio_sweep_stage2_with_recycle099.csv')
        print(json.dumps({'sweep_csv':str(path),'recycle_fraction':a.recycle,'rows':len(rows)},indent=2))
        return
    out=run(a.feed_co2,recycle_fraction=a.recycle,rinse_ratio_stage2=a.rinse_ratio_stage2)
    a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/'corrected_dual_stage_summary.json'; path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary_json':str(path),'stage1':out['stage1'],'stage2':out['stage2'],'validation':out['validation']},indent=2))
if __name__=='__main__': main()
