#!/usr/bin/env python3
"""Conservative 1-D isothermal VPSA cycle model; finite-volume MOL + BDF.
Components are CO2/N2. Total flux is reconstructed from the exact cellwise
overall balance at prescribed spatially uniform pressure. q is mol/kg adsorbent.
"""
import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

OUT=os.path.dirname(os.path.abspath(__file__))
R=8.314462618; T=298.15; L=1.0; D=.05; A=np.pi*D**2/4; eps=.37; rho=1130.
Nz=16; dz=L/Nz; z=(np.arange(Nz)+.5)*dz; Mbed=rho*(1-eps)*A*L
qmax=np.array([4.85,2.10]); b=np.array([3.25e-5,.12e-5]); k=np.array([.18,.25])
Plo,Pint,Peq,Phi=0.05e5,.4e5,.95e5,1.5e5
times={'press':20.,'ads':60.,'rinse':30.,'pe':15.,'blow':20.,'evac':65.}
cycle_time=sum(times.values()); yf=np.array([.15,.85]);

def qstar(y,P):
    p=np.maximum(y,0)*P; den=1+np.dot(b,p)
    return qmax*b*p/den

def stage(state, name, duration, p0,p1, mode, inlet_y=None, inlet_u=0., flow_dir=1):
    # mode=inlet (fixed-P feed/rinse), press (inlet with shut product end),
    # pe (shut feed end, outlet at z=L), vent (countercurrent outlet at z=0)
    def pressure(t): return p0+(p1-p0)*t/duration
    def rhs(t, X):
        s=X[:4*Nz].reshape(4,Nz); y=np.clip(s[:2].T,0,1); q=s[2:].T
        P=pressure(t); c=P/(R*T); dc=(p1-p0)/duration/(R*T)
        dq=(qstar(y,P)-q)*k
        solid=(1-eps)*rho*dq.sum(axis=1)
        # Total molar superficial flux F=u*c at N+1 faces.
        F=np.zeros(Nz+1)
        if mode=='inlet':
            F[0]=inlet_u*c
            for j in range(Nz): F[j+1]=F[j]-dz*(eps*dc+solid[j])
        elif mode=='press':
            F[-1]=0.
            for j in range(Nz-1,-1,-1): F[j]=F[j+1]+dz*(eps*dc+solid[j])
        elif mode=='pe':
            F[0]=0.
            for j in range(Nz): F[j+1]=F[j]-dz*(eps*dc+solid[j])
        elif mode=='vent':
            F[-1]=0.
            for j in range(Nz-1,-1,-1): F[j]=F[j+1]+dz*(eps*dc+solid[j])
        # component fractions on faces via first-order upwind, with specified feed boundary.
        yfcell=y
        yface=np.empty((Nz+1,2))
        for f in range(Nz+1):
            if f==0:
                yface[f]=inlet_y if (mode in ('inlet','press') and F[f]>=0 and inlet_y is not None) else yfcell[0]
            elif f==Nz: yface[f]=yfcell[-1]
            else: yface[f]=yfcell[f-1] if F[f]>=0 else yfcell[f]
        # Feed/exit integrated signed component molar rates (mol/s), F positive z.
        Fi=F[:,None]*yface
        dflux=(Fi[1:]-Fi[:-1])/dz
        dydt=(-dflux+y* (np.diff(F)/dz+solid)[:,None]- (1-eps)*rho*dq)/(eps*c)
        dstate=np.vstack([dydt.T,dq.T]).reshape(-1)
        # accumulator [feed CO2,N2; heavy CO2,N2; rinse CO2; waste CO2,N2; vacuum work]
        feed=np.zeros(2); heavy=np.zeros(2); waste=np.zeros(2); rin=0.; work=0.
        if name in ('press','ads'):
            # feed entering at left, positive flux
            if F[0]>0: feed=F[0]*yface[0]
        if name=='rinse' and F[0]>0: rin=F[0]*yface[0,0]
        if name in ('blow','evac'):
            if F[0]<0: heavy=-Fi[0]
            # idealized vacuum compression work based on extracted total flow, integrated
            if name=='evac' and F[0]<0: work=(-F[0])*R*T*np.log(max(Phi/P,1.0))/.75
        if name in ('ads','rinse') and F[-1]>0: waste=Fi[-1]
        accum=np.r_[feed,heavy,rin,waste,work]
        return np.r_[dstate,accum]
    X0=np.r_[state, np.zeros(8)]
    sol=solve_ivp(rhs,(0,duration),X0,method='BDF',rtol=2e-6,atol=1e-9,max_step=max(duration/30,.1))
    if not sol.success: raise RuntimeError(f'{name}: {sol.message}')
    return sol.y[:4*Nz,-1],sol.y[4*Nz:,-1],sol

def cycle(s):
    totals=np.zeros(9); profiles={}
    stages=[('press',times['press'],Plo,Phi,'press',yf,0.),('ads',times['ads'],Phi,Phi,'inlet',yf,.08),('rinse',times['rinse'],Phi,Phi,'inlet',np.array([1.,0.]),.04),('pe',times['pe'],Phi,Peq,'pe',None,0.),('blow',times['blow'],Peq,Pint,'vent',None,0.),('evac',times['evac'],Pint,Plo,'vent',None,0.)]
    for name,d,p0,p1,mode,iny,u in stages:
        s,a,sol=stage(s,name,d,p0,p1,mode,iny,u); totals+=a
        v=sol.y[:4*Nz,-1].reshape(4,Nz); profiles[name]=v[0].copy()
    return s,totals,profiles

# Initial loading equilibrium at low pressure and feed composition
Pinit=Plo; y0=np.tile(yf[:,None],(1,Nz)); q0=np.tile(qstar(yf,Pinit)[:,None],(1,Nz))
state=np.vstack([y0,q0]).reshape(-1)
records=[]; start=time.time(); css=False
for cyc in range(1,41):
    old=state.copy(); state,acc,prof=cycle(state)
    # convergence of cyclic state using normalized infinity norm
    err=float(np.max(np.abs(state-old)))
    feed=acc[:2]; heavy=acc[2:4]; rinse=acc[4]; waste=acc[5:7]; work=acc[7]
    net=max(heavy[0]-rinse,0.)
    purity=100*heavy[0]/heavy.sum() if heavy.sum()>0 else 0.
    recovery=100*net/feed[0] if feed[0]>0 else 0.
    mass_t=net*.04401/1000
    sec=(work/3.6e6)/mass_t if mass_t>0 else 0.
    records.append(dict(cycle=cyc,css_error=err,feed_CO2_mol=feed[0],feed_N2_mol=feed[1],heavy_CO2_gross_mol=heavy[0],heavy_N2_mol=heavy[1],rinse_CO2_mol=rinse,heavy_CO2_net_mol=net,light_waste_CO2_mol=waste[0],light_waste_N2_mol=waste[1],purity_pct=purity,recovery_pct=recovery,specific_energy_kWh_tCO2=sec))
    print(f'cycle {cyc:02d}: err={err:.3g}, purity={purity:.2f}%, recovery={recovery:.2f}%')
    if err<2e-5 and cyc>=2: css=True; break

df=pd.DataFrame(records); df.to_csv(os.path.join(OUT,'cycle_results.csv'),index=False)
last=df.iloc[-1].to_dict()
Y=state.reshape(4,Nz)[:2].T
bounds={'final_y_min':float(Y.min()),'final_y_max':float(Y.max()),'all_cycles_purity_0_100':bool(df.purity_pct.between(0,100,inclusive='both').all()),'all_cycles_recovery_0_100':bool(df.recovery_pct.between(0,100,inclusive='both').all()),'final_state_y_valid':bool(np.all((Y>=-1e-8)&(Y<=1+1e-8))),'css_converged':css}
summary={'model':'1D conservative finite-volume MOL; isothermal ideal gas; extended competitive Langmuir + LDF; pressure-controlled six-step cycle','parameters':{'Nz':Nz,'bed_length_m':L,'bed_diameter_m':D,'void_fraction':eps,'P_high_bar':Phi/1e5,'P_low_bar':Plo/1e5,'feed_CO2_mole_fraction':yf[0],'cycle_time_s':cycle_time,'cycles_run':len(df)},'final_cycle':last,'checks':bounds,'runtime_s':time.time()-start,'notes':'Flow integrals derive from cellwise total balance and upwind component fluxes. Pressure steps assume spatially uniform prescribed P; no Ergun momentum closure. Energy is idealized evacuation compression work only.'}
with open(os.path.join(OUT,'summary.json'),'w') as f: json.dump(summary,f,indent=2)
fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].plot(df.cycle,df.purity_pct,'o-',label='Purity'); ax[0].plot(df.cycle,df.recovery_pct,'s-',label='Recovery'); ax[0].set(xlabel='Cycle',ylabel='%'); ax[0].set_ylim(0,100); ax[0].grid(True); ax[0].legend(); ax[1].plot(df.cycle,df.css_error,'o-'); ax[1].set(xlabel='Cycle',ylabel='CSS max state difference'); ax[1].set_yscale('log'); ax[1].grid(True); fig.tight_layout(); fig.savefig(os.path.join(OUT,'comparison_css.png'),dpi=160); plt.close(fig)
# Profiles from final cycle, CO2 fraction in cell profile; compare ending of adsorption/rinse/evac.
fig,ax=plt.subplots(figsize=(7,4));
for nm in ('ads','rinse','evac'): ax.plot((np.arange(Nz)+.5)*dz,prof[nm][0],label=nm)
ax.set(xlabel='z (m)',ylabel='CO2 mole fraction',ylim=(0,1)); ax.grid(True); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT,'final_profiles.png'),dpi=160); plt.close(fig)
print('SUMMARY',json.dumps(summary,indent=2))
print('FILES',*[os.path.join(OUT,f) for f in ('cycle_results.csv','summary.json','comparison_css.png','final_profiles.png')],sep='\n')
