#!/usr/bin/env python3
"""Reduced-order dual-stage VPSA rinse/recycle study and GP expected-improvement search.

Illustrative engineering surrogate only: not a validated adsorption/PSA simulator.
Inputs from the supplied breakthrough chart: CO2 breakthrough onset ~500 s in a
600 s trace (front index 0.833); all other process values are explicitly stated
assumptions below and should be replaced by measured cycle/adsorption data.
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from scipy.stats import norm

OUT = '/mnt/data/vpsa_corrected/results'
os.makedirs(OUT, exist_ok=True)
SEED = 27
rng = np.random.default_rng(SEED)
# Transparent base-case assumptions (molar basis)
FEED_MOL_MIN = 100.0
FEED_CO2_MOL_FRACTION = 0.15
STAGE1_CAPTURE = 0.94
BREAKTHROUGH_FRONT_INDEX = 500.0 / 600.0
# Rinse ratio is rinse-gas moles / stage-2 feed moles. The optimum is centered
# at the chart-derived front index. Width/ceiling are calibration assumptions.
RINSE_OPTIMUM = BREAKTHROUGH_FRONT_INDEX
RINSE_WIDTH = 0.22
MAX_STAGE2_RECOVERY = 0.92
MIN_STAGE2_RECOVERY = 0.08


def stage2_recovery(r):
    """Smooth front-matched rinse response, clipped to physical [0,1]."""
    r = np.asarray(r, dtype=float)
    return np.clip(MIN_STAGE2_RECOVERY + (MAX_STAGE2_RECOVERY-MIN_STAGE2_RECOVERY)
                   * np.exp(-((r-RINSE_OPTIMUM)/RINSE_WIDTH)**2), 0.0, 1.0)


def process(rinse_ratio, recycle_fraction):
    """Steady geometric recycle balance, assuming recycled CO2 is reprocessed.

    A = S1*S2 is fresh-pass product recovery; B = S1*(1-S2)*recycle is
    returned CO2 per pass. Total product recovery=A/(1-B). Rinse affects
    recovery via front match; recycle is limited to [0,1].
    """
    s2 = float(stage2_recovery(rinse_ratio))
    rec = float(np.clip(recycle_fraction, 0, 1))
    a = STAGE1_CAPTURE * s2
    b = STAGE1_CAPTURE * (1.0-s2) * rec
    overall = a / max(1.0-b, 1e-9)
    # A transparent approximate CO2 product purity penalty from non-CO2
    # co-desorption and excess rinse/recycle gas (not a detailed column model).
    purity = 0.985 - 0.075*max(0.0, rinse_ratio-RINSE_OPTIMUM) \
             - 0.045*rec - 0.018*abs(rinse_ratio-RINSE_OPTIMUM)
    purity = float(np.clip(purity, 0.50, 0.995))
    feed_co2 = FEED_MOL_MIN * FEED_CO2_MOL_FRACTION
    product_co2 = feed_co2 * overall
    tail_co2 = feed_co2 * (1-overall)
    # Tail-gas CO2 fraction is represented using an assumed 35 mol/min inert
    # tail flow plus unrecovered CO2; recycled portion is removed from net tail.
    net_tail_co2 = tail_co2 * (1-rec)
    tail_fraction = net_tail_co2 / max(35.0 + net_tail_co2, 1e-9)
    rinse_mol_min = rinse_ratio * (feed_co2*STAGE1_CAPTURE)
    return {'stage2_recovery': s2, 'overall_recovery': float(np.clip(overall,0,1)),
            'product_purity': purity, 'product_CO2_mol_min': product_co2,
            'net_tail_CO2_mol_min': net_tail_co2, 'tail_CO2_fraction': tail_fraction,
            'rinse_ratio': float(rinse_ratio), 'recycle_fraction': rec,
            'rinse_mol_min_assumed': rinse_mol_min}


def pareto_ranks(df, maximize_cols):
    vals = df[maximize_cols].to_numpy(float)
    n = len(vals); remaining = set(range(n)); ranks = np.full(n, -1, int); rank = 0
    while remaining:
        front=[]
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j: continue
                if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                    dominated = True; break
            if not dominated: front.append(i)
        for i in front: ranks[i] = rank
        remaining.difference_update(front); rank += 1
    return ranks

# (1) CO2 rinse sweep, including an intentionally poor baseline and front tune.
r_values = np.unique(np.r_[np.linspace(0.05, 1.35, 53), RINSE_OPTIMUM, 0.15])
r_rows=[]
for r in r_values:
    d=process(r, 0.0)
    r_rows.append({'rinse_ratio':r, 'front_index_from_chart':BREAKTHROUGH_FRONT_INDEX,
                   'stage2_recovery':d['stage2_recovery'],
                   'overall_recovery_no_recycle':d['overall_recovery'],
                   'product_purity_est':d['product_purity'],
                   'rinse_mol_min_assumed':d['rinse_mol_min_assumed']})
r_df=pd.DataFrame(r_rows).sort_values('rinse_ratio')
r_df.to_csv(os.path.join(OUT,'rinse_tuning_sweep.csv'),index=False)

# (2) Recycle fraction study at the tuned rinse setting.
r_best=float(r_df.loc[r_df.stage2_recovery.idxmax(),'rinse_ratio'])
recycle_rows=[]
for f in np.linspace(0,0.99,34):
    d=process(r_best,f)
    recycle_rows.append(d)
recycle_df=pd.DataFrame(recycle_rows)
recycle_df['overall_recovery_percent']=100*recycle_df.overall_recovery
recycle_df.to_csv(os.path.join(OUT,'recycle_study.csv'),index=False)

# (3) GP Bayesian optimization of rinse ratio + recycle fraction. EI is applied
# to a stated scalar utility (recovery + purity value - rinse penalty); then
# evaluated designs are independently Pareto-ranked on recovery and purity.
def utility(x):
    d=process(x[0],x[1])
    # 85% weight recovery; 15% purity; modest penalty for rinse use.
    return 0.85*d['overall_recovery'] + 0.15*d['product_purity'] - 0.025*x[0]

bounds=np.array([[0.05,1.35],[0.0,0.99]])
X=rng.uniform(bounds[:,0],bounds[:,1],size=(16,2))
y=np.array([utility(x) for x in X])
records=[process(x[0],x[1]) for x in X]
for it in range(36):
    kernel=C(1.0,(1e-2,1e2))*RBF(length_scale=[0.25,0.25],length_scale_bounds=(1e-2,2.0))+WhiteKernel(noise_level=1e-6,noise_level_bounds=(1e-8,1e-2))
    gp=GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1,
                                random_state=SEED+it, alpha=1e-8)
    gp.fit(X,y)
    cand=rng.uniform(bounds[:,0],bounds[:,1],size=(5000,2))
    mu,sd=gp.predict(cand,return_std=True)
    best=float(np.max(y)); imp=mu-best-0.01
    z=np.divide(imp,sd,out=np.zeros_like(imp),where=sd>1e-12)
    ei=imp*norm.cdf(z)+sd*norm.pdf(z)
    x=cand[int(np.argmax(ei))]
    X=np.vstack([X,x]); y=np.append(y,utility(x)); records.append(process(x[0],x[1]))
bo=pd.DataFrame(records)
bo['utility']=y
bo['pareto_rank_recovery_purity']=pareto_ranks(bo,['overall_recovery','product_purity'])
bo['evaluation_id']=np.arange(1,len(bo)+1)
bo.to_csv(os.path.join(OUT,'bo_pareto.csv'),index=False)

# Report the best Pareto design by recovery among rank-0 solutions and also
# best scalar-utility design. Note: deterministic surrogate has no uncertainty.
front=bo[bo.pareto_rank_recovery_purity==0]
pareto_pick=front.loc[front.overall_recovery.idxmax()]
utility_pick=bo.loc[bo.utility.idxmax()]
base=process(0.15,0.0)
tuned=process(r_best,0.0)
recycled=process(r_best,0.99)
summary={
 'model_status':'illustrative reduced-order surrogate; not validated process simulation',
 'chart_data_used':{'breakthrough_onset_s':500,'trace_duration_s':600,'front_index':BREAKTHROUGH_FRONT_INDEX},
 'assumptions':{'feed_total_mol_min':FEED_MOL_MIN,'feed_CO2_mole_fraction':FEED_CO2_MOL_FRACTION,
                'stage1_CO2_capture_fraction':STAGE1_CAPTURE,'stage2_recovery_peak':MAX_STAGE2_RECOVERY,
                'rinse_optimum_centered_at_front_index':True,
                'rinse_response_width_assumption':RINSE_WIDTH,
                'purity_and_tail_flow_are_empirical_surrogates':True},
 'baseline_rinse_ratio_0.15':base,
 'best_rinse_no_recycle':tuned,
 'best_rinse_ratio':r_best,
 'stage2_recovery_improvement_percentage_points':100*(tuned['stage2_recovery']-base['stage2_recovery']),
 'recycle_at_0.99':recycled,
 'best_pareto_recovery_design':pareto_pick.to_dict(),
 'best_expected_utility_design':utility_pick.to_dict(),
 'bo_evaluations':len(bo),
 'standard_comparison':{
   'requested_claim_stage2_recovery_above_85_percent':bool(tuned['stage2_recovery']>=0.85),
   'requested_claim_overall_recovery_above_90_percent_with_recycle':bool(recycled['overall_recovery']>=0.90),
   'comparison_caveat':'These thresholds are checked against this assumed surrogate, not measured/validated VPSA performance.'
 }
}
with open(os.path.join(OUT,'improved_summary.json'),'w',encoding='utf-8') as f:
    json.dump(summary,f,indent=2,ensure_ascii=False)

# Print the final result table and verify all requested files exist and are nonempty.
print('Rinse baseline vs optimum:')
print(pd.DataFrame([{'case':'baseline rinse r=0.15',**base},{'case':'tuned rinse, no recycle',**tuned}])[
    ['case','rinse_ratio','stage2_recovery','overall_recovery','product_purity']].to_string(index=False))
print('\nRecycle study selected rows:')
print(recycle_df.iloc[[0,10,20,30,33]][['recycle_fraction','stage2_recovery','overall_recovery','product_purity','tail_CO2_fraction']].to_string(index=False))
print('\nBO top Pareto results:')
print(bo[bo.pareto_rank_recovery_purity==0].sort_values('overall_recovery',ascending=False).head(8)[
    ['evaluation_id','rinse_ratio','recycle_fraction','stage2_recovery','overall_recovery','product_purity','utility']].to_string(index=False))
print('\nVerification:')
for name in ['rinse_tuning_sweep.csv','recycle_study.csv','bo_pareto.csv','improved_summary.json']:
    p=os.path.join(OUT,name)
    print(f'{p}: exists={os.path.exists(p)}, bytes={os.path.getsize(p) if os.path.exists(p) else 0}')
print('\nSUMMARY:',json.dumps(summary,indent=2,ensure_ascii=False))
