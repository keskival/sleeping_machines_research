"""B1/R1 error analysis using already-paid packets; oracle columns are diagnostics."""
import argparse, hashlib, json, resource, time
import torch
from torch.utils.flop_counter import FlopCounterMode
from fit_optimizer_aware_credit import ROOT, Correction, native
from cached_learning_effect import values

def load(name):
    d=json.loads((ROOT/name).read_text()); assert d['status']=='completed'
    p=ROOT/d['checkpoint']; assert hashlib.sha256(p.read_bytes()).hexdigest()==d['checkpoint_sha256']
    return d,torch.load(p,map_location='cpu',weights_only=False)

def diagnose(data,head):
    base=data['base']; static=data.get('fixed',data.get('static')); truth=data['aware']; n,r,a=truth.shape
    actions=native.base.actions()[None,None].expand(n,r,-1,-1)
    learned=values(head,data['z'],actions)
    predictions={'baseline':base,'learned_effect':base+learned,'oracle_effect':base+truth-static,'oracle_static_repair':static,'oracle_total':truth}
    flat=truth.flatten(1); oracle=flat.min(1).values; before=flat.gather(1,base.flatten(1).argmin(1)[:,None]).squeeze(1)
    rows=[]
    for name,pred in predictions.items():
        chosen=pred.flatten(1).argmin(1); chosen_loss=flat.gather(1,chosen[:,None]).squeeze(1)
        rows.append(dict(arm=name,mean_oracle_regret=float((chosen_loss-oracle).mean()),gain_over_baseline=float((before-chosen_loss).mean()),changed_prefixes=int((chosen!=base.flatten(1).argmin(1)).sum())))
    gaps=base.flatten(1).sort(1).values; gap=gaps[:,1]-gaps[:,0]
    best=flat.argmin(1); chosen=base.flatten(1).argmin(1); base_error=base.flatten(1)-flat
    pair_error=base_error.gather(1,chosen[:,None])-base_error.gather(1,best[:,None])
    return dict(prefixes=n,receivers=r,rows=rows,baseline_margin_quantiles=torch.quantile(gap,torch.tensor([0.,.25,.5,.75,1.])).tolist(),effect_rms=float((truth-static).square().mean().sqrt()),static_baseline_error_rms=float((base-static).square().mean().sqrt()),decision_pair_error_mean=float(pair_error.mean()),learning_effect_relative_mse=float((learned-(truth-static)).square().sum()/(truth-static).square().sum()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);args=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/'experiments/results/credit'/f'{args.tag}.json';assert not out.exists()
    train_result,train=load('experiments/results/credit/curie_optimizer_aware_fit_20261011T0010Z_pilot.json')
    selection,chosen=load('experiments/results/credit/curie_credit_trust_20261011T0130Z_select.json')
    fresh_result,fresh=load('experiments/results/credit/curie_credit_trust_20261011T0130Z_fresh.json')
    head=Correction(chosen['scale']).double();head.load_state_dict(chosen['head'])
    with torch.no_grad(),FlopCounterMode(display=False) as counter:
        metrics={'train':diagnose(train['train_tensors'],head),'fresh_examined_development':diagnose(fresh['packets'],head)}
    result=dict(status='completed',battle='B1/R1',tag=args.tag,metrics=metrics,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,supported_flops=counter.get_total_flops(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/credit/diagnose_route_decisions.py','experiments/credit/cached_learning_effect.py']},packet_checkpoint_sha256=[d['checkpoint_sha256'] for d in [train_result,selection,fresh_result]],scope='Saved packets only; no fit or new teacher. Oracle effect/static/total use future labels and are diagnostic ceilings, not deployable policies. Fresh packets now examined for development; future confirmation must use a new namespace. One synthetic actor seed170, readout-only causal Adam.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
