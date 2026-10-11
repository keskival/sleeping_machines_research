"""R1 connected future learning-state credit after eight observed forward targets."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM
from measure_token_forward_credit import verify
from fit_forward_credit_predictor import bootstrap
from token_issued_credit_geometry import issue_geometry
from token_forward_credit_vectorized import packet
from token_optimizer_transport_carried import replay_transport
from token_persistent_credit import joint_replay_carry
from token_exact_world_readouts import replay
from token_boundary_credit import BoundaryCredit,boundary_features
from connected_credit_risk import corrected_credit_risk
from token_flop_accounting import TokenFlopCounter
PARENT='curie_forward_credit_scaling_20261011_v1_fit'

def acquire(m,r,exact_all=False):
    seen=r['tokens'][:72];original=issue_geometry(m,seen[:64],seen[:64]);p=packet(m,seen,index=63)
    teacher=replay_transport(p,original['receivers'],original['actions'],.01,'diagonal',diagnostics=True)
    boundary=issue_geometry(m,seen[-1:],seen,p.state,teacher['reference_z'])
    record=dict(r,boundary_geometry=boundary['context']);context,difference=boundary_features(teacher,record,m.embed.weight.detach())
    first=teacher['scores'].mean(1);first=first-first[0]
    if exact_all:
        true=replay(p,original['receivers'],original['actions'],.01);c8=true['scores'].mean(1);c8=c8-c8[0];full=r['costs']-r['costs'][0]
        target=(48*full-8*c8)/40;worlds=true['actual_worlds'];updates=true['readout_updates'];cost=None
    else:
        actual=joint_replay_carry(p,original['receivers'],original['actions'],r['action'],0,.01)
        assert float((actual['actual_scores'][1]-teacher['reference_scores']).abs().max())<1e-10
        c8=(actual['actual_scores'][0]-actual['actual_scores'][1]).mean();target=(48*r['approximate']-8*first)/40;cost=(48*r['exact_cost']-8*c8)/40
        worlds=actual['actual_worlds'];updates=actual['readout_updates']
    return dict(context=context,differences=difference,target=target,action=r.get('action'),probabilities=r.get('probabilities'),exact_cost=cost),dict(actual_worlds=worlds+teacher['actual_worlds'],actual_updates=updates+teacher['reference_updates'],transported_updates=teacher['transported_updates'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);a=ap.parse_args();start=time.monotonic();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();parent=verify(PARENT);assert verify('curie_connected_credit_risk_20261011_v1_contract')['gate']
    if a.mode=='fit':assert verify('curie_boundary_credit_20261011_v1_smoke')['gate']
    cp=ROOT/parent['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==parent['checkpoint_sha256'];saved=torch.load(cp,weights_only=True)
    source=json.loads((ROOT/SOURCE).read_text());c=source['args'];native=ROOT/source['checkpoint'];assert hashlib.sha256(native.read_bytes()).hexdigest()==parent['source_checkpoint_sha256']
    m=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);m.load_state_dict(torch.load(native,weights_only=True));m.requires_grad_(False);m.eval()
    ntrain,ndev=(2,1) if a.mode=='smoke' else (128,16);records=[];dev=[];stages={}
    for split,old,count in [('acquisition',saved['train_records'],ntrain),('scoring_acquisition',saved['dev_records'],ndev)]:
        tick=time.monotonic();counts=dict(actual_worlds=0,actual_updates=0,transported_updates=0)
        with TokenFlopCounter(display=False) as work:
            for i,r in enumerate(old[:count]):
                row,cost=acquire(m,r,split=='scoring_acquisition');(dev if split=='scoring_acquisition' else records).append(row)
                for key in counts:counts[key]+=cost[key]
                if (i+1)%32==0:print(json.dumps(dict(stage=split,prefixes=i+1)),flush=True)
        stages[split]=dict(wall_s=time.monotonic()-tick,supported_flops=work.get_total_flops(),**counts)
    models={};tick=time.monotonic();fits=[]
    with TokenFlopCounter(display=False) as work:
        for arm in ['uncorrected','corrected']:
            torch.manual_seed(99610);model=BoundaryCredit(len(records[0]['context']),m.embed.embedding_dim);opt=torch.optim.Adam(model.parameters(),lr=.001)
            for epoch in range(10):
                for i in torch.randperm(ntrain,generator=torch.Generator().manual_seed(99611+epoch)).tolist():
                    r=records[i];h=model(r['context'],r['differences'])
                    loss=corrected_credit_risk(h,r['target'],r['action'],r['exact_cost'],r['probabilities']) if arm=='corrected' else (h-r['target']).square().mean()
                    loss=loss/.001**2;assert bool(torch.isfinite(loss));opt.zero_grad();loss.backward();opt.step()
            models[arm]=model;fits.append(dict(arm=arm,updates=10*ntrain,parameters=sum(v.numel() for v in model.parameters())))
    stages['fitting']=dict(wall_s=time.monotonic()-tick,supported_flops=work.get_total_flops(),arms=fits)
    rows=[]
    with torch.no_grad():
        for i,r in enumerate(dev):
            predictions={arm:model(r['context'],r['differences']) for arm,model in models.items()};predictions['zero']=torch.zeros_like(r['target'])
            mse={arm:float((h-r['target']).square().mean()) for arm,h in predictions.items()};r['predictions']=predictions
            rows.append(dict(prefix=i,mse=mse,tail_rms=float(r['target'].square().mean().sqrt()),prediction_rms={arm:float(h.square().mean().sqrt()) for arm,h in predictions.items()}))
    comparisons={f'{b}_over_{r}':bootstrap([v['mse'][r]-v['mse'][b] for v in rows]) for b,r in [('corrected','zero'),('corrected','uncorrected'),('uncorrected','zero')]}
    state=out.with_suffix('.pt');torch.save(dict(train_records=records,dev_records=dev,models={arm:model.state_dict() for arm,model in models.items()}),state)
    paths=['experiments/credit/fit_boundary_credit.py','experiments/credit/token_boundary_credit.py','experiments/credit/BOUNDARY_CREDIT_PROTOCOL.md','experiments/credit/token_learner_state_features.py']
    result=dict(status='completed',battle='R1',tag=a.tag,mode=a.mode,gate=True,predictive_gate=comparisons['corrected_over_zero']['paired_prefix_bootstrap95'][0]>0,config=dict(train_prefixes=ntrain,dev_prefixes=ndev,observed_targets=8,future_targets=40,fit_passes=10),stages=stages,rows=rows,comparisons=comparisons,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,inherited_parent=PARENT,inherited_parent_sha256=hashlib.sha256((ROOT/f'experiments/results/credit/{PARENT}.json').read_bytes()).hexdigest(),inherited_full48_stages=parent['stages'],checkpoint=str(state.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Reuse128TRAIN/16privateDEV trajectories; after8causalobservedtargets predict40futurelearner-state effects, not issuance foresight. Fixednative2layer/GPT2tokens, shared parameter/moment sensitivity field, approximate labels plus exact sampled residual training, exact reconstructed DEVtail. Currentphysical routeeffect omitted from predictorfeatures but included in tail labels. Inherited48target labels/scoring charged separately; no actor/fullnative learning, public scoring, confirmation or depth-efficiency claim. Unsupported arithmetic included in wall; standalone scoring predictor inference work not separately measured.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(comparisons=comparisons,predictive_gate=result['predictive_gate'],wall_s=result['wall_s'])),flush=True)
if __name__=='__main__':main()
