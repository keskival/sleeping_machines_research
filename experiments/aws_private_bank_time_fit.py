"""Guarded source-bound positive-time private-bank fit; use only through run_safe."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--manifest',required=True);parser.add_argument('--tag',required=True)
    parser.add_argument('--memory-time-scale',required=True,type=float)
    a,extra=parser.parse_known_args()
    if not math.isfinite(a.memory_time_scale) or a.memory_time_scale<=0:raise ValueError('Positive memory time scale required')
    packet=json.loads(Path(a.manifest).read_text());pins=packet['token_source_sha256']
    def check():
        for name,digest in pins.items():
            if sha(ROOT/name)!=digest:raise ValueError('Source binding mismatch: '+name)
    check()
    for item in packet['data']:
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Pinned data mismatch: '+item['path'])
    from aws_private_bank_admission import validate_contract
    contract=json.loads((ROOT/packet['contract_result']).read_text())
    validate_contract(contract,packet['contract_source_sha256'])
    for label,fields in (
        ('time_contract_result',('scale_one_factual_gradient_exact','scaled_temporal_gradients_nonzero',
          'actual_alternative_suffix_credit_nonzero','model_state_rng_continuation_exact','cross_scale_checkpoint_rejected')),
        ('time_resume_result',('optimizer_continuation_exact',))):
        receipt=json.loads((ROOT/packet[label]).read_text())
        if receipt.get('status')!='completed' or any(receipt.get(k) is not True for k in fields):
            raise ValueError('Completed temporal contract required: '+label)
        for name,digest in receipt['source_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Temporal contract source changed: '+name)
    out=ROOT/'experiments/results/token_language'/a.tag
    if any(out.with_suffix(s).exists() for s in ('.json','.partial.json','.aws.json','.initial.pt','.selection.json')):
        raise FileExistsError('Unique bank run tag required')
    import torch
    from torch.nn import functional as F
    import horizon_token_language as selection
    import horizon_token_language_engine as engine
    import sleeping_machines.sparse_counterfactual_episodes as episodes
    from aws_private_bank_time_model import TimeScaledPrivateBankModel
    parent=engine.Model
    models=[]
    class ObservedBankModel(TimeScaledPrivateBankModel):
        def __init__(self,args,order,counts):
            if args.compiled or args.deterministic_eval or args.future_site!='uniform' or args.future_every!=1 or args.future_score_positions!=4:
                raise ValueError('Bank fit requires eager stochastic evaluation and uniform-site K4 future credit')
            super().__init__(args,order,counts,a.memory_time_scale)
            self.initial_clock_audit=[];self.initial_probe_done=False
            models.append(self)
        def forward(self,*args,**kwargs):
            if self.initial_probe_done:return super().forward(*args,**kwargs)
            original=episodes.sparse_counterfactual_layer
            def observe(x,arrival,m,arr,seen,reads,active,noise,alt_noise,mix_w,mix_b,query,key,key_read,clock_bias,*rest,**kw):
                # Same arithmetic as the untouched race kernel, only on truly cold heads.
                with torch.no_grad():
                    n,H,U,P=m.shape
                    features=F.layer_norm(F.linear(x,mix_w,mix_b),(H*P,))
                    q=torch.einsum('hpd,ld->lhp',query,features)
                    raw=(q[:,:,None,:]*reads).sum(-1)/math.sqrt(P)+clock_bias.view(H,U)
                    cold=(~seen.any(-1)) & (m==0).all(dim=(-1,-2))
                    cold_scores=raw[cold]
                    if cold_scores.numel():
                        canonical=cold_scores-self.bank_policy['clock_bias_shift']
                        self.initial_clock_audit.append(dict(candidate_scores=cold_scores.numel(),
                            shifted_min=float(cold_scores.min()),shifted_max=float(cold_scores.max()),
                            canonical_min=float(canonical.min()),canonical_max=float(canonical.max()),
                            unclipped=bool(((cold_scores>-12)&(cold_scores<12)&(canonical>-12)&(canonical<12)).all())))
                return original(x,arrival,m,arr,seen,reads,active,noise,alt_noise,mix_w,mix_b,query,key,key_read,clock_bias,*rest,**kw)
            episodes.sparse_counterfactual_layer=observe
            try:return super().forward(*args,**kwargs)
            finally:
                episodes.sparse_counterfactual_layer=original
                self.initial_probe_done=True
    engine.Model=ObservedBankModel
    old_argv=sys.argv
    sys.argv=['aws_private_bank_fit.py','--tag',a.tag,*extra]
    try:selection.main()
    finally:engine.Model=parent;sys.argv=old_argv
    check()
    result=json.loads(out.with_suffix('.json').read_text())
    chosen=json.loads(out.with_suffix('.selection.json').read_text())
    model=models[0]
    result['memory_time_scale']=a.memory_time_scale
    result['bank_recipe']=model.get_extra_state()
    result['initial_clock_audit']=dict(rows=model.initial_clock_audit,
        all_observed_cold_scores_unclipped=bool(model.initial_clock_audit) and all(r['unclipped'] for r in model.initial_clock_audit),
        scope='Cold heads in first initial evaluation forward only; no whole-population or trained-race equivalence claim')
    result['source_sha256']=pins
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    result['selection']=chosen
    result['original_best_trained_dev_nll']=result['best_dev_nll']
    result['best_dev_nll']=chosen['selected']['dev_nll']
    bundle=out.with_suffix('.artifacts.zip')
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_STORED) as archive:
        for artifact in sorted(out.parent.glob(a.tag+'.*')):
            if artifact.is_file() and artifact!=bundle:archive.write(artifact,arcname=artifact.name)
    result.update(final_weights=str(bundle.relative_to(ROOT)),artifact_sha256=sha(bundle),
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    check();out.with_suffix('.aws.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
