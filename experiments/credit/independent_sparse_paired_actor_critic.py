"""B1/R1 source-preserving independent-state repair and exact cached replay."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import time
import torch
import sparse_paired_actor_critic as old
from cached_paired_native_replay import replay_pairs

ROOT=old.ROOT
TAG='curie_sparse_paired_ac_v3_20261011T0727Z'
SOURCES=[str(Path(__file__).relative_to(ROOT)),'experiments/credit/cached_paired_native_replay.py',*old.SOURCES]
old_unpack=old.unpack


def independent_unpack(payload,baseline):
    # load_state_dict copies module weights but optimizer tensors/lists may alias.
    return old_unpack(deepcopy(payload),baseline)


def contract():
    _,baseline,_,_,_=old.initial()
    prior,checkpoint=old.load(f'experiments/results/credit/{old.TAG}_acquire.json')
    payload=checkpoint['states']['sparse_learned']
    left=independent_unpack(payload,baseline);right=independent_unpack(payload,baseline)
    assert left['history'] is not right['history'] and left['buffer'] is not right['buffer']
    left_opt=left['policy_opt'].state_dict()['state'];right_opt=right['policy_opt'].state_dict()['state']
    checks=0
    for key in left_opt:
        for name in ['step','exp_avg','exp_avg_sq']:
            u,v=left_opt[key][name],right_opt[key][name]
            assert u.data_ptr()!=v.data_ptr()
            before=v.clone();u.add_(1.);assert torch.equal(before,v);checks+=1
    left['history'].append({'independence_probe':True})
    assert len(right['history'])==128 and len(payload['history'])==128
    assert all(u.data_ptr()!=v.data_ptr() for u,v in zip(left['actor'].parameters(),right['actor'].parameters()))
    return dict(optimizer_storage_checks=checks,history_buffer_independent=True,original_acquisition_unchanged=True,
                acquired_cursor=right['cursor'],source_checkpoint_sha256=prior['checkpoint_sha256'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['contract','smoke','reuse'],required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic()
    gate=json.loads((ROOT/'experiments/results/credit/curie_cached_pair_replay_20261011T0723Z_contract.json').read_text())
    assert gate['status']=='completed' and gate['kernel_admission']
    for p,digest in gate['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    old.unpack=independent_unpack;old.replay_pairs=replay_pairs
    if a.mode=='reuse':
        for mode in ['contract','smoke']:
            d=json.loads((ROOT/f'experiments/results/credit/{TAG}_{mode}.json').read_text());assert d['status']=='completed'
            for p,digest in d['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    if a.mode=='contract':metrics,payload=contract(),None
    else:
        metrics,payload=old.run(a.mode,a.tag)
        for name,state in payload['states'].items():assert len(state['history'])==state['cursor'],name
        names=list(payload['states'])
        for i,name in enumerate(names):
            for other in names[i+1:]:
                u,v=payload['states'][name],payload['states'][other]
                assert u['history'] is not v['history']
                if u['policy_optimizer']['state'] and v['policy_optimizer']['state']:
                    assert u['policy_optimizer']['state'][0]['exp_avg'].data_ptr()!=v['policy_optimizer']['state'][0]['exp_avg'].data_ptr()
    result=dict(status='completed',battle='B1/R1',tag=a.tag,mode=a.mode,metrics=metrics,
                whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
                acquisition_source=f'experiments/results/credit/{old.TAG}_acquire.json',
                reuse_protocol_repair='Independent deep copies of every acquired payload, including history/buffer/optimizer tensors; source128 acquisition preserved. Cached replay passes scores/gradients/Adam/weights and suffix contracts at depths2/4/8.',
                scope='One-seed synthetic connected route-learning development. Fixed native temporal/keyed depth4 carrier, optional hard writes, factual feedback/four causal branch-local readout Adam updates per episode. Same controller/critic initialization/data/update recipe as v2;512 controls now independently cloned. Acquisition full-prefix replay cost plus continuation cached replay cost charged in each cumulative fit. Whole wall includes unsupported/scalar/optimizer work; supported FLOPs alone are not complete arithmetic. Shared DEV verification and original source pretraining additional. No TEST/all-carrier/depth-scaling claim.')
    if payload is not None:
        cp=out.with_suffix('.pt');torch.save(payload,cp);result.update(checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
