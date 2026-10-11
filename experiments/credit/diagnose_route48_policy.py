"""R1 inspect saved final policies against paid exact costs, with no new fitting."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM
from measure_token_forward_credit import verify
from fit_route48_transport import new_actor
from token_issued_credit_geometry import issue_geometry
from token_flop_accounting import TokenFlopCounter
PARENT='curie_route48_transport_20261011_v1_fit'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    parent=verify(PARENT);cp=ROOT/parent['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==parent['checkpoint_sha256']
    saved=torch.load(cp,weights_only=True);source=json.loads((ROOT/SOURCE).read_text());c=source['args'];native=ROOT/source['checkpoint']
    assert hashlib.sha256(native.read_bytes()).hexdigest()==parent['source_checkpoint_sha256']
    model=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);model.load_state_dict(torch.load(native,weights_only=True));model.requires_grad_(False);model.eval()
    rows=[]
    with TokenFlopCounter(display=False) as work,torch.no_grad():
        for i,w in enumerate(saved['dev_worlds']):
            issued=issue_geometry(model,w['tokens'][:64],w['tokens'][:64]);assert issued['receivers']==w['receivers'];assert torch.equal(issued['actions'],w['actions'])
            initial=new_actor(issued);l0=initial(issued['context'],issued['descriptors']);p0=.9*l0.softmax(0)+.1/len(l0);cost=w['costs'];centered=cost-cost.mean();headroom=float(cost.mean()-cost.min())
            for arm,state in saved['actors'].items():
                actor=new_actor(issued);actor.load_state_dict(state);logits=actor(issued['context'],issued['descriptors']);pi=.9*logits.softmax(0)+.1/len(logits)
                gain=float((p0-pi)@centered);tv=float((pi-p0).abs().sum()/2)
                rows.append(dict(prefix=i,arm=arm,total_variation_from_initial=tv,kl_from_initial=float((pi*(pi.log()-p0.log())).sum()),initial_gain=gain,uniform_oracle_headroom=headroom,headroom_fraction=gain/headroom,logit_span=float(logits.max()-logits.min()),expected_nll=float(pi@cost),initial_nll=float(p0@cost)))
    summary={arm:{key:float(np.mean([r[key] for r in rows if r['arm']==arm])) for key in ['total_variation_from_initial','kl_from_initial','initial_gain','headroom_fraction']} for arm in saved['actors']}
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,parent=PARENT,parent_sha256=hashlib.sha256((ROOT/f'experiments/results/credit/{PARENT}.json').read_bytes()).hexdigest(),rows=rows,summary=summary,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,supported_flops=work.get_total_flops(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/credit/diagnose_route48_policy.py']},scope='Post-fit engineering diagnosis of eight reused private scoring prefixes and paid exact costs, no fitting/new future replay/public scoring. Policy issuance uses only64observed tokens; future costs are used only for diagnosis. Not independent confirmation.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
