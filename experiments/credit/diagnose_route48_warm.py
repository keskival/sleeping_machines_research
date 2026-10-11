"""R1 warm lexical/Adam state diagnostic with identical cold native context."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM,load
from measure_token_forward_credit import verify
from fit_route48_transport import new_actor,comparison
from token_issued_credit_geometry import issue_geometry
from token_forward_credit_vectorized import packet
from token_persistent_credit import joint_replay_carry
from token_exact_world_readouts_carried import replay
from token_flop_accounting import TokenFlopCounter
PARENT='curie_route48_transport_20261011_v1_fit'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','measure'],required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);started=time.monotonic();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    parent=verify(PARENT);assert verify('curie_exact_worlds_carried_20261011_v1_contract')['gate']
    if a.mode=='measure':assert verify('curie_route48_warm_20261011_v1_smoke')['gate']
    cp=ROOT/parent['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==parent['checkpoint_sha256'];saved=torch.load(cp,weights_only=True)
    source=json.loads((ROOT/SOURCE).read_text());c=source['args'];native=ROOT/source['checkpoint'];assert hashlib.sha256(native.read_bytes()).hexdigest()==parent['source_checkpoint_sha256']
    m=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);m.load_state_dict(torch.load(native,weights_only=True));m.requires_grad_(False);m.eval()
    data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');rows=[];errors=[];snapshots=[];worlds=updates=0
    with TokenFlopCounter(display=False) as work:
        for i,w in enumerate(saved['dev_worlds'][:1 if a.mode=='smoke' else 8]):
            offset=parent['config']['private_dev_offset']+256*i;prefix=w['tokens'][:64]
            history=torch.tensor(np.array(data[offset-192:offset+64],dtype=np.int64));assert torch.equal(history[-64:],prefix)
            wp=packet(m,history,index=63);zero=torch.zeros(1,m.heads,m.dk)
            warm=joint_replay_carry(wp,[int(prefix[-1])],zero,0,0,.01);incoming=warm['factual_learner']
            # Reissue the last event from the same cold-native pre-event state.
            before=packet(m,prefix,index=62).state
            issued=issue_geometry(m,prefix[-1:],prefix,before,incoming['z'])
            cold=issue_geometry(m,prefix,prefix)
            assert issued['receivers']==w['receivers'];assert torch.equal(issued['actions'],w['actions'])
            assert torch.equal(issued['forecast']['log_probabilities'],cold['forecast']['log_probabilities'])
            p=packet(m,w['tokens'],index=63);true=replay(p,issued['receivers'],issued['actions'],.01,initial_learner=incoming)
            actual=joint_replay_carry(p,issued['receivers'],issued['actions'],0,0,.01,initial_learner=incoming)
            error=float((true['scores'][0]-actual['actual_scores'][0]).abs().max());assert error<1e-10;errors.append(error)
            costs=true['scores'].mean(1);actors=dict(saved['actors']);scores={}
            with torch.no_grad():
                for arm in ['initial',*actors]:
                    actor=new_actor(issued)
                    if arm!='initial':actor.load_state_dict(actors[arm])
                    logits=actor(issued['context'],issued['descriptors']);pi=.9*logits.softmax(0)+.1/len(logits);scores[arm]=float(pi@costs)
            rows.append(dict(prefix=i,expected_nll=scores,cold_expected_nll=parent['dev'][i]['expected_nll'],uniform_nll=float(costs.mean()),oracle_nll=float(costs.min()),noop_nll=float(costs[0]),warm_lexical_norm=float(incoming['z'].norm()),routes=len(costs)))
            snapshots.append(dict(tokens=w['tokens'],incoming=incoming,costs=costs))
            worlds+=warm['actual_worlds']+true['actual_worlds']+actual['actual_worlds'];updates+=warm['readout_updates']+true['readout_updates']+actual['readout_updates']
            print(json.dumps(rows[-1]),flush=True)
    checkpoint=out.with_suffix('.pt');torch.save(snapshots,checkpoint)
    paths=['experiments/credit/diagnose_route48_warm.py','experiments/credit/token_exact_world_readouts_carried.py','experiments/credit/WARM_POLICY_DIAGNOSIS_PROTOCOL.md']
    result=dict(status='completed',battle='R1',tag=a.tag,mode=a.mode,gate=True,parent=PARENT,parent_sha256=hashlib.sha256((ROOT/f'experiments/results/credit/{PARENT}.json').read_bytes()).hexdigest(),rows=rows,comparisons={f'{b}_over_{r}':comparison(rows,b,r) for b,r in [('transport','static'),('transport','zero'),('transport','initial'),('static','initial')]},reference_errors=errors,worlds=worlds,readout_updates=updates,supported_flops=work.get_total_flops(),wall_s=time.monotonic()-started,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Reused private development prefixes/selected final policies, no fitting/public scoring. Only lexical/Adam state warm; native64context same as previous cold scoring. Warm causal191updates and all exact worlds/reference paid. Measure includes smoke-prefix reuse. Not a confirmation or whole-native online/depth result.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['comparisons']),flush=True)
if __name__=='__main__':main()
