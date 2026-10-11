"""R1 bounded route-policy development with exhaustive fresh-prefix scoring."""
import argparse,hashlib,json,resource,time
from copy import deepcopy
import numpy as np
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM,load
from measure_token_forward_credit import verify
from token_issued_route_features import RouteController
from token_issued_credit_geometry import issue_geometry as issue_with_history
from token_forward_credit_vectorized import packet
from token_persistent_credit import joint_replay_carry
from token_static_carried_credit import static_carried_scores
from token_exact_world_readouts import replay
from route_policy_credit import route_credit_objective
from token_flop_accounting import TokenFlopCounter

SEED=98410
CONTEXT=64
LR=1e-4

def new_actor(issued):
    torch.manual_seed(SEED)
    return RouteController(len(issued['context']),issued['descriptors'].shape[1])

def fit(model,tokens,arm,windows):
    carrier=learner=actor=opt=None;rows=[];begin=time.monotonic()
    with TokenFlopCounter(display=False) as work:
        for i in range(windows):
            edge=CONTEXT+i*8
            prefix=tokens[:edge] if i==0 else tokens[edge-1:edge]
            current=tokens[:edge+8] if i==0 else tokens[edge-1:edge+8]
            index=CONTEXT-1 if i==0 else 0
            issued=issue_with_history(model,prefix,tokens[:edge],carrier,None if learner is None else learner['z'])
            if actor is None:actor=new_actor(issued);opt=torch.optim.Adam(actor.parameters(),lr=LR)
            logits=actor(issued['context'],issued['descriptors']);pi=.9*logits.softmax(0)+.1/len(logits)
            action=int(torch.multinomial(pi.detach(),1,generator=torch.Generator().manual_seed(SEED+1000+i)))
            p=packet(model,current,index=index,initial_state=carrier)
            actual=joint_replay_carry(p,issued['receivers'],issued['actions'],action,0,.01,initial_learner=learner)
            if arm=='zero':h=torch.zeros_like(logits)
            elif arm=='static':
                z=p.embedding.new_zeros(p.embedding.shape[1]) if learner is None else learner['z']
                h=static_carried_scores(p,issued['receivers'],issued['actions'],z).mean(1);h=h-h[0]
            else:h=actual['projected_scores'].mean(1);h=h-h[0]
            cost=actual['actual_scores'][0].mean()-actual['actual_scores'][1].mean()
            objective=route_credit_objective(logits,h,action,cost);opt.zero_grad();objective.backward()
            norm=float(sum(v.grad.square().sum() for v in actor.parameters()).sqrt());assert np.isfinite(norm)
            opt.step();carrier=actual['factual_carrier'];learner=actual['factual_learner']
            rows.append(dict(window=i,routes=len(logits),action=action,nll=float(actual['actual_scores'][0].mean()),gradient_norm=norm,worlds=actual['actual_worlds'],updates=actual['readout_updates']))
            if (i+1)%16==0:print(json.dumps(dict(arm=arm,windows=i+1,nll=rows[-1]['nll'])),flush=True)
    return actor,dict(arm=arm,windows=rows,wall_s=time.monotonic()-begin,supported_flops=work.get_total_flops(),actor_parameters=sum(v.numel() for v in actor.parameters()),actual_worlds=sum(r['worlds'] for r in rows),readout_updates=sum(r['updates'] for r in rows))

def evaluate(model,data,offset,actors,prefixes):
    rows=[];saved=[];begin=time.monotonic();worlds=updates=0
    with TokenFlopCounter(display=False) as work:
        for i in range(prefixes):
            tokens=torch.tensor(np.array(data[offset+i*256:offset+i*256+CONTEXT+8],dtype=np.int64))
            issued=issue_with_history(model,tokens[:CONTEXT],tokens[:CONTEXT]);p=packet(model,tokens,index=CONTEXT-1)
            true=replay(p,issued['receivers'],issued['actions'],.01);costs=true['scores'].mean(1)
            worlds+=true['actual_worlds'];updates+=true['readout_updates']
            policies=dict(actors,initial=new_actor(issued));scores={};greedy={}
            with torch.no_grad():
                for name,actor in policies.items():
                    logits=actor(issued['context'],issued['descriptors']);pi=.9*logits.softmax(0)+.1/len(logits)
                    scores[name]=float(pi@costs);greedy[name]=float(costs[int(logits.argmax())])
            rows.append(dict(prefix=i,expected_nll=scores,greedy_nll=greedy,noop_nll=float(costs[0]),uniform_nll=float(costs.mean()),oracle_nll=float(costs.min()),routes=len(costs)))
            saved.append(dict(tokens=tokens,receivers=issued['receivers'],actions=issued['actions'],costs=costs))
    return rows,saved,dict(wall_s=time.monotonic()-begin,supported_flops=work.get_total_flops(),worlds=worlds,readout_updates=updates)

def comparison(rows,better,reference):
    gains=np.array([r['expected_nll'][reference]-r['expected_nll'][better] for r in rows])
    rng=np.random.default_rng(98411);boot=gains[rng.integers(0,len(gains),(10000,len(gains)))].mean(1)
    return dict(gain=float(gains.mean()),paired_prefix_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),per_prefix=gains.tolist())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    for name in ['curie_continuous_token_route_20261011_v2_smoke','curie_route_policy_credit_20261011_v1_contract','curie_token_static_carried_credit_20261011_v1_contract','curie_token_issued_route_pool_20261011_v1_contract','curie_token_flop_accounting_20261011_v1_contract','curie_token_credit_geometry_20261011_v2_contract']:assert verify(name)['gate']
    if a.mode=='fit':assert verify('curie_continuous_token_route_dev_20261011_v2_smoke')['gate']
    source=json.loads((ROOT/SOURCE).read_text());c=source['args'];checkpoint=ROOT/source['checkpoint']
    model=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);model.load_state_dict(torch.load(checkpoint,weights_only=True));model.requires_grad_(False);model.eval()
    windows,prefixes=(4,2) if a.mode=='smoke' else (64,8)
    train_offset=c['train_tokens']+32768;dev_offset=c['train_tokens']+65536
    data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');tokens=torch.tensor(np.array(data[train_offset:train_offset+CONTEXT+windows*8],dtype=np.int64))
    actors={};fitting=[]
    for arm in ['zero','static','reference']:
        actor,row=fit(model,tokens,arm,windows);actors[arm]=actor;fitting.append(row)
    rows,saved,scoring=evaluate(model,data,dev_offset,actors,prefixes)
    comparisons={f'{b}_over_{r}':comparison(rows,b,r) for b,r in [('reference','zero'),('reference','static'),('reference','initial'),('static','zero'),('zero','initial')]}
    state=out.with_suffix('.pt');torch.save(dict(actors={k:v.state_dict() for k,v in actors.items()},dev_worlds=saved),state)
    paths=['experiments/credit/fit_continuous_token_routes_v2.py','experiments/credit/token_issued_credit_geometry.py','experiments/credit/TOKEN_ROUTE_GEOMETRY_DESIGN.md','experiments/credit/token_issued_route_pool.py','experiments/credit/token_issued_route_features.py','experiments/credit/token_persistent_credit.py','experiments/credit/token_static_carried_credit.py','experiments/credit/token_exact_world_readouts.py','experiments/credit/route_policy_credit.py','experiments/credit/token_flop_accounting.py','experiments/credit/CONTINUOUS_TOKEN_ROUTE_DEV_PROTOCOL.md']
    result=dict(status='completed',battle='R1',tag=a.tag,mode=a.mode,gate=True,config=dict(seed=SEED,context=CONTEXT,windows=windows,dev_prefixes=prefixes,actor_lr=LR,fast_lr=.01,exploration=.1,amplitude=.25,train_offset=train_offset,private_dev_offset=dev_offset),fitting=fitting,dev=rows,scoring=scoring,comparisons=comparisons,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoint=str(state.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},train_tokens_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),scope='One development initialization/fixed64window online route fit; fixed native2layer pretrained model,64dim adaptive lexical learner, connected normalized/forward-geometry controller. Exact expected stochastic eight-target costs on fresh private held-out TRAIN-tail prefixes, not public DEV/TEST. DEV episode lexical/key state starts fresh after64context; train carries continuous state. No recurrent actor state/full-carrier learning. All exhaustive scoring worlds/reference/projection/discovery/copies/duplicate issuance charged; supported FLOPs omit unsupported scalar/optimizer work included in wall. Source pretraining and gates additional. Endpoint checks are engineering evidence, not a family verdict or benchmark claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],comparisons=comparisons,whole_wall_s=result['whole_wall_s'])),flush=True)
if __name__=='__main__':main()
