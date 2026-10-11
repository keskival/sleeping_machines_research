"""Bounded R1 complete route-learning cycle on genuine TRAIN tokens."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM,load
from measure_token_forward_credit import verify
from token_issued_route_features import issue,RouteController
from token_forward_credit_vectorized import packet
from token_persistent_credit import joint_replay_carry
from token_static_carried_credit import static_carried_scores
from route_policy_credit import route_credit_objective
from token_flop_accounting import TokenFlopCounter


def run(model,tokens,arm):
    torch.manual_seed(98400);carrier=learner=None;actor=optimizer=None;rows=[]
    begin=time.monotonic()
    with TokenFlopCounter(display=False) as work:
        for i in range(4):
            prefix=tokens[:16] if i==0 else tokens[15+i*8:16+i*8]
            current=tokens[:24] if i==0 else tokens[15+i*8:24+i*8]
            index=15 if i==0 else 0
            z=None if learner is None else learner['z']
            issued=issue(model,prefix,initial_state=carrier,fast_state=z)
            # After initial context, this smoke restricts discovery to the
            # current observed address. It tests integration, not final pool quality.
            if actor is None:
                actor=RouteController(len(issued['context']),issued['descriptors'].shape[1])
                optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4)
            logits=actor(issued['context'],issued['descriptors'])
            pi=.9*logits.softmax(0)+.1/len(logits)
            generator=torch.Generator().manual_seed(98600+i)
            action=int(torch.multinomial(pi.detach(),1,generator=generator))
            p=packet(model,current,index=index,initial_state=carrier)
            actual=joint_replay_carry(p,issued['receivers'],issued['actions'],action,0,.01,initial_learner=learner)
            if arm=='zero':h=torch.zeros_like(logits)
            elif arm=='static':
                initial=p.embedding.new_zeros(p.embedding.shape[1]) if learner is None else learner['z']
                h=static_carried_scores(p,issued['receivers'],issued['actions'],initial).mean(1)
                h=h-h[0]
            else:
                h=actual['projected_scores'].mean(1);h=h-h[0]
            cost=actual['actual_scores'][0].mean()-actual['actual_scores'][1].mean()
            objective=route_credit_objective(logits,h,action,cost)
            optimizer.zero_grad();objective.backward()
            assert all(torch.isfinite(v.grad).all() for v in actor.parameters())
            gradient_norm=float(sum(v.grad.square().sum() for v in actor.parameters()).sqrt())
            optimizer.step();carrier=actual['factual_carrier'];learner=actual['factual_learner']
            rows.append(dict(window=i,routes=len(logits),action=action,
                             factual_nll=float(actual['actual_scores'][0].mean()),
                             gradient_norm=gradient_norm,actual_worlds=actual['actual_worlds'],
                             readout_updates=actual['readout_updates']))
    return dict(arm=arm,windows=rows,wall_s=time.monotonic()-begin,
                supported_flops=work.get_total_flops(),actor_parameters=sum(v.numel() for v in actor.parameters()),
                final_fast_norm=float(learner['z'].norm()),final_active_keys=len(carrier['keys']) if carrier else 0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    for name in ['curie_route_policy_credit_20261011_v1_contract',
                 'curie_token_issued_route_features_20261011_v1_contract',
                 'curie_token_static_carried_credit_20261011_v1_contract',
                 'curie_token_persistent_credit_20261011_v1_contract',
                 'curie_token_vectorized_transport_20261011_v1_contract',
                 'curie_token_flop_accounting_20261011_v1_contract']:assert verify(name)['gate']
    source=json.loads((ROOT/SOURCE).read_text());c=source['args'];checkpoint=ROOT/source['checkpoint']
    model=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout'])
    model.load_state_dict(torch.load(checkpoint,weights_only=True));model.requires_grad_(False);model.eval()
    offset=c['train_tokens']+20480;data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin')
    tokens=torch.tensor(np.array(data[offset:offset+48],dtype=np.int64))
    rows=[run(model,tokens,arm) for arm in ['zero','static','reference']]
    assert all(r['windows'][0]['action']==rows[0]['windows'][0]['action'] for r in rows)
    paths=['experiments/credit/smoke_continuous_token_route_learning.py',
           'experiments/credit/token_issued_route_features.py','experiments/credit/route_policy_credit.py',
           'experiments/credit/token_static_carried_credit.py','experiments/credit/token_persistent_credit.py',
           'experiments/credit/token_forward_credit_vectorized.py','experiments/credit/token_flop_accounting.py']
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,rows=rows,
                train_token_offset=offset,context_tokens=16,scored_targets_per_arm=32,
                source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope='Integration smoke only: four8-target windows/three learned-route arms, proper GPT-2 TRAIN, fixed native2layer carrier, adaptive64dim lexical Adam and connected actor. First pool up8 observed addresses; later windows current-address-only restriction, no recurrent actor state. Duplicate issued-prefix acquisition and unused reference/frozen projections charged; not efficient final implementation. No DEV/TEST, utility/variance/depth/whole-fit claim. Source pretraining/contracts additional. Actual/reference learner and key state carry with EOS reset, actor updated only after window, issued features prefix-only.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
