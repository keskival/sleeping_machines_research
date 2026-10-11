"""R1 branch influence beyond the existing eight-target credit window."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM,load
from measure_token_forward_credit import verify
from token_issued_route_pool import issue_with_history
from token_forward_credit_vectorized import packet
from token_persistent_credit import joint_replay_carry
from token_flop_accounting import TokenFlopCounter


def trajectory(model,tokens,receiver,action,lr,windows):
    carrier=learner=None;rows=[]
    for i in range(windows):
        edge=64+i*8;current=tokens[:edge+8] if i==0 else tokens[edge-1:edge+8]
        p=packet(model,current,index=63 if i==0 else 0,initial_state=carrier)
        delta=action if i==0 else torch.zeros_like(action)
        actual=joint_replay_carry(p,[receiver],delta[None],0,0,lr,initial_learner=learner)
        carrier=actual['factual_carrier'];learner=actual['factual_learner']
        rows.append(dict(scores=actual['actual_scores'][0].clone(),z=learner['z'].clone()))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','measure'],required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    assert verify('curie_token_persistent_credit_20261011_v1_contract')['gate']
    if a.mode=='measure':assert verify('curie_token_credit_tail_20261011_v1_smoke')['status']=='completed'
    count,windows=(1,2) if a.mode=='smoke' else (4,6)
    source=json.loads((ROOT/SOURCE).read_text());c=source['args'];checkpoint=ROOT/source['checkpoint']
    m=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);m.load_state_dict(torch.load(checkpoint,weights_only=True));m.requires_grad_(False);m.eval()
    data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');offset=c['train_tokens']+98304
    rows=[]
    with TokenFlopCounter(display=False) as work:
        for j in range(count):
            tokens=torch.tensor(np.array(data[offset+j*256:offset+j*256+64+windows*8],dtype=np.int64))
            issued=issue_with_history(m,tokens[:64],tokens[:64]);d=m.embed.embedding_dim;k=m.heads*m.dk
            q=issued['context'][d:d+k].reshape(m.heads,m.dk)
            beta=(issued['actions']*q[None]).sum((1,2))/np.sqrt(m.dk)
            index=int(beta.abs().argmax());receiver=issued['receivers'][index];action=issued['actions'][index]
            for lr in [0.,.01]:
                factual=trajectory(m,tokens,receiver,action,lr,windows);reference=trajectory(m,tokens,receiver,torch.zeros_like(action),lr,windows)
                differences=[]
                for i,(f,r) in enumerate(zip(factual,reference)):
                    differences.append(dict(window=i,mean_loss_contrast=float((f['scores']-r['scores']).mean()),learning_state_distance=float((f['z']-r['z']).norm())))
                rows.append(dict(prefix=j,lr=lr,receiver=receiver,issued_beta=float(beta[index]),windows=differences,first8_sum=float((factual[0]['scores']-reference[0]['scores']).sum()),later_sum=sum(float((f['scores']-r['scores']).sum()) for f,r in zip(factual[1:],reference[1:]))))
    paths=['experiments/credit/measure_token_credit_tail.py','experiments/credit/token_persistent_credit.py','experiments/credit/token_issued_route_pool.py','experiments/credit/token_forward_credit_vectorized.py','experiments/credit/token_flop_accounting.py']
    result=dict(status='completed',battle='R1',tag=a.tag,mode=a.mode,rows=rows,train_offset=offset,context=64,targets_per_world=windows*8,actual_worlds=count*4,readout_updates=count*4*windows*8,supported_flops=work.get_total_flops(),whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Four fresh TRAIN prefixes, prefix-only maximum absolute immediate-logit perturbation selects issued route. Independent factual/no-op trajectories across48targets; fixed-readout/direct-only vs independent lexical Adam(.01), same source carrier. Measures tail of this selected diagnostic, not policy utility or a native contraction certificate. Full acquisition/state/normalized output/readout/projection work paid; lr0 optimizer calls counted. No DEV/TEST.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
