"""R1 contract exact carried-world batching against independent actual learners."""
import argparse,hashlib,json,time
import torch
from check_token_forward_credit import ROOT,KeyedTokenLM
from token_forward_credit_vectorized import packet
from token_persistent_credit import joint_replay_carry
from token_exact_world_readouts_carried import replay

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);rows=[]
    for depth in [2,4,8]:
        torch.manual_seed(99200+depth);m=KeyedTokenLM(8,3,depth,2,4,1,0.);m.embed=torch.nn.Embedding(23,8);m.bias=torch.nn.Parameter(torch.zeros(23));m.requires_grad_(False);m.eval()
        actions=torch.zeros(3,2,4);actions[1,0,0]=.25;actions[2,1,1]=-.25;receivers=[2,4,6]
        warm=joint_replay_carry(packet(m,torch.tensor([2,4,6,8,2,4,6,8,4,6,2,8]),index=3),receivers,actions,1,0,.01)
        incoming=warm['factual_learner'];before=incoming['z'].clone()
        for reset in [False,True]:
            tokens=torch.tensor([2,4,6,8,4,6,2,8]);tokens[5]=7 if reset else 6
            p=packet(m,tokens,index=3,eos_token=7,initial_state=warm['factual_carrier'])
            batched=replay(p,receivers,actions,.01,True,initial_learner=incoming);errors=[]
            for k in range(3):
                ref=joint_replay_carry(p,receivers,actions,k,0,.01,initial_learner=incoming,diagnostics=True)
                errors.extend([float((batched['scores'][k]-ref['actual_scores'][0]).abs().max()),float((batched['z'][k]-ref['factual_learner']['z']).abs().max())])
                fields=next(iter(ref['factual_learner']['optimizer']['state'].values()))
                bulk=next(iter(batched['optimizer']['state'].values()))
                for name in ['exp_avg','exp_avg_sq']:errors.append(float((bulk[name][k]-fields[name]).abs().max()))
            assert torch.equal(before,incoming['z']);assert max(errors)<1e-10
            rows.append(dict(depth=depth,reset=reset,max_error=max(errors)))
    paths=['experiments/credit/check_exact_worlds_carried.py','experiments/credit/token_exact_world_readouts_carried.py','experiments/credit/token_persistent_credit.py']
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,checks=rows,wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
