"""R1 independent optimizer/gradient contract for all-route lexical learners."""
import argparse,hashlib,json,resource,time
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM,independent,load,np
from token_forward_credit import packet
from token_exact_world_readouts import replay
SOURCES=['experiments/credit/check_token_exact_world_readouts.py','experiments/credit/token_exact_world_readouts.py','experiments/credit/token_forward_credit.py','experiments/credit/check_token_forward_credit.py','experiments/credit/token_episode_state.py','experiments/r1_token_keyed_lm.py','experiments/tpp/race_tpp_v5.py']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    gate=json.loads((ROOT/'experiments/results/credit/curie_token_forward_credit_20261011_v1_contract.json').read_text());assert gate['status']=='completed' and gate['gate']
    for p,digest in gate['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    checks=[]
    for kind in ['small','proper_gpt2']:
        if kind=='small':
            torch.manual_seed(1001);model=KeyedTokenLM(8,3,4,2,4,1,0.);model.embed=torch.nn.Embedding(23,8);model.bias=torch.nn.Parameter(torch.randn(23)*.02);tokens=torch.tensor([2,4,6,8,4,7,6,8]);eos=7
        else:
            source=json.loads((ROOT/SOURCE).read_text());c=source['args'];model=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);model.load_state_dict(torch.load(ROOT/source['checkpoint'],weights_only=True));data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');tokens=torch.tensor(np.array(data[c['train_tokens']+4104:c['train_tokens']+4112],dtype=np.int64));eos=50256
        model.requires_grad_(False);model.eval();p=packet(model,tokens,eos_token=eos)
        receivers=[int(tokens[2]),len(model.embed.weight)-1,int(tokens[3])];actions=torch.zeros(3,model.heads,model.dk);actions[0,0,0]=.25;actions[1,-1,-1]=-.25
        batch=replay(p,receivers,actions,.01,True);errors={}
        for k in range(3):
            actual=independent(model,tokens,receivers[k],actions[k],.01,eos_token=eos)
            errors[f'scores{k}']=float((batch['scores'][k]-actual['scores']).abs().max())
            errors[f'z{k}']=float((batch['z'][k]-actual['z']).abs().max())
            errors[f'gradients{k}']=max(float((g-batch['gradients'][j][k]).abs().max()) for j,g in enumerate(actual['gradients']))
            errors[f'adam{k}']=max([float((v[field]-batch['optimizer']['state'][n][field][k]).abs().max()) for n,v in actual['optimizer']['state'].items() for field in ['exp_avg','exp_avg_sq']]+[0.])
        assert max(errors.values())<1e-10,errors;checks.append(dict(kind=kind,vocabulary=len(model.embed.weight),errors=errors))
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,contracts=checks,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},scope='Three actual independently adapting readouts share a native carrier; summed losses retain per-world gradients and independent Adam coordinates. Native separate-world scores/gradients/final vectors/moments including EOS reset contracted for23 and50257 token vocabularies. No policy fit/public scoring/TEST; original pretrained source and all verification work charged separately.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
