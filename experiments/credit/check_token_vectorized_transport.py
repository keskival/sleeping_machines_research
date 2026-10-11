"""R1 fixed-version native forecast/state/all-parameter-gradient execution parity."""
import argparse,hashlib,json,resource,time
from copy import deepcopy
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM,load,np
from token_episode_state import consume as old
from token_episode_state_vectorized import consume as new
SOURCES=['experiments/credit/check_token_vectorized_transport.py','experiments/credit/token_episode_state_vectorized.py','experiments/credit/token_forward_credit_vectorized.py','experiments/credit/token_episode_state.py','experiments/credit/token_forward_credit.py','experiments/r1_token_keyed_lm.py','experiments/tpp/race_tpp_v5.py']

def run(model,tokens,fn,eos):
    state=None;losses=[];forecasts=[]
    for j in range(len(tokens)-1):
        if int(tokens[j])==eos:state=None
        forecast,state=fn(model,tokens[j:j+1],state,parameter_version=0)
        forecasts.append(forecast['log_probabilities'].detach())
        losses.append(-forecast['log_probabilities'][0,tokens[j+1]])
    objective=torch.stack(losses).mean()
    gradients=torch.autograd.grad(objective,tuple(model.parameters()),allow_unused=True)
    return torch.cat(forecasts),state,gradients


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();checks=[]
    for kind in ['small2','small4','small8','proper_gpt2']:
        if kind.startswith('small'):
            depth=int(kind[-1]);torch.manual_seed(1100+depth);model=KeyedTokenLM(8,3,depth,2,4,1,0.);model.embed=torch.nn.Embedding(23,8);model.bias=torch.nn.Parameter(torch.randn(23)*.02);tokens=torch.tensor([2,4,6,8,4,2,7,8,4,2,6,8]);eos=7
        else:
            source=json.loads((ROOT/SOURCE).read_text());c=source['args'];model=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);model.load_state_dict(torch.load(ROOT/source['checkpoint'],weights_only=True));data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');offset=c['train_tokens']+13000;tokens=torch.tensor(np.array(data[offset:offset+8],dtype=np.int64));eos=50256
        model.eval();left,state,g=run(model,tokens,old,eos);right,other,gg=run(model,tokens,new,eos)
        errors=dict(forecasts=float((left-right).abs().max()),gradients=max([float((x-y).abs().max()) for x,y in zip(g,gg) if x is not None]+[0.]),layers=max(float((x-y).abs().max()) for x,y in zip(state['layers'],other['layers'])),keys=max(float((v[0]-other['keys'][k][0]).abs().max()) for k,v in state['keys'].items()))
        assert [x is None for x in g]==[x is None for x in gg] and max(errors.values())<1e-10,errors
        checks.append(dict(kind=kind,vocabulary=len(model.embed.weight),errors=errors))
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,contracts=checks,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},scope='Same fixed parameter version/dropout-off native temporal/token/key equations, vectorized exp/transport over active lazy key slots. Full normalized predictions/state and all used native parameter gradients, including producer/key rates and EOS, contracted at small depth2/4/8 and actual50257 vocabulary. All active keys still scored and dense vocabulary output retained. No fit/quality/hardware or changing-key-rate lazy-semantics claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
