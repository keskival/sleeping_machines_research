"""B10 exact source acquisition with compact own-clock context derivatives."""
import argparse,copy,json
from market_native_early_request import ROOT,torch,np,restored,load,warm,window,control,step,sha


def clock_context_window(model,state,t,m,start,group):
    trace=[np.zeros((l.n,2,2)) for l in model.layers]
    totals=np.zeros(2*len(model.layers));losses=[];scores=[]
    for k in range(start,start+group):
        elapsed=float((t[k]-state['last_time'])[0])/model.scale
        old=[z.detach().numpy()[0].reshape(2,l.n) for l,z in zip(model.layers,state['layers'])]
        forecast,end=control.market_consume(model,t[k:k+1],m[k:k+1],state)
        loss=-control.revealed_log_probability(model,forecast,t[k+1:k+2],m[k+1:k+2]).mean()
        cotangents=torch.autograd.grad(loss/group,tuple(end['layers']),retain_graph=True)
        for j,(layer,(real,imag),cot) in enumerate(zip(model.layers,old,cotangents)):
            rate=np.exp(layer.log_rate.detach().numpy());angle=layer.freq.detach().numpy()*elapsed
            decay=np.exp(-rate*elapsed);cr=decay*np.cos(angle);ci=decay*np.sin(angle)
            rr=cr*real-ci*imag;ri=ci*real+cr*imag
            sr,si=trace[j][:,0,:],trace[j][:,1,:]
            updated=np.stack((cr[:,None]*sr-ci[:,None]*si,ci[:,None]*sr+cr[:,None]*si),1)
            updated[:,0,0]+=-elapsed*rate*rr;updated[:,1,0]+=-elapsed*rate*ri
            updated[:,0,1]+=-elapsed*ri;updated[:,1,1]+=elapsed*rr
            trace[j]=updated
            c=cot.detach().numpy()[0].reshape(2,layer.n).T
            totals[2*j:2*j+2]+=(c[:,:,None]*updated).sum((0,1))/np.sqrt(layer.n)
        losses.append(loss);scores.append(float(loss.detach()));state=end
    params=tuple(model.parameters());grads=torch.autograd.grad(torch.stack(losses).mean(),params,allow_unused=True)
    return control.detach_state(state),[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,grads)],scores,totals


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);args=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    model,inputs=restored();opt=torch.optim.Adam(model.parameters(),lr=.0003)
    t,m,path=load('2026-08-24',128+3*32);inputs[str(path.relative_to(ROOT))]=sha(path)
    state=warm(model,t,m,128);rows=[]
    for block in range(3):
        old=copy.deepcopy(state);k=127+32*block
        end,exact,scores,projection=clock_context_window(model,state,t,m,k,32)
        _,own,reference_scores=window(model,copy.deepcopy(old),t,m,k,32,'own')
        _,bp,bp_scores=window(model,copy.deepcopy(old),t,m,k,32,'bptt')
        by=dict(zip((n for n,_ in model.named_parameters()),own))
        expected=np.array([float(by[f'layers.{j}.{key}'].sum())/np.sqrt(l.n)
                           for j,l in enumerate(model.layers) for key in ['log_rate','freq']])
        derivative=max(float((a-b).abs().max()) for a,b in zip(exact,bp))
        context_error=float(np.max(np.abs(projection-expected)))
        score=max(abs(a-b) for a,b in zip(scores,reference_scores))
        assert max(derivative,context_error,score)<1e-12
        rows.append(dict(block=block,exact_actor_gradient_max_error=derivative,
                         own_clock_projection_max_error=context_error,forecast_max_error=score))
        step(model,opt,exact);state=end
    files=['experiments/credit/market_clock_context_contract.py','experiments/credit/market_streaming_clock_updates.py',
           'experiments/market/local_eligibility_state.py','experiments/credit/local_clock_eligibility.py']
    result=dict(status='completed',battle='B10',rows=rows,clock_context_trace_floats=sum(4*l.n for l in model.layers),
        original_own_trace_floats=7280,targets=96,actual_actor_updates=3,
        source_sha256={p:sha(ROOT/p) for p in files},input_sha256=inputs,
        scope='Exact actor BPTT acquisition plus128float factorized own-clock context traces replaces need for7280float own-parameter bank on unaudited source windows only. Current-event state cotangents retain factual graph; actual32target updates and four ownclock projections compared with frozen implementations. Source BPTT graph is still retained; proposed teacher/source execution repair, not deployment replacement, complete-work saving or new forecast-quality claim. Audited proposal windows still need full own/projected endpoints.')
    out=ROOT/f'experiments/results/market/{args.tag}.json';assert not out.exists()
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
