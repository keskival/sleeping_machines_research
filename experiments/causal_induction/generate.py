"""B5/R1 development data: hidden sparse causal circuits, public observations only.
Oracle structure is stored separately and is forbidden as learner supervision.
"""
import argparse,heapq,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def graph(rng,depth,width=4):
    parents=[[] for _ in range(width)];bias=[0]*width;delays=[[] for _ in range(width)];signature=[1<<i for i in range(width)]
    for layer in range(1,depth+1):
        previous=np.arange((layer-1)*width,layer*width)
        for _ in range(width):
            for trial in range(32):
                count=int(rng.integers(1,3));p=rng.choice(previous,count,replace=False).tolist();mask=0
                for j in p:mask^=signature[j]
                if mask:break
            if not mask:p=[int(previous[0])];mask=signature[p[0]]
            parents.append(p);signature.append(mask);bias.append(int(rng.integers(2)));delays.append(rng.uniform(.01,.04,len(p)).tolist())
    return dict(parents=parents,bias=bias,delays=delays,settled_root_parity=signature)

def episode(seed,depth,pulses=64,padding=0,scale=1.,rename_seed=0,switch=False):
    rng=np.random.default_rng(seed);g=graph(rng,depth);active=4*(depth+1);capacity=active+padding;names=np.random.default_rng(rename_seed).permutation(capacity)[:active];state=np.zeros(active,dtype=np.int8);inbox=[np.zeros(len(p),dtype=np.int8) for p in g['parents']];children=[]
    def links():
        out=[[] for _ in range(active)]
        for child,ps in enumerate(g['parents']):
            for slot,parent in enumerate(ps):out[parent].append((child,slot,g['delays'][child][slot]))
        return out
    children=links();events=[];pending=[];serial=0;queries=[];oracle=[dict(pulse=0,graph=g)];period=.12*(depth+1)
    def emit(t,node,value,external):
        nonlocal serial
        changed=int(state[node])!=value;state[node]=value
        if changed or external:events.append((t,int(names[node]),value,int(external)))
        if changed:
            for child,slot,delay in children[node]:serial+=1;heapq.heappush(pending,(t+delay,serial,child,slot,value))
    def flush(until):
        while pending and pending[0][0]<=until:
            t,_,node,slot,value=heapq.heappop(pending);inbox[node][slot]=value
            new=int(g['bias'][node])
            for bit in inbox[node]:new^=int(bit)
            emit(t,node,new,False)
    # Initial zero state is an explicit public convention, not a settled gate state.
    for pulse in range(pulses):
        t=pulse*period;flush(t)
        if switch and pulse==pulses//2:
            g=graph(rng,depth);children=links();inbox=[np.array([state[p] for p in ps],dtype=np.int8) for ps in g['parents']];oracle.append(dict(pulse=pulse,graph=g))
        root=int(rng.integers(4));emit(t,root,1-int(state[root]),True)
        issued=t+.005;query=int(rng.integers(4,active));horizon=float(rng.uniform(.01,.05*depth));queries.append((issued,int(names[query]),horizon,pulse))
        flush(t+period-.0001)
    event=np.array(sorted(events),dtype=np.float64);query=np.array(queries,dtype=np.float64);feedback=[]
    # Read observable node state at delivery; no parent/gate field enters public data.
    for issued,node,horizon,qid in queries:
        delivered=issued+horizon;mask=(event[:,0]<=delivered)&(event[:,1]==node);value=int(event[mask][-1,2]) if mask.any() else 0;feedback.append((delivered,qid,value))
    event[:,0]*=scale;query[:,0]*=scale;query[:,2]*=scale;feedback=np.array(feedback,dtype=np.float64);feedback[:,0]*=scale
    return dict(events=event,queries=query,feedback=feedback,capacity=np.array(capacity,dtype=np.int64)),dict(seed=seed,depth=depth,active_nodes=active,capacity=capacity,names=names.tolist(),phases=oracle,private_only=True)

def contract():
    base,_=episode(73,4,pulses=24);renamed,_=episode(73,4,pulses=24,rename_seed=99);scaled,_=episode(73,4,pulses=24,scale=10.);padded,_=episode(73,4,pulses=24,padding=100)
    assert np.array_equal(base['feedback'][:,2],renamed['feedback'][:,2]);assert np.array_equal(base['feedback'][:,2],scaled['feedback'][:,2]);assert np.array_equal(base['feedback'][:,2],padded['feedback'][:,2])
    assert np.allclose(base['events'][:,0]*10,scaled['events'][:,0]);assert (base['feedback'][:,0]>=base['queries'][:,0]).all()
    early,_=episode(73,4,pulses=24,switch=False);late,_=episode(73,4,pulses=24,switch=True);boundary=12*.12*5
    assert np.array_equal(early['events'][early['events'][:,0]<boundary],late['events'][late['events'][:,0]<boundary]);assert np.array_equal(early['feedback'][early['feedback'][:,0]<boundary],late['feedback'][late['feedback'][:,0]<boundary])
    return dict(relabel_target_parity=True,time_rescale_target_parity=True,inactive_capacity_target_parity=True,future_rewire_prefix_causality=True,oracle_not_in_public_fields=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['contract','prepare'],required=True);a=ap.parse_args();out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);metrics=contract();records=[]
    if a.mode=='prepare':
        folder=ROOT/'data/causal_induction'/a.tag;folder.mkdir(parents=True,exist_ok=False)
        for split,seeds,depth in [('train',range(1000,1032),4),('heldout_graph',range(2000,2008),4),('heldout_depth',range(3000,3008),8)]:
            for seed in seeds:
                for variant,options in [('base',{}),('relabel',dict(rename_seed=99)),('rescale',dict(scale=10.)),('capacity',dict(padding=128)),('rewire',dict(switch=True))]:
                    if split=='train' and variant!='base':continue
                    public,private=episode(seed,depth,**options);name=f'{split}_{seed}_{variant}';path=folder/f'{name}.npz';np.savez(path,**public);private_path=folder/f'{name}.oracle.json';private_path.write_text(json.dumps(private)+'\n');records.append(dict(split=split,variant=variant,path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),events=len(public['events']),queries=len(public['queries']),capacity=int(public['capacity'])))
    result=dict(status='completed',battle='B5/R1 causal structural induction development',tag=a.tag,mode=a.mode,contracts=metrics,records=records,source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},scope='Data/semantics only; no learned-bias, Transformer, task-quality or hardware claim. Hidden parent/gate labels are private evaluation diagnostics and forbidden as fitting or selection inputs. Public feedback is delivered after request issuance. Root flips are observed interventions, not parent labels.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(contracts=metrics,episodes=len(records))),flush=True)
if __name__=='__main__':main()
