"""Broader B5/R1 structural data: variable factors, nonlinear gates, foreign events."""
import argparse,hashlib,heapq,json
from pathlib import Path
import numpy as np
from native_sensor import ROOT

def make_graph(rng,depth,width):
 parents=[[] for _ in range(width)];tables=[[] for _ in range(width)];delays=[[] for _ in range(width)]
 nonlinear=[[0,0,0,1],[0,1,1,1],[1,1,1,0],[1,0,0,0]]
 for layer in range(1,depth+1):
  previous=np.arange((layer-1)*width,layer*width)
  for _ in range(width):
   ps=rng.choice(previous,2,replace=False).tolist();table=([0,1,1,0] if rng.random()<.5 else nonlinear[int(rng.integers(len(nonlinear)))])
   if rng.random()<.5:table=[1-x for x in table]
   parents.append(ps);tables.append(table);delays.append(rng.uniform(.01,.04,2).tolist())
 return dict(parents=parents,tables=tables,delays=delays)

def episode(seed,depth,width,pulses=128,scale=1.,padding=0,rename_seed=0,switch=False):
 rng=np.random.default_rng(seed);g=make_graph(rng,depth,width);active=width*(depth+1);capacity=active+padding;names=np.random.default_rng(rename_seed).permutation(capacity)[:active];state=np.zeros(active,dtype=np.int8);inbox=[np.zeros(len(p),dtype=np.int8) for p in g['parents']];pending=[];events=[];queries=[];serial=0;phases=[dict(pulse=0,graph=g)];period=.12*(depth+1)
 def links():
  result=[[] for _ in range(active)]
  for child,ps in enumerate(g['parents']):
   for slot,parent in enumerate(ps):result[parent].append((child,slot,g['delays'][child][slot]))
  return result
 children=links()
 def emit(t,node,value,ext=False):
  nonlocal serial
  change=state[node]!=value;state[node]=value
  if change or ext:events.append((t,int(names[node]),value,int(ext)))
  if change:
   for child,slot,delay in children[node]:serial+=1;heapq.heappush(pending,(t+delay,serial,child,slot,value))
 def flush(until):
  while pending and pending[0][0]<=until:
   t,_,node,slot,value=heapq.heappop(pending);inbox[node][slot]=value;index=int(inbox[node][0])+2*int(inbox[node][1]);emit(t,node,g['tables'][node][index])
 for pulse in range(pulses):
  t=period*pulse;flush(t)
  if switch and pulse==pulses//2:
   assert not pending;g=make_graph(rng,depth,width);children=links();inbox=[np.array([state[p] for p in ps],dtype=np.int8) for ps in g['parents']];phases.append(dict(pulse=pulse,graph=g))
  root=int(rng.integers(width));emit(t,root,1-int(state[root]),True);queries.append((t+.005,int(names[int(rng.integers(width,active))]),float(rng.uniform(.01,.05*depth)),pulse));flush(t+period-.0001)
 ev=np.array(sorted(events),dtype=np.float64);qs=np.array(queries,dtype=np.float64);feedback=[]
 for issued,node,horizon,qid in queries:
  delivered=issued+horizon;subset=ev[(ev[:,0]<=delivered)&(ev[:,1]==node)];feedback.append((delivered,qid,int(subset[-1,2]) if len(subset) else 0))
 fb=np.array(feedback,dtype=np.float64);ev[:,0]*=scale;qs[:,0]*=scale;qs[:,2]*=scale;fb[:,0]*=scale
 return dict(events=ev,queries=qs,feedback=fb,capacity=np.array(capacity)),dict(seed=seed,width=width,depth=depth,names=names.tolist(),phases=phases,private_only=True)

def foreign(home,other):
 result={k:v.copy() for k,v in home.items()};ev=other['events'].copy();ev[:,1]+=int(home['capacity']);ev[:,0]+=.007;result['events']=np.array(sorted(np.concatenate((home['events'],ev)).tolist()));result['capacity']=np.array(int(home['capacity'])+int(other['capacity']));return result

def contracts():
 b,private=episode(411,4,8,pulses=32);ren,_=episode(411,4,8,pulses=32,rename_seed=53);sc,_=episode(411,4,8,pulses=32,scale=10);pad,_=episode(411,4,8,pulses=32,padding=128);rw,_=episode(411,4,8,pulses=32,switch=True);oth,_=episode(912,4,8,pulses=32);mixed=foreign(b,oth)
 for variant in [ren,sc,pad,mixed]:assert np.array_equal(b['feedback'][:,2],variant['feedback'][:,2])
 assert np.array_equal(b['queries'],mixed['queries']) and np.array_equal(b['feedback'],mixed['feedback']);boundary=16*.12*5
 assert np.array_equal(b['events'][b['events'][:,0]<boundary],rw['events'][rw['events'][:,0]<boundary]);assert (b['feedback'][:,0]>b['queries'][:,0]).all()
 tables=private['phases'][0]['graph']['tables'][8:];assert any(t[0]-t[1]-t[2]+t[3]!=0 and t not in [[0,1,1,0],[1,0,0,1]] for t in tables)
 return dict(nonlinear_local_gates_present=True,rename_scale_capacity_target_invariance=True,foreign_independent_events_preserve_queries_and_targets=True,future_rewire_prefix_causality=True,strict_delayed_feedback=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists();checks=contracts();folder=ROOT/'data/causal_induction'/a.tag;folder.mkdir(parents=True,exist_ok=False);rows=[]
 for split,seeds,depth,width in [('train',range(4000,4016),4,8),('heldout_graph',range(5000,5004),4,8),('heldout_factors',range(6000,6004),4,16),('heldout_depth',range(7000,7004),8,8),('heldout_factors_depth',range(8000,8004),8,16)]:
  for seed in seeds:
   for variant,options in [('base',{}),('relabel',dict(rename_seed=53)),('rescale',dict(scale=10)),('capacity',dict(padding=128)),('rewire',dict(switch=True)),('foreign',{})]:
    if split=='train' and variant!='base':continue
    public,private=episode(seed,depth,width,**options)
    if variant=='foreign':other,_=episode(seed+100000,depth,width);public=foreign(public,other)
    name=f'{split}_{seed}_{variant}';path=folder/f'{name}.npz';np.savez(path,**public);(folder/f'{name}.oracle.json').write_text(json.dumps(private)+'\n');rows.append(dict(split=split,variant=variant,width=width,depth=depth,path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),events=len(public['events']),queries=len(public['queries']),capacity=int(public['capacity']),positive_fraction=float(public['feedback'][:,2].mean())))
 out.write_text(json.dumps(dict(status='completed',battle='B5/R1',tag=a.tag,contracts=checks,records=rows,source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},scope='Data only: hidden nonlinear local mechanisms, factor count and graph depth vary independently; public timestamps/node values/exogenous flips/queries/delayed binary feedback only. Oracle labels separate and forbidden for fitting/selection. Foreign-component observations alter input interleaving but preserve home sensor queries/labels. No learned structural bias, benchmark advantage or Transformer claim.'),indent=2)+'\n');print(json.dumps(dict(episodes=len(rows),contracts=checks)),flush=True)
if __name__=='__main__':main()
