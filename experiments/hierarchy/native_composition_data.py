"""Raw native FIT-prefix preparation and stochastic-preserving composition checks."""
import argparse,csv,hashlib,json,resource,struct,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sleeping_machines.native_memory_composition import NativeMemoryComposition

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_hashes():
 files={str(Path(m.__file__).resolve().relative_to(ROOT)) for name,m in tuple(sys.modules.items()) if name.startswith('sleeping_machines') and getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(ROOT))}
 files.add('experiments/hierarchy/native_composition_data.py')
 return {n:sha(ROOT/n) for n in sorted(files)}

def read_prefixes(path,limit=32):
 with path.with_name(path.stem+'_labels.csv').open() as f:labels=list(csv.DictReader(f))
 rows=[[] for _ in labels]
 with path.open('rb') as f:
  while True:
   line=f.readline(4096)
   if line.strip()==b'#!END-HEADER':break
   if not line or not line.startswith(b'#'):raise ValueError('bad AEDAT header')
  while header:=f.read(28):
   if len(header)!=28:raise ValueError('truncated header')
   kind,_,size,_,overflow,capacity,number,_=struct.unpack('<hhiiiiii',header)
   if min(size,capacity,number)<0 or number>capacity or size*capacity>32*1024*1024:raise ValueError('bad packet')
   raw=f.read(size*capacity)
   if len(raw)!=size*capacity:raise ValueError('truncated packet')
   if kind!=1 or size!=8 or not number:continue
   data=np.frombuffer(raw,dtype='<u4',count=2*number).reshape(-1,2)
   address=data[:,0];t=data[:,1].astype(np.int64)|(np.int64(overflow)<<31)
   x,y,pol=(address>>17)&0x1FFF,(address>>2)&0x1FFF,(address>>1)&1
   for i,label in enumerate(labels):
    if len(rows[i])>=limit:continue
    start,end=int(label['startTime_usec']),int(label['endTime_usec'])
    valid=((address&1)==1)&(t>=start)&(t<min(end,start+1000000))&(x<128)&(y<128)
    for index in np.flatnonzero(valid)[:limit-len(rows[i])]:
     rows[i].append((int(t[index]-start),int(y[index]),int(x[index]),int(pol[index])))
   if all(len(row)==limit for row in rows):break
 if not all(len(row)==limit for row in rows):raise ValueError('insufficient event prefix')
 events=np.array([sorted(row,key=lambda e:e[0]) for row in rows],dtype=np.int64);targets=np.array([int(r['class'])-1 for r in labels],dtype=np.int64)
 return events,targets

def execute(model,events):
 state=model.new_state()
 for microseconds,y,x,pol in events:
  model.observe(state,(int(y),int(x)),float(microseconds)/1e6,[float(pol)*2-1,1.])
 logits,ready=model.compose(state)
 return logits,ready,model.work(state)

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(101);start=time.monotonic()
 directory=ROOT/'data/dvsgesture/DvsGesture';train=directory/'trials_to_train.txt'
 names=sorted(n.strip() for n in train.read_text().splitlines() if n.endswith('.aedat') and 1<=int(n[4:6])<=16)
 path=directory/names[0];events,labels=read_prefixes(path)
 signatures={};conflicts=[]
 for i,event in enumerate(events):
  digest=hashlib.sha256(event.tobytes()).hexdigest()
  if digest in signatures and labels[signatures[digest]]!=labels[i]:conflicts.append([signatures[digest],i])
  signatures[digest]=i
 outdir=ROOT/'experiments/results/hierarchy';outdir.mkdir(parents=True,exist_ok=True)
 cache=outdir/(a.tag+'_raw.npz')
 if cache.exists():raise ValueError('existing raw cache')
 np.savez_compressed(cache,events=events,labels=labels)
 model=NativeMemoryComposition();torch.manual_seed(107)
 t=time.monotonic();model.train();logits,ready,work=execute(model,events[0])
 loss=torch.nn.functional.cross_entropy(logits[None],torch.tensor([int(labels[0])]))
 loss.backward();update_s=time.monotonic()-t
 finite=bool(torch.isfinite(loss)) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 with torch.no_grad():
  torch.manual_seed(107);model.eval();eval_logits,eval_ready,eval_work=execute(model,events[0])
 parity=torch.equal(logits,eval_logits) and torch.equal(ready,eval_ready)
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 histogram=np.bincount(labels,minlength=11);prior=histogram/histogram.sum()
 record=dict(status='completed',battle='G-HIER',stage='native_composition_data',
  eligible_for_learning=finite and parity and not conflicts and rss*1.25<3000000 and update_s*64*1.5<1200,
  source_sha256=source_hashes(),input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [train,path,path.with_name(path.stem+'_labels.csv')]},
  data=dict(path=str(cache.relative_to(ROOT)),sha256=sha(cache),episodes=len(labels),events_per_episode=events.shape[1],unique_input_signatures=len(signatures),cross_label_collisions=conflicts),
  label_histogram=histogram.tolist(),constant_predictor_nll=float(-(prior[prior>0]*np.log(prior[prior>0])).sum()),
  majority_accuracy=float(prior.max()),parameters=sum(p.numel() for p in model.parameters()),
  gradients_finite=finite,train_eval_bitwise_parity=parity,update_s=update_s,max_rss_kb=rss,process_wall_s=time.monotonic()-start,work=work,
  scope='First32 raw events of each gesture in one official FIT recording; annotation boundaries, no absolute recording times or class IDs in model inputs. Native learnability/resource gate, not gesture benchmark.',
  decision='Only admit64updates if finite/parity/identifiability/resources pass. Input uniqueness is necessary, not evidence of useful class features or generalization.')
 with (outdir/(a.tag+'.json')).open('x') as f:json.dump(record,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in record.items() if k not in ['source_sha256','input_sha256']}))
if __name__=='__main__':main()
