"""Bounded raw-TRAIN native temporal-object resource check, no optimizer."""
import argparse,csv,hashlib,json,resource,struct,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sleeping_machines.native_object_hierarchy import NativeObjectHierarchy

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raw_prefix(path,limit=32):
 with path.with_name(path.stem+'_labels.csv').open() as f:label=next(csv.DictReader(f))
 start,end=int(label['startTime_usec']),int(label['endTime_usec'])
 rows=[]
 with path.open('rb') as f:
  while True:
   line=f.readline(4096)
   if line.strip()==b'#!END-HEADER':break
   if not line or not line.startswith(b'#'):raise ValueError('bad AEDAT header')
  while header:=f.read(28):
   if len(header)!=28:raise ValueError('truncated packet header')
   kind,_,size,_,overflow,capacity,number,_=struct.unpack('<hhiiiiii',header)
   if min(size,capacity,number)<0 or number>capacity or size*capacity>32*1024*1024:raise ValueError('bad packet size')
   raw=f.read(size*capacity)
   if len(raw)!=size*capacity:raise ValueError('truncated packet')
   if kind!=1 or size!=8 or not number:continue
   data=np.frombuffer(raw,dtype='<u4',count=2*number).reshape(-1,2)
   addr=data[:,0];t=data[:,1].astype(np.int64)|(np.int64(overflow)<<31)
   x,y,pol=(addr>>17)&0x1FFF,(addr>>2)&0x1FFF,(addr>>1)&1
   valid=((addr&1)==1)&(t>=start)&(t<min(end,start+1000000))&(x<128)&(y<128)
   for index in np.flatnonzero(valid):
    rows.append(((int(y[index]),int(x[index])),float(t[index]-start)/1e6,[float(pol[index])*2-1,1.]))
    if len(rows)==limit:return rows,int(label['class'])-1
 raise ValueError('insufficient raw events')

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(73);start=time.monotonic()
 directory=ROOT/'data/dvsgesture/DvsGesture';train=directory/'trials_to_train.txt'
 names=sorted(n.strip() for n in train.read_text().splitlines() if n.endswith('.aedat') and 1<=int(n[4:6])<=16)
 path=directory/names[0];events,label=raw_prefix(path)
 model=NativeObjectHierarchy();state=model.new_state()
 for address,t,content in events:model.observe(state,address,t,content)
 cutoff=max(t for _,t,_ in events)+len(events)*8*4*.011+.1
 while state.queue:
  if state.queue[0][0]>cutoff:raise ValueError('unexpected native completion bound')
  model.advance(state,state.queue[0][0])
 cutoff=state.now
 target=torch.tensor([label]);loss=torch.nn.functional.cross_entropy(model.logits(state)[None],target)
 loss.backward()
 finite=bool(torch.isfinite(loss)) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 # Exercise actual indexed query and autonomous-in-absence-of-input emission.
 source=next(iter(state.banks[0].published))
 model.query(state,0,source,7,(0,0));model.advance(state,cutoff+.1)
 model.arm(state,0,source);model.advance(state,cutoff+.6)
 files={str(Path(m.__file__).resolve().relative_to(ROOT)) for name,m in tuple(sys.modules.items()) if name.startswith('sleeping_machines') and getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(ROOT))}
 files.add('experiments/hierarchy/native_object_preflight.py')
 result=dict(status='completed',battle='G-HIER',stage='native_object_preflight',
  scope='32 raw events, first FIT gesture, native32-stage hierarchy, no fit; diagnostic loss is not a benchmark score',
  source_sha256={n:sha(ROOT/n) for n in sorted(files)},
  input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [train,path,path.with_name(path.stem+'_labels.csv')]},
  raw_events=len(events),parameters=sum(p.numel() for p in model.parameters()),gradients_finite=finite,
  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,process_wall_s=time.monotonic()-start,
  work=model.work(state),pending_events=len(state.queue),virtual_time=state.now,
  next_decision='Resource and credit coverage decide raw-event learning admission; no automatic fit or hardware-time claim')
 out=ROOT/f'experiments/results/hierarchy/{a.tag}.json';out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
