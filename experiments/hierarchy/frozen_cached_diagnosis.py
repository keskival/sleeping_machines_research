"""Reuse completed FIT models to diagnose the failed synchronous learning probe."""
import argparse,hashlib,json,resource,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from sleeping_machines.cached_parent_hierarchy import EventCachedParentHierarchy

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
 probe_path=ROOT/'experiments/results/hierarchy/aws_g_hier_cached_20261015_v2_probe.json';probe=json.loads(probe_path.read_text())
 pre_path=ROOT/'experiments/results/hierarchy/aws_g_hier_cached_20261015_v2_preflight.json';pre=json.loads(pre_path.read_text())
 data=ROOT/pre['data']['path'];assert sha(data)==pre['data']['sha256']
 with np.load(data) as packed:counts=packed['counts'].copy();labels=packed['labels'].copy()
 rows=[];diagnostics=[]
 for name in ['initial','final']:
  torch.manual_seed(31);model=EventCachedParentHierarchy(width=32,levels=8,refinements=4,roles=4)
  if name=='final':model.load_state_dict(torch.load(ROOT/probe['final_weights'],map_location='cpu',weights_only=True))
  for seed in [None,83,89,97]:
   losses=[];correct=0;features=[];null=raw=empty_raw=0
   with torch.no_grad():
    for i in range(len(labels)):
     events=torch.tensor(counts[i:i+1],dtype=torch.float32)
     output=model(events,seed=0 if seed is None else seed,sample=seed is not None)
     losses.append(float(F.cross_entropy(output['logits'],torch.tensor([int(labels[i])]))))
     correct+=int(output['logits'].argmax()==labels[i]);features.append(output['memory']['levels'][-1].mean((-2,-1))[0])
     for audit in output['audits'][-4:]:
      for candidate in audit['winners'][0,0].tolist():
       address=audit['addresses'][candidate]
       if address is None:null+=1
       elif address[0]==0:
        raw+=1;empty_raw+=int(not bool(counts[i,:,:,address[1],address[2]].sum()))
   rows.append(dict(model=name,seed=seed,nll=sum(losses)/len(losses),accuracy=correct/len(labels),
         root_feature_std=float(torch.stack(features).std(0).mean()),root_null_choices=null,
         root_raw_choices=raw,root_empty_raw_choices=empty_raw))
 final_det=next(r for r in rows if r['model']=='final' and r['seed'] is None)
 initial_det=next(r for r in rows if r['model']=='initial' and r['seed'] is None)
 stoch={n:sum(r['nll'] for r in rows if r['model']==n and r['seed'] is not None)/3 for n in ['initial','final']}
 files={str(Path(m.__file__).resolve().relative_to(ROOT)) for name,m in tuple(sys.modules.items()) if name.startswith('sleeping_machines') and getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(ROOT))};files.add('experiments/hierarchy/frozen_cached_diagnosis.py')
 record=dict(status='completed',battle='G-HIER',stage='frozen_cached_diagnosis',rows=rows,
    deterministic_nll_change=final_det['nll']-initial_det['nll'],stochastic_mean_nll_change=stoch['final']-stoch['initial'],
    scope='One FIT recording,12episodes,three fixed noise seeds and argmax; no optimizer, new selection or held-out evidence',
    source_sha256={n:sha(ROOT/n) for n in sorted(files)},input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [probe_path,pre_path,data,ROOT/probe['final_weights']]},
    max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    next_decision='Use source-use, feature variation and stochastic/deterministic changes to target transport/route-credit repair; no unchanged deeper sweep')
 out=ROOT/f'experiments/results/hierarchy/{a.tag}.json'
 with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
 print(json.dumps(record))
if __name__=='__main__':main()
