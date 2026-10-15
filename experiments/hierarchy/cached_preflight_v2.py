"""Real-TRAIN cached replay admission, no optimizer or fitting."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sleeping_machines.cached_parent_hierarchy import EventCachedParentHierarchy
from experiments.hierarchy.visual_representation import recording
FILES=['sleeping_machines/cached_parent_hierarchy.py','sleeping_machines/addressable_parent_hierarchy.py',
 'sleeping_machines/parent_race_hierarchy.py','sleeping_machines/spatial_race_hierarchy.py',
 'experiments/hierarchy/visual_representation.py','experiments/hierarchy/cached_preflight_v2.py']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(31);start=time.monotonic()
 directory=ROOT/'data/dvsgesture/DvsGesture';train=directory/'trials_to_train.txt'
 names=sorted(n.strip() for n in train.read_text().splitlines() if n.endswith('.aedat') and 1<=int(n[4:6])<=16)
 path=directory/names[0];counts,labels=recording(path)
 outdir=ROOT/'experiments/results/hierarchy';outdir.mkdir(parents=True,exist_ok=True)
 cache=outdir/(a.tag+'_fit_recording.npz')
 if cache.exists():raise ValueError('existing cache')
 np.savez_compressed(cache,counts=counts,labels=np.array([r[0] for r in labels],dtype=np.int64))
 events=torch.tensor(counts[:2],dtype=torch.float32);target=torch.tensor([r[0] for r in labels[:2]])
 model=EventCachedParentHierarchy(width=32,levels=8,refinements=4,roles=4)
 site,q=model.sample_audit_site(events,torch.Generator().manual_seed(37))
 t=time.monotonic();loss,info=model.route_objective(events,target,site,seed=41,site_probability=q)
 loss.backward();update_s=time.monotonic()-t
 gradients={n:float(v.grad.abs().sum()) for n,v in model.named_parameters() if v.grad is not None}
 finite=bool(torch.isfinite(loss)) and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
 output=info.pop('output');index=info['audited_cell'];audit=output['audits'][index]
 # Check one forced alternative at this same real-input audit position.
 offset=site-sum(x['winners'].shape[1]*x['winners'].shape[2] for x in output['audits'][:index])
 location,role=divmod(offset,model.roles)
 candidate=int(audit['valid'][0,location].nonzero().flatten()[-1])
 with torch.no_grad():
  suffix,_=model.suffix_logits(output,index,location,role,candidate)
  full=model(events,seed=41,force=(index,location,role,candidate))
  parity=torch.equal(suffix,full['logits'])
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 eligible=finite and parity and rss*1.25<3000000 and update_s*64*1.5<5400
 sources={n:sha(ROOT/n) for n in FILES}
 record=dict(status='completed',battle='G-HIER',stage='cached_preflight',eligible_for_probe=eligible,
  scope='One official FIT recording,128x128 fixed first1s; no optimizer or benchmark score. Probe gate only.',
  data=dict(path=str(cache.relative_to(ROOT)),sha256=sha(cache),episodes=len(labels)),
  input_sha256={str(f.relative_to(ROOT)):sha(f) for f in [train,path,path.with_name(path.stem+'_labels.csv')]},
  source_sha256=sources,max_rss_kb=rss,process_wall_s=time.monotonic()-start,update_s=update_s,
  gradients_finite=finite,gradient_l1=gradients,forced_suffix_full_bitwise_equal=parity,
  replay=dict(replays=info['replays'],audited_cell=index,suffix_cell_evaluations=info['suffix_cell_evaluations'],
    full_replay_cell_evaluations=info['full_replay_cell_evaluations']),resource_accounting=model.work_descriptor(128,128))
 with (outdir/(a.tag+'.json')).open('x') as f:json.dump(record,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in record.items() if k not in ['gradient_l1','source_sha256','input_sha256','resource_accounting']}))
if __name__=='__main__':main()
