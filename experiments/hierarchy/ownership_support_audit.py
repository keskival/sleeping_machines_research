"""G-HIER: anonymous sensor-component alignment with native geometric supports.

No latent part ids,scene arrangement codes or class labels are read. A cut is
selected from TRAIN spatial MST gaps;DEV checks the resulting structure only.
This is a support diagnostic,not a benchmark or learned ownership policy.
"""
import argparse,json,math,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.native_composition_data import sha

def mst(xy):
 n=len(xy);parents=list(range(n));sizes=[1]*n
 def root(i):
  while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
  return i
 edges=sorted((float(np.linalg.norm(xy[i]-xy[j])),i,j) for i in range(n) for j in range(i))
 chosen=[]
 for d,i,j in edges:
  a,b=root(i),root(j)
  if a==b:continue
  if sizes[a]<sizes[b]:a,b=b,a
  parents[b]=a;sizes[a]+=sizes[b];chosen.append((d,i,j))
  if len(chosen)==n-1:break
 return chosen

def components(n,tree,threshold):
 groups=[{i} for i in range(n)]
 for d,i,j in tree:
  if d>threshold:continue
  a=next(g for g in groups if i in g);b=next(g for g in groups if j in g)
  if a is not b:a.update(b);groups.remove(b)
 return [sorted(g) for g in groups]

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();began=time.monotonic()
 observations=[];cuts=[]
 for row in corpus():
  if row.split=='test':continue
  events,_=row.model_input();xy=events[:,1:3].astype(float);tree=mst(xy)
  lengths=np.array([e[0] for e in tree]);assert (lengths>0).all()
  ratios=lengths[1:]/lengths[:-1];cut=int(np.argmax(ratios))
  if row.split=='train':cuts.append(float(math.sqrt(lengths[cut]*lengths[cut+1])))
  observations.append((row.split,xy,tree))
 threshold=float(np.median(cuts));rows={}
 for split in ['train','dev']:
  levelcounts=[dict(components=0,fragmented=0,mixed_cells=0,cells=0) for _ in range(8)]
  total=never=0;hist={};firstlevels=[]
  for subset,xy,tree in observations:
   if subset!=split:continue
   groups=components(len(xy),tree,threshold);hist[len(groups)]=hist.get(len(groups),0)+1
   total+=len(groups);pure_seen=[False]*len(groups);origin=xy.min(0)
   for level in range(8):
    address=xy.astype(int) if level==0 else np.floor((xy-origin)/2**level).astype(int)
    bycell={}
    for c,g in enumerate(groups):
     cells={tuple(address[i]) for i in g}
     levelcounts[level]['components']+=1;levelcounts[level]['fragmented']+=len(cells)>1
     for cell in cells:bycell.setdefault(cell,set()).add(c)
    levelcounts[level]['cells']+=len(bycell);levelcounts[level]['mixed_cells']+=sum(len(v)>1 for v in bycell.values())
    for c,g in enumerate(groups):
     cells={tuple(address[i]) for i in g}
     if len(cells)==1 and len(bycell[next(iter(cells))])==1 and not pure_seen[c]:pure_seen[c]=True;firstlevels.append(level)
   never+=sum(not x for x in pure_seen)
  rows[split]=dict(scenes=sum(s==split for s,_,_ in observations),component_count_histogram=hist,total_components=total,never_pure_complete_components=never,never_pure_complete_fraction=never/total,first_pure_level_histogram={i:firstlevels.count(i) for i in sorted(set(firstlevels))},by_level=levelcounts)
 result=dict(status='completed',battle='G-HIER',stage='ownership_support_audit',threshold=threshold,training_cut_range=[min(cuts),max(cuts)],rows=rows,
  source_sha256={f:sha(ROOT/f) for f in ['experiments/hierarchy/ownership_support_audit.py','experiments/hierarchy/part_combination_task.py']},
  pair_distance_evaluations=len(observations)*32*31//2,process_wall_s=time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  scope='Anonymous32pixel scenes,TRAIN spatial MST largest adjacent edge-ratio cut and median threshold,DEV structural evaluation. No latent part identity,arrangement bits,class labels or TEST used. Components are observed spatial groups,not proven semantic objects;static fixture diagnostic only.',
  decision='If observed groups never acquire pure complete native supports,repair support/ownership before another unchanged semantic fit;versioned part access is necessary but does not fix fragmented ownership by itself.')
 with (ROOT/f'experiments/results/hierarchy/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(threshold=threshold,rows={s:r['never_pure_complete_fraction'] for s,r in rows.items()})),flush=True)
if __name__=='__main__':main()
