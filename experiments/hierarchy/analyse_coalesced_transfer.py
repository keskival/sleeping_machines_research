"""Reuse cached representations to diagnose held-out composition failure."""
import hashlib,json
from pathlib import Path
import numpy as np
from experiments.hierarchy.part_combination_task import corpus
ROOT=Path(__file__).resolve().parents[2]

def main():
    source=ROOT/'experiments/results/hierarchy/aws_g_hier_coalesced_readout_20261015_v1_features.npz'
    rows=corpus();counts={}
    for split in ('train','dev','test'):
        for n in sorted({r.nuisance for r in rows if r.split==split}):
            totals=[]
            for r in rows:
                if r.split!=split or r.nuisance!=n:continue
                xy=r.events[:,1:3]
                totals.append(sum(len(np.unique(xy//2**(l+1),axis=0)) for l in range(8)))
            counts[f'{split}_nuisance{n}']=dict(min_objects=min(totals),max_objects=max(totals))
    statistics={}
    with np.load(source) as data:
        for variant in ('initial','final'):
            x=data[variant+'_train'].astype(float);v=data[variant+'_dev'].astype(float)
            sd=np.maximum(x.std(0),1e-6);mean_shift=(v.mean(0)-x.mean(0))/sd
            within=[]
            # DEV order: combination,root,nuisance2 then3. Pair distances change
            # nuisance while holding the whole arrangement and label fixed.
            nuisance_distance=np.linalg.norm(v[::2]-v[1::2],axis=1).mean()
            # Root relation pair changes label, holds both child codes/nuisance.
            relation_distance=np.linalg.norm(v.reshape(-1,2,2,v.shape[1])[:,0]-v.reshape(-1,2,2,v.shape[1])[:,1],axis=-1).mean()
            statistics[variant]=dict(train_std_mean=float(x.std(0).mean()),dev_std_mean=float(v.std(0).mean()),
                standardized_dev_mean_shift_rms=float(np.sqrt(np.mean(mean_shift**2))),
                nuisance_pair_distance=float(nuisance_distance),root_relation_pair_distance=float(relation_distance))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    out=ROOT/'experiments/results/hierarchy/aws_g_hier_coalesced_transfer_diagnosis_20261015_v1.json'
    packet=dict(status='completed',battle='G-HIER',statistics=statistics,geometric_occupancy=counts,
        input_sha256={str(source.relative_to(ROOT)):sha(source)},source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__)),
        'experiments/hierarchy/part_combination_task.py':sha(ROOT/'experiments/hierarchy/part_combination_task.py')},
        scope='Cached root features and generator geometry only. No refitting, native execution or TEST model scoring. Distances do not independently identify a causal bottleneck.',
        decision='Repair nuisance/role binding and audit intermediate-state information before any larger fit. Keep measured resource advance separate from failed short-fit transfer.')
    with out.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
    print(json.dumps(dict(statistics=statistics,geometric_occupancy=counts),indent=1))
if __name__=='__main__':main()
