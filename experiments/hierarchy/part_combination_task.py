"""Anonymous event scenes with balanced, held-out recursive arrangements.

Latent tree codes are audit metadata only. The model interface returns sensor
(time, y, x, polarity) and one whole-scene label, never constituent identities.
"""
from dataclasses import dataclass
import hashlib
import itertools
import numpy as np


@dataclass(frozen=True)
class Scene:
    events: np.ndarray
    label: int
    split: str
    left: int
    right: int
    root_swap: int
    nuisance: int

    def model_input(self):
        return self.events.copy(), self.label

    @property
    def digest(self):
        return hashlib.sha256(self.events.tobytes()).hexdigest()


def split_for(left,right):
    # Both children have every code in each split. TRAIN differences 1,3,7
    # span all three binary relation dimensions; no relation bit is absent.
    # Every split has both even and odd difference parity. Partition pairs only.
    difference=left ^ right
    return 'train' if difference in (0,1,3,7) else ('dev' if difference in (2,6) else 'test')


def render(left,right,root_swap,nuisance=0,split=None):
    """Three binary composition levels; eight distinguishable four-event parts.

All seven arrangement bits contribute to the whole label. Each scene and its
root-swapped partner use exactly the same timestamps, polarities and parts.
No early prefix is promoted to a task with assumed complete evidence.
"""
    if not (0<=left<8 and 0<=right<8 and root_swap in (0,1)):
        raise ValueError('three-bit child codes and one root bit required')
    bits=[root_swap,(left>>2)&1,(right>>2)&1,
          (left>>1)&1,left&1,(right>>1)&1,right&1]
    placements={}
    def place(node,depth,identities,y,x,height,width):
        if depth==3:
            placements[identities[0]]=(y,x,height,width);return
        half=len(identities)//2;groups=[identities[:half],identities[half:]]
        if bits[node]:groups.reverse()
        # Alternate spatial axes. Parent role is relative position, not time.
        if depth%2==0:
            step=width//2
            place(2*node+1,depth+1,groups[0],y,x,height,step)
            place(2*node+2,depth+1,groups[1],y,x+step,height,width-step)
        else:
            step=height//2
            place(2*node+1,depth+1,groups[0],y,x,step,width)
            place(2*node+2,depth+1,groups[1],y+step,x,height-step,width)
    place(0,0,list(range(8)),8,8,112,112)
    rng=np.random.default_rng(1771+nuisance)
    translation=rng.integers(-3,4,size=2)
    # A shared nuisance tape across every arrangement; not a per-label clock.
    permutation=rng.permutation(32);times=np.cumsum(rng.integers(50,151,size=32))
    events=[]
    for identity,(y,x,h,w) in placements.items():
        # Three positions encode part identity through shape; two + and two -.
        offsets=[(1,1),(1,6),(6,1),(3+identity//4,2+identity%4)]
        for j,(dy,dx) in enumerate(offsets):
            serial=identity*4+j
            events.append((int(times[permutation[serial]]),int(y+h//2-4+dy+translation[0]),
                           int(x+w//2-4+dx+translation[1]),1 if j<2 else -1))
    events=np.array(sorted(events),dtype=np.int64)
    return Scene(events,sum(bits)%2,split or split_for(left,right),left,right,root_swap,nuisance)


def scenes(nuisances=range(2)):
    return [render(a,b,r,n) for a,b,r,n in itertools.product(range(8),range(8),range(2),nuisances)]


def corpus():
    # Withhold both combinations and independently generated nuisance tapes.
    rows=[]
    for a,b,r in itertools.product(range(8),range(8),range(2)):
        split=split_for(a,b);base={'train':0,'dev':2,'test':4}[split]
        rows.extend(render(a,b,r,n) for n in range(base,base+2))
    return rows


def audit(rows):
    digests={};summary={}
    for row in rows:
        previous=digests.setdefault(row.digest,row)
        if previous is not row:raise ValueError('duplicate anonymous stream')
    for split in ('train','dev','test'):
        part=[r for r in rows if r.split==split]
        assert part and {r.left for r in part}==set(range(8)) and {r.right for r in part}==set(range(8))
        assert sum(r.label for r in part)*2==len(part)
        # Exact paired nuisance distribution, event counts, polarity totals and
        # timestamps; absolute placement jitter has the same distribution.
        for n in {r.nuisance for r in part}:
            subset=[r for r in part if r.nuisance==n]
            assert sum(r.label for r in subset)*2==len(subset)
            signatures={tuple(r.events[:,0]) for r in subset};assert len(signatures)==1
            assert all(len(r.events)==32 and int(r.events[:,3].sum())==0 for r in subset)
        # Even a complete one-child code together with root orientation cannot
        # predict the label: the missing other-child relation is necessary.
        for code in range(8):
            for root in (0,1):
                for side in ('left','right'):
                    subset=[r for r in part if getattr(r,side)==code and r.root_swap==root]
                    assert sum(r.label for r in subset)*2==len(subset)
        summary[split]=dict(scenes=len(part),positive=sum(r.label for r in part),child_codes=8,
                            combinations=len({(r.left,r.right) for r in part}))
    combinations=[{(r.left,r.right) for r in rows if r.split==s} for s in ('train','dev','test')]
    assert all(not a&b for a,b in itertools.combinations(combinations,2))
    return summary
