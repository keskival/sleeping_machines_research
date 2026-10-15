"""Match geometric counterparts using TRAIN sensor arrays and labels only.

No latent part IDs, tree codes or split metadata enter pairing. Every partner
must already exist in the supplied TRAIN set; no new label is manufactured.
"""
import hashlib
import numpy as np


def swap_observed_halves(events):
    events=np.asarray(events,dtype=np.int64)
    x=events[:,2];mid=(int(x.min())+int(x.max()))/2
    left=x<mid;right=~left
    if not left.any() or not right.any():raise ValueError('both observed halves required')
    distance=float(x[right].mean()-x[left].mean())
    if abs(distance-round(distance))>1e-9:raise ValueError('this task requires an exact integer counterpart')
    result=events.copy();result[left,2]+=round(distance);result[right,2]-=round(distance)
    return result


def training_pairs(examples):
    # examples are (anonymous sensor array,whole label), not generator Scenes.
    digest=lambda x:hashlib.sha256(np.asarray(x,dtype=np.int64).tobytes()).hexdigest()
    lookup={digest(events):i for i,(events,_) in enumerate(examples)}
    if len(lookup)!=len(examples):raise ValueError('duplicate training input')
    used=set();pairs=[]
    for i,(events,label) in enumerate(examples):
        if i in used:continue
        partner=lookup.get(digest(swap_observed_halves(events)))
        if partner is None or partner==i or partner in used:
            raise ValueError('observed counterpart missing from TRAIN')
        if label==examples[partner][1]:raise ValueError('counterpart labels are not opposite')
        used.update((i,partner));pairs.append((i,partner))
    assert len(used)==len(examples)
    return pairs
