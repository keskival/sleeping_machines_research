"""Broader observed geometry/time support; original arrangement split unchanged.

Generator metadata defines labels/splits, never model input or pairing. The
finite topology ceiling is64TRAIN arrangements; new geometry is not a new tree.
"""
from dataclasses import replace
import hashlib,itertools
import numpy as np
from .part_combination_task import render,split_for
from .observed_training_pairs import training_pairs

TRAIN_TAPES=tuple(range(1000,1032))
FRESH_DEV_TAPES=(2000,2001)


def metric_scene(left,right,root,nuisance):
    row=render(left,right,root,nuisance)
    # Detect the canonical observed columns/rows,not hidden constituent IDs.
    # Every original four-pixel part has its extrema within an eight-pixel cell.
    events=row.events.copy();x=events[:,2];y=events[:,1]
    def bins(values,count):
        unique=sorted(set(map(int,values)));groups=[]
        for value in unique:
            if not groups or value-groups[-1][-1]>10:groups.append([])
            groups[-1].append(value)
        if len(groups)!=count:raise ValueError('canonical observed grid changed')
        return {value:i for i,g in enumerate(groups) for value in g},[min(g) for g in groups]
    cols,oldx=bins(x,4);rows,oldy=bins(y,2)
    rng=np.random.default_rng(49157+nuisance)
    inner=int(rng.integers(20,33));outer=int(rng.integers(52,69));vertical=int(rng.integers(40,65))
    shift=rng.integers(-3,4,size=2)
    newx=[19+int(shift[1]),19+inner+int(shift[1]),19+outer+int(shift[1]),19+outer+inner+int(shift[1])]
    newy=[33+int(shift[0]),33+vertical+int(shift[0])]
    for j in range(len(events)):
        c=cols[int(x[j])];r=rows[int(y[j])]
        events[j,2]=newx[c]+int(x[j])-oldx[c]
        events[j,1]=newy[r]+int(y[j])-oldy[r]
    if not ((events[:,1:3]>=0).all() and (events[:,1:3]<128).all()):raise ValueError('native root support exceeded')
    return replace(row,events=events)


def training_corpus():
    return [metric_scene(a,b,r,n) for a,b,r,n in itertools.product(range(8),range(8),range(2),TRAIN_TAPES) if split_for(a,b)=='train']


def fresh_dev_corpus():
    return [metric_scene(a,b,r,n) for a,b,r,n in itertools.product(range(8),range(8),range(2),FRESH_DEV_TAPES) if split_for(a,b)=='dev']


def broad_calibration(rows):
    # One observed root-swapped pair per fixed TRAIN arrangement pair. Spread
    # tapes independently of labels;choose using index only,not model scores.
    pairs=training_pairs([r.model_input() for r in rows]);by_tape={n:[] for n in TRAIN_TAPES}
    for i,j in pairs:by_tape[rows[i].nuisance].append((i,j))
    result=[]
    for k in range(32):
        i,j=by_tape[TRAIN_TAPES[k]][k];result.extend((rows[i],rows[j]))
    assert len(result)==64
    return result


def audit_corpus(rows):
    expected=64*len(TRAIN_TAPES)
    assert len(rows)==expected and all(r.split=='train' for r in rows)
    layouts={(r.left,r.right,r.root_swap) for r in rows};assert len(layouts)==64
    assert {r.nuisance for r in rows}==set(TRAIN_TAPES)
    assert sum(r.label for r in rows)*2==len(rows)
    pairs=training_pairs([r.model_input() for r in rows]);assert len(pairs)==expected//2
    digests={r.digest for r in rows};assert len(digests)==len(rows)
    for i,j in pairs:assert rows[i].nuisance==rows[j].nuisance and rows[i].label!=rows[j].label
    h=hashlib.sha256()
    for r in rows:h.update(r.events.tobytes());h.update(bytes([r.label]))
    return dict(examples=len(rows),abstract_arrangements=len(layouts),tapes=len(TRAIN_TAPES),pairs=len(pairs),data_sha256=h.hexdigest(),scope='Original64TRAIN topologies only;32new timing/order/translation/metric-layout tapes. No DEV/TEST combinations admitted.')
