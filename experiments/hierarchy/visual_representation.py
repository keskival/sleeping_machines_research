"""G-HIER: deep trainable spatial representations on fixed-prefix DVS data.

All numerical stages are admitted through run_safe/coordinator; no official-test
files are opened. This is a developed architecture project, not a benchmark verdict.
"""
import argparse
import csv
import hashlib
import json
import math
import resource
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sleeping_machines.spatial_race_hierarchy import SpatialRaceHierarchy

FILES = ['sleeping_machines/spatial_race_hierarchy.py', 'experiments/hierarchy/visual_representation.py']
OUT = ROOT/'experiments/results/hierarchy'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sources():
    return {n: digest(ROOT/n) for n in FILES}


def write_result(tag, d):
    OUT.mkdir(parents=True, exist_ok=True)
    d.update(status='completed', battle='G-HIER', source_sha256=sources(),
             max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with (OUT/(tag+'.json')).open('x') as f:
        json.dump(d, f, indent=2); f.write('\n')


def recording(path, bins=8):
    """Retain 128x128 arrangement; first one second; exact integer bin counts."""
    with path.with_name(path.stem+'_labels.csv').open() as f:
        labels = [(int(r['class'])-1, int(r['startTime_usec']), int(r['endTime_usec'])) for r in csv.DictReader(f)]
    counts = np.zeros((len(labels), bins, 2, 128, 128), dtype=np.uint32)
    with path.open('rb') as f:
        while True:
            line = f.readline(4096)
            if line.strip() == b'#!END-HEADER': break
            if not line or not line.startswith(b'#'): raise ValueError('Bad AEDAT header')
        while header := f.read(28):
            if len(header) != 28: raise ValueError('Truncated AEDAT header')
            kind, _, size, _, overflow, capacity, number, _ = struct.unpack('<hhiiiiii', header)
            if min(size, capacity, number) < 0 or number > capacity or size*capacity > 32*1024*1024:
                raise ValueError('Invalid packet size')
            raw = f.read(size*capacity)
            if len(raw) != size*capacity: raise ValueError('Truncated AEDAT packet')
            if kind != 1 or size != 8 or not number: continue
            ev = np.frombuffer(raw, dtype='<u4', count=2*number).reshape(-1, 2)
            addr = ev[:, 0]; t = ev[:, 1].astype(np.int64) | (np.int64(overflow) << 31)
            x, y, pol = (addr >> 17) & 0x1FFF, (addr >> 2) & 0x1FFF, (addr >> 1) & 1
            for i, (_, start, end) in enumerate(labels):
                keep = ((addr & 1) == 1) & (t >= start) & (t < min(end, start+1_000_000)) & (x < 128) & (y < 128)
                dt = ((t[keep]-start)*bins//1_000_000).astype(np.int64)
                np.add.at(counts[i], (dt, pol[keep], y[keep], x[keep]), 1)
    return counts, labels


def prepare(a):
    data = ROOT/'data/dvsgesture/DvsGesture'
    official_train = data/'trials_to_train.txt'
    names = [x.strip() for x in official_train.read_text().splitlines() if x.endswith('.aedat')]
    cache = OUT/(a.tag+'_data'); cache.mkdir(parents=True, exist_ok=False)
    rows, bindings = [], {str(official_train.relative_to(ROOT)): digest(official_train)}
    for name in sorted(names):
        user = int(name[4:6])
        if not 1 <= user <= 23: continue
        path = data/name; counts, labels = recording(path)
        split = 'fit' if user <= 16 else 'select' if user <= 19 else 'dev'
        for i, (label, _, _) in enumerate(labels):
            p = cache/f'{path.stem}_{i}.npz'; np.savez_compressed(p, counts=counts[i])
            rows.append(dict(path=str(p.relative_to(ROOT)), sha256=digest(p), label=label, user=user,
                             identity=f'{name}:{i}', split=split))
        for p in [path, path.with_name(path.stem+'_labels.csv')]: bindings[str(p.relative_to(ROOT))] = digest(p)
        print(json.dumps(dict(recording=name, examples=len(labels), split=split)), flush=True)
    for split in ['fit', 'select', 'dev']:
        assert any(r['split'] == split for r in rows)
    write_result(a.tag, dict(stage='prepare', rows=rows, input_sha256=bindings,
                 counts={s:sum(r['split']==s for r in rows) for s in ['fit','select','dev']},
                 protocol='Official TRAIN files only; users1..16 fitting,17..19 selection,20..23 held-out DEV; first1s,8 fixed bins,128x128 polarity counts. Official test untouched.'))


def load_batch(rows):
    tensors = []
    for row in rows:
        p = ROOT/row['path']
        if digest(p) != row['sha256']: raise ValueError('Changed count artifact')
        with np.load(p, allow_pickle=False) as z: tensors.append(torch.from_numpy(z['counts'].astype(np.float32)))
    return torch.stack(tensors), torch.tensor([r['label'] for r in rows])


def evaluate(model, rows, ablate_stage=None):
    model.eval(); total = hits = n = 0; stage_nll=np.zeros(len(model.stages));stage_hits=np.zeros(len(model.stages))
    with torch.no_grad():
        for i in range(0, len(rows), 2):
            x, y = load_batch(rows[i:i+2]); o = model(x, ablate_stage=ablate_stage)
            total += F.cross_entropy(o['logits'], y, reduction='sum').item()
            hits += int((o['logits'].argmax(-1) == y).sum()); n += len(y)
            for level,lg in enumerate(o['stage_logits']):
                stage_nll[level]+=F.cross_entropy(lg,y,reduction='sum').item()
                stage_hits[level]+=int((lg.argmax(-1)==y).sum())
    return dict(nll=total/n, accuracy=hits/n, targets=n, stages=[dict(stage=i,nll=float(v/n),accuracy=float(stage_hits[i]/n)) for i,v in enumerate(stage_nll)])


def train(a):
    torch.set_num_threads(1); torch.manual_seed(a.seed)
    parent = ROOT/a.data; receipt = json.loads(parent.read_text()); assert receipt['status']=='completed'
    if a.mode == 'fit':
        smoke_path=ROOT/a.smoke_result;smoke=json.loads(smoke_path.read_text())
        assert smoke['status']=='completed' and smoke['stage']=='smoke'
        assert smoke['source_sha256']==sources() and smoke['input_sha256'][a.data]==digest(parent)
        if smoke['max_rss_kb']*1.25>3_000_000:raise ValueError('Smoke exceeds admitted RSS reservation')
        fit_count=receipt['counts']['fit']
        estimated=smoke['history'][0]['epoch_wall_s']/smoke['updates']*math.ceil(fit_count/2)*a.epochs*1.5
        if estimated>a.max_wall_s:raise ValueError(f'Measured deep-fit estimate {estimated:.0f}s exceeds {a.max_wall_s}s; optimize rather than silently reduce model depth')
    rows = receipt['rows']; fit = [r for r in rows if r['split']=='fit']; select = [r for r in rows if r['split']=='select']
    dev = [r for r in rows if r['split']=='dev']
    assert not ({r['user'] for r in fit} & {r['user'] for r in select})
    model = SpatialRaceHierarchy(blocks_per_stage=a.blocks_per_stage)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    rng = np.random.default_rng(a.seed); gen = torch.Generator().manual_seed(a.seed+700)
    checkpoint = OUT/(a.tag+'.progress.pt'); bestpath = OUT/(a.tag+'.best.pt')
    start = time.monotonic(); history=[]; best=math.inf; epoch0=0; updates=0; prior_wall=0.
    if checkpoint.exists():
        saved=torch.load(checkpoint,weights_only=False); assert saved['sources']==sources() and saved['data_sha256']==digest(parent) and saved['configuration']==vars(a)
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); rng.bit_generator.state=saved['numpy_rng']
        gen.set_state(saved['race_rng']); torch.set_rng_state(saved['torch_rng'])
        history=saved['history']; best=saved['best']; epoch0=saved['epoch_next']; updates=saved['updates']; prior_wall=saved['wall_s']
    def save(epoch_next):
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),numpy_rng=rng.bit_generator.state,
                   race_rng=gen.get_state(),torch_rng=torch.get_rng_state(),history=history,best=best,
                   epoch_next=epoch_next,updates=updates,wall_s=prior_wall+time.monotonic()-start,
                   sources=sources(),data_sha256=digest(parent),configuration=vars(a))
        temp=checkpoint.with_suffix('.tmp');torch.save(state,temp);temp.replace(checkpoint)
    gradrows=[]
    for epoch in range(epoch0, 1 if a.mode=='smoke' else a.epochs):
        model.train(); order=rng.permutation(len(fit)); tot=n=0; e0=time.monotonic()
        if a.mode=='smoke': order=order[:8]
        for i in range(0,len(order),2):
            batch=[fit[int(j)] for j in order[i:i+2]];x,y=load_batch(batch)
            o=model(x,generator=gen);loss,parts=model.objective(o,y,audit_site=updates)
            if not torch.isfinite(loss): raise ValueError('Nonfinite integrated loss')
            opt.zero_grad();loss.backward()
            if i==0:
                gradrows=[dict(stage=l,block=b,composition_norm=float(block.compose.weight.grad.norm()),
                      key_norm=float(block.keys.weight.grad.norm())) for l,s in enumerate(model.stages) for b,block in enumerate(s)]
                if not all(r['composition_norm']>0 and r['key_norm']>0 for r in gradrows):
                    raise ValueError('A representation stage lacks active composition/key credit')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();updates+=1
            tot+=float(parts['main'])*len(y);n+=len(y)
        score=evaluate(model,select) if a.mode!='smoke' else None
        row=dict(epoch=epoch,train_nll=tot/n,select=score,epoch_wall_s=time.monotonic()-e0)
        history.append(row);print(json.dumps(row),flush=True)
        if score is not None and score['nll']<best:
            best=score['nll'];torch.save(dict(model=model.state_dict(),epoch=epoch),bestpath)
        save(epoch+1)
        if time.monotonic()-start>a.max_wall_s: raise TimeoutError('Bounded stage timed out; exact epoch checkpoint retained')
        if len(history)>5 and best<math.inf and all(r['select']['nll']>best for r in history[-5:]):break
    if a.mode=='smoke':
        return write_result(a.tag,dict(stage='smoke',updates=updates,parameters=sum(p.numel() for p in model.parameters()),
            work=model.work_descriptor(),history=history,gradients=gradrows,process_wall_s=time.monotonic()-start,
            input_sha256={a.data:digest(parent)},scope='Integrated 32-block real-TRAIN resource/gradient smoke; no DEV or TEST scoring; no hierarchy-quality claim.'))
    saved=torch.load(bestpath,weights_only=False);model.load_state_dict(saved['model'])
    scores=evaluate(model,dev)
    # Frozen representation interventions on a declared first32 DEV records, not model selection.
    ablations={str(l):evaluate(model,dev[:32],ablate_stage=l) for l in range(7)}
    write_result(a.tag,dict(stage='fit',parameters=sum(p.numel() for p in model.parameters()),history=history,
        dev=scores,ablation_reference=evaluate(model,dev[:32]),spatial_stage_ablations=ablations,
        gradients=gradrows,work=model.work_descriptor(),final_weights=str(bestpath.relative_to(ROOT)),
        process_wall_s=prior_wall+time.monotonic()-start,input_sha256={a.data:digest(parent)},
        scope='Single-seed development of deep native spatial feature races; selection on users17..19, DEV20..23. Fixed1s observation. No cat/part labels, official test, hierarchy semantics or hardware advantage inferred.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--mode',choices=['prepare','smoke','fit'],required=True)
    p.add_argument('--data');p.add_argument('--smoke-result');p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--blocks-per-stage',type=int,default=4);p.add_argument('--max-wall-s',type=float,default=14400)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/(a.tag+'.json')).exists():raise ValueError('Preserve completed evidence')
    prepare(a) if a.mode=='prepare' else train(a)


if __name__=='__main__': main()
