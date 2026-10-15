"""Untrained synthetic numerical signal diagnostic, never a benchmark score."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from torch.nn import functional as F
from sleeping_machines.parent_race_hierarchy import ParentRaceHierarchy


def run():
    torch.set_num_threads(1)
    records = []
    for transport, dtype in ((False,torch.float32),(False,torch.float64),(True,torch.float32),(True,torch.float64)):
        torch.manual_seed(17)
        model = ParentRaceHierarchy(input_width=3,width=8,levels=8,refinements=4,roles=3,classes=3,transport=transport).to(dtype)
        features = torch.rand(1,3,16,16,dtype=torch.float32).to(dtype)
        with torch.no_grad():
            factual = model(features,seed=19)
            rows = []
            for index,audit in enumerate(factual['audits']):
                losses = []
                for candidate in audit['valid'][0,0].nonzero().flatten().tolist():
                    alternate = model(features,seed=19,force=(index,0,0,candidate))
                    losses.append(float(F.cross_entropy(alternate['logits'],torch.tensor([1]))))
                rows.append(dict(cell=index,abstraction_level=index//4,
                                 final_loss_contrast=max(losses)-min(losses)))
        records.append(dict(transport=transport,dtype=str(dtype),cells=rows,
                     nonzero_contrasts=sum(row['final_loss_contrast']>0 for row in rows)))
    sources = ['sleeping_machines/parent_race_hierarchy.py',__file__]
    return dict(status='completed',kind='untrained synthetic numerical contract',
                scope='one initialization; fixed 16x16 feature surface; no fit or semantic claim',
                records=records,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); args=parser.parse_args()
    result=run(); path=Path(args.out)
    if path.exists():
        raise SystemExit('refusing to overwrite evidence')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({str(r['transport'])+' '+r['dtype']:dict(nonzero=r['nonzero_contrasts'],total=len(r['cells']),
          first_contrast=r['cells'][0]['final_loss_contrast']) for r in result['records']}))
