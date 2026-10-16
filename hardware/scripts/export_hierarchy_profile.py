#!/usr/bin/env python3
"""Read the acquired hierarchy checkpoint; no fit or model evaluation."""
import hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
def main():
 name='experiments/results/hierarchy/aws_g_hier_carried_continued_20261015_v1.raw.pt'
 path=ROOT/name;weights=torch.load(path,map_location='cpu',weights_only=True)
 levels=sorted({int(k.split('.')[1]) for k in weights if k.startswith('objects.')})
 profiles=[]
 for level in levels:
  prefix=f'objects.{level}.core.'+('base.' if level else '')
  transport=weights[prefix+'transport_rate'];freq=weights[prefix+'transport_frequency']
  assert transport.shape==freq.shape
  depth,heads,dynamic_pairs=transport.shape
  key=next(k for k in weights if k.startswith(prefix+'units.0.0.') and k.endswith('.key'))
  payload=weights[key].numel();protected_pairs=payload//2-dynamic_pairs
  profiles.append(dict(level=level,stages=depth,heads=heads,payload=payload,protected_pairs=protected_pairs,learned_transport_modes=dynamic_pairs,transport_rate_tensor=prefix+'transport_rate',transport_frequency_tensor=prefix+'transport_frequency'))
 p=dict(model='Acquired original relational native hierarchy',selection_scope='Saved whole-label DEV reference0.686609;one development seed,no semantic benchmark claim.',checkpoint=dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),parameters=sum(v.numel() for v in weights.values()),levels=profiles,tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in weights.items()},rtl_boundary='These RTL primitives do not implement this complete actor. Original floating-point matrices,layernorm,GELU/sigmoid/softplus,learned proposals,key scoring,clock transforms/randomness,full scheduling,retained value versions,class readout and counterfactual learning require implementation/export and numerical validation. Semantic tick scale and quantized precision remain design choices,not established checkpoint parity.')
 (ROOT/'hardware/profiles/native_hierarchy.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(dict(parameters=p['parameters'],levels=profiles),indent=2))
if __name__=='__main__':main()
