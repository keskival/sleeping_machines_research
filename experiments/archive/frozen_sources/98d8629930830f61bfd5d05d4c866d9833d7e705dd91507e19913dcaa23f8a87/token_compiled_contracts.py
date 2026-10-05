"""Bounded compiled/eager token-core parity and warm CPU throughput contract."""
import hashlib,json,platform,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.fast_native_core import fast_class
from sleeping_machines.token_episodes_compiled import token_features

torch.set_num_threads(1);torch.manual_seed(6)
m=fast_class(AddressedEventHeads)(sources=1,content_dim=50257,classes=1,payload=16,depth=2,heads=2,pool=4)
ids=torch.randint(0,50256,(8,4));times={};results=[]
for compiled in (False,True):
    m.zero_grad(set_to_none=True);before=time.monotonic()
    x,st,_=token_features(m,ids,generator=torch.Generator().manual_seed(7),compiled=compiled)
    x.square().mean().backward();times[str(compiled)]=time.monotonic()-before
    results.append((x.detach().clone(),{n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None},st['last_writes']))
a,b=results
assert torch.allclose(a[0],b[0],atol=2e-4,rtol=2e-4)
assert torch.equal(a[2],b[2])
for n,g in a[1].items():
    assert torch.allclose(g,b[1][n],atol=2e-4*max(1.,float(g.abs().max())),rtol=2e-4),n
out=dict(status='completed',contract='float32 eager/compiled token features, every gradient and hard writes',
    cold_forward_backward_s=times,host=platform.node(),max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    sources={str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
    ('experiments/token_compiled_contracts.py','sleeping_machines/token_episodes_compiled.py','sleeping_machines/recruit_layer.py')})
p=ROOT/'experiments/results/diagnostics/curie_token_compiled_contracts_20261005_v1.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
