"""R1 instrument the reserved deep fit without changing arithmetic or RNG."""
import json
import sys
import time
from collections import defaultdict
import torch
import allocated_depth_token_language as engine
from sleeping_machines.token_readout import TokenReadout


def main():
    timings=defaultdict(float);calls=defaultdict(int);originals=[]
    def wrap(cls,name,label):
        original=getattr(cls,name);originals.append((cls,name,original))
        def timed(self,*args,**kwargs):
            key=label(self);start=time.perf_counter()
            try:return original(self,*args,**kwargs)
            finally:
                timings[key]+=time.perf_counter()-start;calls[key]+=1
        setattr(cls,name,timed)
    wrap(engine.Model,'forward',lambda m:'evaluation_core' if not m.training else
         ('alternative_core' if m.shadow_mode else 'factual_core'))
    wrap(TokenReadout,'nll',lambda m:'training_readout' if m.training else 'evaluation_readout')
    wrap(torch.Tensor,'backward',lambda _:'backward')
    wrap(torch.optim.AdamW,'step',lambda _:'optimizer')
    start=time.perf_counter()
    try:engine.main()
    finally:
        for cls,name,original in reversed(originals):setattr(cls,name,original)
    tag=sys.argv[sys.argv.index('--tag')+1]
    path=engine.ROOT/f'experiments/results/token_language/{tag}.json'
    packet=json.loads(path.read_text())
    packet['profile']=dict(stage_wall_s=dict(timings),stage_calls=dict(calls),
        instrumented_total_s=sum(timings.values()),process_wall_s=time.perf_counter()-start,
        scope='Nonoverlapping factual/alternative/evaluation core, readout, backward and AdamW calls. Timing includes instrumentation overhead; remaining process work includes data, audit allocation, diagnostics, checkpointing and setup. No FLOP or energy claim. Profile covers this process only after a resume.')
    packet['instrumentation_source_sha256']={str(__file__):engine.sha(__file__)}
    tmp=path.with_suffix('.profile.tmp');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(path)


if __name__=='__main__':main()
