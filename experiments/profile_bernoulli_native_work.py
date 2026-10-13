"""R1 operation/payload/semantic accounting of the unchanged integrated fit.

Tensor payload exposures are a logical execution ledger, not cache/DRAM reads.
Registry FLOPs are coverage-limited; unsupported operations remain explicit.
"""
import json
import math
import sys
from collections import Counter,defaultdict
from contextlib import contextmanager
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import bernoulli_teacher_token_language as engine
from credit.token_flop_accounting import TokenFlopCounter
from credit.native_scalar_work import scalar_work,validate_scalar_rules
from sleeping_machines.token_readout import TokenReadout
from sleeping_machines import sparse_counterfactual_episodes as episodes


def tensors(value):
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,(list,tuple)):
        for x in value:yield from tensors(x)
    elif isinstance(value,dict):
        for x in value.values():yield from tensors(x)


class Ledger(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.scalar=defaultdict(Counter);self.coverage=defaultdict(Counter);self.registry={};self.phase='other_unclassified';self.operations=defaultdict(Counter)
        self.payload=defaultdict(Counter);self.saved=defaultdict(Counter);self.semantic=defaultdict(Counter)
        self.storage_seen=set();self.saved_window_bytes=0;self.max_saved_window_bytes=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        out=func(*args,**(kwargs or {}));name=str(func);phase=self.phase
        self.operations[phase][name]+=1
        coverage,work=scalar_work(func,args,kwargs or {},out,self.registry)
        self.scalar[phase].update(work);self.coverage[phase][name+' | '+coverage]+=1
        metadata=name.split('.')[1] in {'view','_unsafe_view','as_strided','transpose','t','permute','detach','alias','slice','select','unsqueeze','squeeze','expand'}
        if not metadata:
            self.payload[phase]['input_tensor_payload_bytes']+=sum(t.numel()*t.element_size() for t in tensors((args,kwargs or {})))
            self.payload[phase]['output_tensor_payload_bytes']+=sum(t.numel()*t.element_size() for t in tensors(out))
        return out
    def pack(self,tensor):
        self.saved[self.phase]['saved_tensor_references']+=1
        self.saved[self.phase]['referenced_tensor_payload_bytes']+=tensor.numel()*tensor.element_size()
        storage=tensor.untyped_storage();key=(storage.data_ptr(),storage.nbytes())
        if key not in self.storage_seen:
            self.storage_seen.add(key);self.saved_window_bytes+=storage.nbytes()
            self.max_saved_window_bytes=max(self.max_saved_window_bytes,self.saved_window_bytes)
        return tensor
    @staticmethod
    def unpack(tensor):return tensor


def main():
    scalar_contract=validate_scalar_rules()
    ledger=Ledger();counts=defaultdict(int);flops=defaultdict(int);restores=[]
    counter=TokenFlopCounter(display=False);ledger.registry=counter.flop_registry
    @contextmanager
    def phase(name):
        old=ledger.phase;ledger.phase=name;before=counter.get_total_flops()
        try:yield
        finally:
            flops[name]+=counter.get_total_flops()-before;counts[name]+=1;ledger.phase=old
    def wrap(cls,name,label):
        original=getattr(cls,name);restores.append((cls,name,original))
        def invoke(self,*args,**kwargs):
            with phase(label(self)):return original(self,*args,**kwargs)
        setattr(cls,name,invoke)
    wrap(engine.Model,'forward',lambda m:'evaluation_core' if not m.training else ('alternative_core' if m.shadow_mode else 'factual_core'))
    wrap(TokenReadout,'nll',lambda m:'evaluation_readout' if not m.training else 'training_readout')
    wrap(torch.Tensor,'backward',lambda _:'backward')
    original_step=torch.optim.AdamW.step
    restores.append((torch.optim.AdamW,'step',original_step))
    optimizer_storage={}
    def step(optimizer,*args,**kwargs):
        with phase('optimizer'):result=original_step(optimizer,*args,**kwargs)
        optimizer_storage['parameter_bytes']=sum(p.numel()*p.element_size() for g in optimizer.param_groups for p in g['params'])
        optimizer_storage['gradient_bytes']=sum(p.grad.numel()*p.grad.element_size() for g in optimizer.param_groups for p in g['params'] if p.grad is not None)
        optimizer_storage['optimizer_tensor_bytes']=sum(t.numel()*t.element_size() for state in optimizer.state.values() for t in state.values() if isinstance(t,torch.Tensor))
        ledger.storage_seen.clear();ledger.saved_window_bytes=0
        return result
    torch.optim.AdamW.step=step
    clip=torch.nn.utils.clip_grad_norm_;restores.append((torch.nn.utils,'clip_grad_norm_',clip))
    def clipped(*args,**kwargs):
        with phase('global_clip'):return clip(*args,**kwargs)
    torch.nn.utils.clip_grad_norm_=clipped
    original_layer=episodes.sparse_counterfactual_layer;restores.append((episodes,'sparse_counterfactual_layer',original_layer))
    def layer(*args,**kwargs):
        n,H,U,P=args[2].shape;s=ledger.semantic[ledger.phase]
        s['scored_keys']+=n*H*U;s['selected_receiver_updates']+=n*H
        s['selected_value_float32_payload_bytes']+=n*H*P*4
        s['head_arrival_float64_payload_bytes']+=n*H*8
        s['head_max_reductions']+=n;s['race_min_reductions']+=n*H
        s['sampled_local_alternative_proposals']+=n*H*int(bool(kwargs.get('sampled_credit',False)))
        return original_layer(*args,**kwargs)
    episodes.sparse_counterfactual_layer=layer
    try:
        with ledger,counter,torch.autograd.graph.saved_tensors_hooks(ledger.pack,ledger.unpack):engine.main()
    finally:
        for obj,name,original in reversed(restores):setattr(obj,name,original)
    tag=sys.argv[sys.argv.index('--tag')+1];path=engine.ROOT/f'experiments/results/token_language/{tag}.json'
    r=json.loads(path.read_text());targets=r['presentations_total']
    fit=('factual_core','alternative_core','training_readout','backward','optimizer','global_clip')
    registry=counter.flop_registry
    registered_packets=sorted(str(packet) for packet in registry)
    # Operation names are retained in full so omissions cannot be hidden by
    # treating the registry subtotal as complete fitting FLOPs.
    accounting=dict(fit_targets=targets,scalar_rule_contract=scalar_contract,scalar_algorithm_work_by_phase={k:dict(v) for k,v in ledger.scalar.items()},
        operation_coverage_by_phase={k:dict(v) for k,v in ledger.coverage.items()},registered_operator_packets=registered_packets,phase_calls=dict(counts),registered_flops_by_phase=dict(flops),
        registered_fit_flops=sum(flops[p] for p in fit),registered_fit_flops_per_target=sum(flops[p] for p in fit)/targets,
        operations_by_phase={k:dict(v) for k,v in ledger.operations.items()},
        logical_payload_by_phase={k:dict(v) for k,v in ledger.payload.items()},
        saved_graph_by_phase={k:dict(v) for k,v in ledger.saved.items()},
        maximum_unique_saved_storage_bytes_per_optimizer_window=ledger.max_saved_window_bytes,
        semantic_events_by_phase={k:dict(v) for k,v in ledger.semantic.items()},optimizer_storage=optimizer_storage,
        synchronization=dict(global_gradient_clip_calls=counts['global_clip'],global_optimizer_step_calls=counts['optimizer'],
            cpu_backward_calls=counts['backward'],credit_window_length=r['args']['credit_window']),
        boundary='Complete operation ledger for the probe; other_unclassified includes setup and unwrapped learning/protocol operations, so declared fitting-phase FLOPs are a subtotal. Evaluation is separately labelled. Registry matrix FLOPs plus declared scalar algorithm counts cover named arithmetic only; nonlinear functions/comparisons remain separate units and unknown operations are explicit. Declared algorithm rules do not measure fused-kernel instruction counts. Never a complete-FLOP advantage while uncovered arithmetic remains. Tensor input/output payload exposures are logical upper-style operand accounting, not measured memory reads, DRAM traffic or energy. Saved storage includes parameters and reused backing storage; it is referenced graph storage, not allocator peak. Semantic value/arrival bytes are declared float32/64 event payloads before placement/fanout/handshake costs; not measured wire traffic. Dense state masking/materialization remains in the operation ledger. Global clipping, batch Adam and reverse-mode graph remain; this is not clockless learning. Probe profiling overhead invalidates wall-speed comparisons.')
    r['native_work_accounting']=accounting
    for source in ('experiments/profile_bernoulli_native_work.py','experiments/credit/token_flop_accounting.py','experiments/credit/native_scalar_work.py'):
        r['source_sha256'][source]=engine.sha(engine.ROOT/source)
    r['identity']['source_sha256']=dict(r['source_sha256'])
    path.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(registered_fit_flops=accounting['registered_fit_flops'],semantic=accounting['semantic_events_by_phase'],storage=optimizer_storage)),flush=True)


if __name__=='__main__':main()
