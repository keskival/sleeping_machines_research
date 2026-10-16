"""R1 integration gate: native lower-memory queries on real tokenized TRAIN."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import numpy as np
import pytest
import torch
from r1_cerl import load_fineweb
from r1_local_native_deep import LocalNativeDeepLM,word_classes
from sleeping_machines.queryable_local_deep_race import QueryableLocalDeepRace


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(161016);began=time.monotonic()
    tests=['tests/test_queryable_local_deep_race.py']
    assert pytest.main(['-q',*tests])==0
    data=load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin')
    counts=np.bincount(np.asarray(data[:4194304]).astype(np.int64),minlength=50257)
    classes,words=word_classes(counts,256)
    model=LocalNativeDeepLM(50257,classes,words,width=64,depth=8,heads=4,pool=16,concept_units=4096,topk=4,explore=2,dk=32,packed=False)
    model.core=QueryableLocalDeepRace(width=64,depth=8,heads=4,pool=16,publication_capacity=16)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);state=model.new_state()
    g=torch.Generator().manual_seed(13);alt=torch.Generator().manual_seed(14)
    tokens=torch.from_numpy(np.asarray(data[:33]).astype(np.int64));reports=[];start=time.monotonic()
    for b,e in [(0,16),(16,32)]:
        opt.zero_grad(set_to_none=True)
        nll,obj,state,packets,report=model.stream_nll(tokens[b:e],tokens[b+1:e+1],state,g,alt,stats=True)
        obj.backward();model.deliver_credit(packets,state)
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        report['assembler_gradient_norms']=[float(m.pair_weight.grad.norm()) for m in model.core.assemblers]
        assert min(report['assembler_gradient_norms'])>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();reports.append(report)
    assert state['events']==state['cause_bank'].count==32
    names=['events','candidate_scores','selected_updates','alternative_values','credit_projections',
        'memory_key_scores','memory_selected_reads','memory_alternative_reads','memory_transport_heads',
        'memory_pair_products','queried_historical_versions','memory_gc_removed']
    source=['sleeping_machines/queryable_local_deep_race.py','sleeping_machines/local_deep_race.py',
        'sleeping_machines/bound_native_composition.py','sleeping_machines/relational_frame_composition.py',
        'sleeping_machines/native_receiver_eligibility.py','experiments/r1_local_native_deep.py',
        'experiments/r1_cerl.py',*tests,'experiments/hierarchy/queryable_native_language_gate.py']
    result=dict(status='completed',battle='R1',stage='queryable_native_language_gate',gate='PASS',
        targets=32,updates=2,depth=8,parameters=sum(p.numel() for p in model.parameters()),reports=reports,
        learning_counts={k:state[k] for k in names},storage=model.core.storage(state),
        optimizer_gate_s=time.monotonic()-start,process_wall_s=time.monotonic()-began,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in source},
        scope='Numerical integration only,32GPT2FineWebTRAINtargets,twooptimizersteps.No DEV/TEST,ablation or quality claim.Native8layerreceiver races and lowerpublication query races,versionedbindings,two-rolejointcomposer,own-write eligibility on historical query reads. Local incoming/route credit and parameter-changing history remain approximate;optimizer synchronous. All publication keys scored,selected and sampled losing values decoded. Bounded root rings do not imply constant transitive lineage.')
    out=ROOT/f'experiments/results/diagnostics/{a.tag}.json';assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['gate','optimizer_gate_s','max_rss_kb']}),flush=True)
if __name__=='__main__':main()
