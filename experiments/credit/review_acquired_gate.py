"""Acquired eight-layer native repair integration and exact recovery gate."""
import argparse,copy,hashlib,io,json,resource,subprocess,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'experiments/credit')]
import numpy as np
import torch
import r1_native_reciprocal_future as base
from sleeping_machines.integrated_review_learning import enable_reviewed_learning
from sleeping_machines.native_optimizer_policy import regroup_native_optimizer
from sleeping_machines.document_boundary_causes import document_cause_law
from tests.test_native_query_migration import identical

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def bundle(model,opt,state,g,a,cursor):
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,g=g.get_state(),alt=a.get_state(),rng=torch.get_rng_state(),cursor=cursor)
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args();torch.set_num_threads(1);start=time.monotonic()
    evidence_path=ROOT/'experiments/results/diagnostics/aws_r1_calibrated_reciprocal_continue_20261017_v1.json'
    evidence=json.loads(evidence_path.read_text());path=ROOT/evidence['ordinary_checkpoint'];assert sha(path)==evidence['ordinary_checkpoint_sha256']
    sources=base.runtime_sources();sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    # Exact native-head law check against the source used by the acquired trial.
    historical=subprocess.check_output(['git','show','75ca8099d:experiments/r1_cerl.py']);assert hashlib.sha256(historical).hexdigest()==evidence['source_sha256']['experiments/r1_cerl.py']
    legacy=types.ModuleType('legacy_r1_cerl_review');legacy.__file__=str(ROOT/'experiments/r1_cerl.py');exec(compile(historical,legacy.__file__,'exec'),legacy.__dict__)
    class LegacyHead:
        def __init__(self,head):self.head=head
        def __getattr__(self,name):return getattr(self.head,name)
        def wake(self,*args,**kwargs):return legacy.CERL.wake(self,*args,**kwargs)
        def unit_score(self,*args,**kwargs):return legacy.CERL.unit_score(self,*args,**kwargs)
    parent=torch.load(path,map_location='cpu',weights_only=False);cursor=parent['cursor'];data=base.load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin')
    model,opt,state,g,a=base.actor(parent)
    probe=torch.randn(8,64);token=int(data[cursor]);target=int(data[cursor+1]);stamp=state['events']
    bank=copy.deepcopy(state['cause_bank']);bank.append(token,probe,[],stamp);candidate=bank.query(token,[],stamp,maximum=model.bank_candidates)
    with torch.no_grad():
        expected_head=document_cause_law(LegacyHead(model.causes),probe,target,candidate,explore=False)
        actual_head=document_cause_law(model.causes,probe,target,candidate,explore=False)
    assert torch.equal(expected_head[0],actual_head[0])
    words=torch.from_numpy(np.asarray(data[cursor:cursor+5]).astype(np.int64))
    model.eval()
    with torch.no_grad():expected=model.stream_nll(words[:-1],words[1:],state,g,a,learn=False)[0]
    expected_g=g.get_state();del model,opt,state,parent
    parent=torch.load(path,map_location='cpu',weights_only=False);model,opt,state,g,a=base.actor(parent)
    before=copy.deepcopy(opt.state_dict());receipt=enable_reviewed_learning(model,state)
    assert identical(before,opt.state_dict()) and all(torch.equal(v,model.state_dict()[n]) for n,v in parent['model'].items())
    model.eval()
    with torch.no_grad():actual=model.stream_nll(words[:-1],words[1:],state,g,a,learn=False)[0]
    assert torch.equal(expected,actual) and torch.equal(expected_g,g.get_state())
    del model,opt,state,parent
    parent=torch.load(path,map_location='cpu',weights_only=False);model,opt,state,g,a=base.actor(parent);enable_reviewed_learning(model,state)
    moments={param:copy.deepcopy(value) for param,value in opt.state.items()};groups=regroup_native_optimizer(model,opt,.01)
    assert all(identical(value,opt.state[param]) for param,value in moments.items());del moments,parent
    first=base.update(model,opt,state,g,a,data,cursor,targets=4)[0];cursor+=4
    saved=io.BytesIO();torch.save(bundle(model,opt,state,g,a,cursor),saved)
    expected=base.update(model,opt,state,g,a,data,cursor,targets=4)[0]
    endpoint=copy.deepcopy(bundle(model,opt,state,g,a,cursor+4))
    saved.seek(0);saved_state=torch.load(saved,weights_only=False);model.load_state_dict(saved_state['model']);opt.load_state_dict(saved_state['optimizer']);state=saved_state['state'];g.set_state(saved_state['g']);a.set_state(saved_state['alt']);torch.set_rng_state(saved_state['rng'])
    actual=base.update(model,opt,state,g,a,data,cursor,targets=4)[0]
    assert actual==expected and identical(endpoint,bundle(model,opt,state,g,a,cursor+4));cursor+=4
    assert state['direct_producer_replays']>0 and state['nonlinear_route_audits']>0 and state['episodic_writer_replays']>0
    checkpoint=ROOT/f'experiments/results/credit/{args.tag}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema='reviewed_native_v1',parent_checkpoint=str(path.relative_to(ROOT)),parent_sha256=sha(path),**bundle(model,opt,state,g,a,cursor)),checkpoint)
    result=dict(status='completed',battle='R1',stage='reviewed acquired native learner gate',gate='PASS',source_sha256=sources,parent_result=str(evidence_path.relative_to(ROOT)),parent_checkpoint_sha256=sha(path),parent_cursor=cursor-8,final_cursor=cursor,actual_targets=8,discarded_recovery_targets=4,legacy_head_law_equal=True,acquired_forecast_equal=True,exact_model_adam_state_rng_recovery=True,receipt=receipt,optimizer_policy=groups,training_losses=first+actual,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),direct_replays=state['direct_producer_replays'],nonlinear_audits=state['nonlinear_route_audits'],episodic_writer_replays=state['episodic_writer_replays'],elapsed_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Acquired eight-layer native integration and finite updates, not a prediction or hardware win. Nonlinear route suffix is one actual event. All snapshots/replays paid; earlier source-bound jobs unchanged.')
    (ROOT/f'experiments/results/diagnostics/{args.tag}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
