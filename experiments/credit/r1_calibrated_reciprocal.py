"""Prospective native reciprocal repair: calibrated values, independent audits.

Reuses the best saved complete-native reference and the acquired credit core.
Audits never decide a forecast that has already been issued.
"""
import argparse,gc,hashlib,json,os,resource,shutil,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'experiments/credit')]
import numpy as np
import torch,pytest
import r1_native_reciprocal_future as base
from sleeping_machines.calibrated_native_future_credit import CalibratedNativeFutureCredit
from sleeping_machines.native_future_update_credit import addressed_mix_action
from tests.test_native_query_migration import identical


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path,value):
    writing=path.with_suffix('.writing');torch.save(value,writing);os.replace(writing,path)
def actor_bundle(model,opt,state,g,a,cursor):
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,g=g.get_state(),alt=a.get_state(),rng=torch.get_rng_state(),cursor=cursor)
def snapshot(model,opt,state,g,a,cursor,critic,meta,cs,cg,ca,audit_rng):
    return dict(actor=actor_bundle(model,opt,state,g,a,cursor),critic=critic.state_dict(),meta=meta.state_dict(),
                credit_state=cs,cg=cg.get_state(),ca=ca.get_state(),audit_rng=audit_rng.get_state())
def sources():
    result=base.runtime_sources()
    for name in ['experiments/credit/r1_calibrated_reciprocal.py','tests/test_calibrated_native_future_credit.py']:
        result[name]=sha(ROOT/name)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--mode',choices=['contract','acquired','continue'],required=True);p.add_argument('--parent')
    args=p.parse_args();torch.set_num_threads(1);started=time.monotonic();past=0.;old_peak=0
    loaded=sources();out=ROOT/f'experiments/results/diagnostics/{args.tag}.json';assert not out.exists()
    if args.mode=='contract':
        assert pytest.main(['-q','tests/test_calibrated_native_future_credit.py'])==0
        result=dict(status='completed',battle='R1',stage='calibrated native reciprocal numerical gate',gate='PASS')
    else:
        continuation=args.mode=='continue';evidence=json.loads((ROOT/args.parent).read_text())
        if continuation:
            assert evidence['status']=='completed' and evidence['gate']=='PASS'
            assert evidence['stage']=='calibrated native reciprocal acquired gate'
            path=ROOT/evidence['checkpoint'];assert sha(path)==evidence['checkpoint_sha256']
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['source_sha256']==loaded
            model,opt,state,g,a=base.actor(saved['actor']);actor_rng=torch.get_rng_state()
            critic=CalibratedNativeFutureCredit(64,depth=8,heads=4,pool=4)
            critic.load_state_dict(saved['critic']);cs=saved['credit_state']
            cg=torch.Generator().set_state(saved['cg']);ca=torch.Generator().set_state(saved['ca'])
            audit_rng=torch.Generator().set_state(saved['audit_rng']);cursor=saved['actor']['cursor']
            meta=torch.optim.AdamW(critic.parameters(),lr=.0001,weight_decay=0.)
            meta.load_state_dict(saved['meta']);del saved
        else:
            assert evidence['stage']=='native reciprocal future credit prospective continuation'
            for name,digest in evidence['source_sha256'].items():assert sha(ROOT/name)==digest,name
            path=ROOT/evidence['baseline_checkpoint'];assert sha(path)==evidence['baseline_checkpoint_sha256']
            saved=torch.load(path,map_location='cpu',weights_only=False)
            model,opt,state,g,a=base.actor(saved);cursor=saved['cursor'];del saved
            actor_rng=torch.get_rng_state()
            critic=CalibratedNativeFutureCredit(64,depth=8,heads=4,pool=4)
            old_path=ROOT/evidence['checkpoint'];assert sha(old_path)==evidence['checkpoint_sha256']
            old=torch.load(old_path,map_location='cpu',weights_only=False)
            body={k:v for k,v in old['critic'].items() if k.startswith(('core.','context.'))}
            missing=critic.load_state_dict(body,strict=False)
            assert set(missing.missing_keys)=={'response','covariance','residual_variance','observations'} and not missing.unexpected_keys
            for name,value in body.items():assert torch.equal(critic.state_dict()[name],value)
            cs=old['credit_state'];cg=torch.Generator().set_state(old['cg']);ca=torch.Generator().set_state(old['ca'])
            audit_rng=torch.Generator().manual_seed(20177);del old
            # Only the changed credit objective gets fresh optimizer moments.
            # Actor Adam and acquired native feature/core weights are preserved.
            meta=torch.optim.AdamW(critic.parameters(),lr=.0001,weight_decay=0.)
        torch.set_rng_state(actor_rng);parent_digest=sha(path)
        data=base.load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin')
        episodes,horizon=(64,32) if continuation else (2,8)
        rows=[];start_cursor=cursor;progress=ROOT/f'experiments/results/credit/{args.tag}.resume.pt'
        counters=dict(actual=0,audit=0,control=0,ordinary=0,immediate=0)
        with tempfile.TemporaryDirectory(prefix='calibrated-credit-',dir=ROOT/'.git') as folder:
            folder=Path(folder);anchor=folder/'anchor.pt';issued=folder/'issued.pt'
            ordinary=folder/'ordinary.pt';immediate=folder/'immediate.pt'
            if continuation:
                base.dump(ordinary,model,opt,state,g,a,cursor);base.dump(immediate,model,opt,state,g,a,cursor)
            if continuation and progress.exists():
                resumed=torch.load(progress,map_location='cpu',weights_only=False)
                assert resumed['source_sha256']==loaded and resumed['parent_sha256']==parent_digest
                del model,opt,state,g,a;gc.collect()
                model,opt,state,g,a=base.actor(resumed['actor']);cursor=resumed['actor']['cursor']
                critic.load_state_dict(resumed['critic']);meta.load_state_dict(resumed['meta']);cs=resumed['credit_state']
                cg.set_state(resumed['cg']);ca.set_state(resumed['ca']);audit_rng.set_state(resumed['audit_rng'])
                rows=resumed['rows'];start_cursor=resumed['start_cursor'];counters=resumed['counters'];base.WORK.update(resumed['work'])
                past=resumed['recorded_process_s'];old_peak=resumed['max_rss_kb']
                torch.save(resumed['ordinary'],ordinary);torch.save(resumed['immediate'],immediate);del resumed
            for episode in range(len(rows),episodes):
                support,gradient,_=base.update(model,opt,state,g,a,data,cursor,targets=4);cursor+=4
                critic.begin_decision();all_quotes=[];packets=[];directions=[]
                for depth in range(8):
                    key=state['publications'][depth][-1];record=state['bindings'].entries[key].native
                    direction=addressed_mix_action(model,{'core.mix':gradient},depth);directions.append(direction)
                    quote,cs,local=critic.quote(record.key_view,record.residual,direction@record.residual,depth/7,.001,
                                                cs,cg,ca,timestamp=state['bindings'].entries[key].available)
                    all_quotes.append(quote);packets.extend(local)
                quotes=torch.stack(all_quotes).flatten()
                service,uncertainty=critic.choose_service(quotes)
                audit,probability=critic.choose_audit(audit_rng)
                depth,kind=divmod(service,3);sign=kind-1;ad,ak=divmod(audit,3);audit_sign=ak-1
                before_quotes=quotes.detach().clone();base.dump(anchor,model,opt,state,g,a,cursor)
                with torch.no_grad():model.core.mix[depth].add_(sign*directions[depth])
                actual=base.advance(model,opt,state,g,a,data,cursor,horizon);counters['actual']+=4+horizon
                base.dump(issued,model,opt,state,g,a,cursor+horizon)
                del model,opt,state,g,a;gc.collect()
                control=actual
                if sign!=0 or (not continuation and episode==0):
                    saved=torch.load(anchor,map_location='cpu',weights_only=False)
                    model,opt,state,g,a=base.actor(saved)
                    assert identical((model.state_dict(),opt.state_dict(),state,g.get_state(),a.get_state(),torch.get_rng_state()),
                                     (saved['model'],saved['optimizer'],saved['state'],saved['g'],saved['alt'],saved['rng']))
                    del saved
                    control=base.advance(model,opt,state,g,a,data,cursor,horizon);counters['control']+=horizon
                    if sign==0:assert actual==control
                    del model,opt,state,g,a;gc.collect()
                if audit==service:
                    audit_losses=actual
                else:
                    saved=torch.load(anchor,map_location='cpu',weights_only=False)
                    model,opt,state,g,a=base.actor(saved);del saved
                    with torch.no_grad():model.core.mix[ad].add_(audit_sign*directions[ad])
                    audit_losses=base.advance(model,opt,state,g,a,data,cursor,horizon);counters['audit']+=horizon
                    del model,opt,state,g,a;gc.collect()
                target=(sum(audit_losses)-sum(control))/horizon
                assert torch.equal(before_quotes,quotes.detach()),'audit changed issued prediction'
                teaching=critic.learn_audit(quotes,audit,target,packets,cs,meta)
                row=dict(episode=episode,service_depth=depth,service_sign=sign,service_uncertainty=uncertainty,
                         audit_depth=ad,audit_sign=audit_sign,audit_probability=probability,
                         issued_losses=actual,support_losses=support,control_losses=control,audit_losses=audit_losses,
                         service_advantage=(sum(actual)-sum(control))/horizon,**teaching)
                if continuation:
                    saved=torch.load(ordinary,map_location='cpu',weights_only=False)
                    model,opt,state,g,a=base.actor(saved);del saved
                    row['ordinary_losses']=base.advance(model,opt,state,g,a,data,cursor-4,horizon+4)
                    counters['ordinary']+=horizon+4;base.dump(ordinary,model,opt,state,g,a,cursor+horizon)
                    del model,opt,state,g,a;gc.collect()
                    saved=torch.load(immediate,map_location='cpu',weights_only=False)
                    model,opt,state,g,a=base.actor(saved);del saved
                    warmup,greedy_gradient,_=base.update(model,opt,state,g,a,data,cursor-4,targets=4)
                    greedy=int(greedy_gradient.flatten(1).square().sum(1).argmax())
                    change=addressed_mix_action(model,{'core.mix':greedy_gradient},greedy)
                    with torch.no_grad():model.core.mix[greedy].sub_(change)
                    row['immediate_losses']=warmup+base.advance(model,opt,state,g,a,data,cursor,horizon)
                    counters['immediate']+=horizon+4;base.dump(immediate,model,opt,state,g,a,cursor+horizon)
                    del model,opt,state,g,a;gc.collect()
                rows.append(row)
                saved=torch.load(issued,map_location='cpu',weights_only=False)
                model,opt,state,g,a=base.actor(saved);cursor=saved['cursor'];del saved
                if continuation:
                    bundle=snapshot(model,opt,state,g,a,cursor,critic,meta,cs,cg,ca,audit_rng)
                    bundle.update(ordinary=torch.load(ordinary,map_location='cpu',weights_only=False),
                                  immediate=torch.load(immediate,map_location='cpu',weights_only=False),rows=rows,
                                  counters=counters,work=dict(base.WORK),start_cursor=start_cursor,parent_sha256=parent_digest,
                                  source_sha256=loaded,recorded_process_s=past+time.monotonic()-started,
                                  max_rss_kb=max(old_peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                    atomic(progress,bundle);del bundle
                print(json.dumps(dict(episode=episode,cursor=cursor,service_sign=sign,
                                      audited_advantage=target,predicted_advantage=teaching['predicted_audit_advantage'])),flush=True)
            if continuation:
                ordinary_final=ROOT/f'experiments/results/credit/{args.tag}.ordinary.pt'
                immediate_final=ROOT/f'experiments/results/credit/{args.tag}.immediate.pt'
                for source,destination in [(ordinary,ordinary_final),(immediate,immediate_final)]:
                    if destination.exists():assert sha(source)==sha(destination)
                    else:shutil.copyfile(source,destination)
        checkpoint=ROOT/f'experiments/results/credit/{args.tag}.pt'
        bundle=snapshot(model,opt,state,g,a,cursor,critic,meta,cs,cg,ca,audit_rng)
        bundle.update(source_sha256=loaded,parent_sha256=parent_digest);atomic(checkpoint,bundle)
        result=dict(status='completed',battle='R1',stage='calibrated native reciprocal prospective continuation' if continuation else 'calibrated native reciprocal acquired gate',
                    rows=rows,parent_result=args.parent,parent_sha256=parent_digest,parent_cursor=start_cursor,final_cursor=cursor,
                    checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),target_counts=counters,
                    native_actor_work_counts=dict(base.WORK),native_credit_work_counts={k:v for k,v in cs.items() if isinstance(v,int)},
                    actor_storage=model.core.storage(state),credit_storage=critic.core.storage(cs),
                    critic_audit_mse=sum(r['audit_squared_error'] for r in rows)/len(rows),
                    zero_predictor_audit_mse=sum(r['actual_audit_advantage']**2 for r in rows)/len(rows),
                    service_actions=sum(r['service_sign']!=0 for r in rows),
                    mean_service_advantage=sum(r['service_advantage'] for r in rows)/len(rows),
                    scope='Native eight-layer actor and credit core, preserved numerical state and actor Adam. Neutral action-scaled least-squares response; fresh credit-optimizer moments after objective change. Independent full-support audits provide labels after outcomes. Uncertainty is a heuristic with changing native features, not a coverage guarantee. One-parent TRAIN development; no TEST, public, matched-work or hardware claim. Local eligibility and producer replay retain their declared conditional scope.')
        if continuation:
            actual=[v for r in rows for v in r['support_losses']+r['issued_losses']]
            ordinary_losses=[v for r in rows for v in r['ordinary_losses']]
            immediate_losses=[v for r in rows for v in r['immediate_losses']]
            result.update(prequential_nll=sum(actual)/len(actual),ordinary_nll=sum(ordinary_losses)/len(ordinary_losses),
                          immediate_nll=sum(immediate_losses)/len(immediate_losses),
                          gain_vs_ordinary=(sum(ordinary_losses)-sum(actual))/len(actual),
                          gain_vs_immediate=(sum(immediate_losses)-sum(actual))/len(actual),
                          ordinary_checkpoint=str(ordinary_final.relative_to(ROOT)),ordinary_checkpoint_sha256=sha(ordinary_final),
                          immediate_checkpoint=str(immediate_final.relative_to(ROOT)),immediate_checkpoint_sha256=sha(immediate_final))
        else:result['gate']='PASS'
    result.update(source_sha256=loaded,elapsed_s=past+time.monotonic()-started,
                  max_rss_kb=max(old_peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  accounting='All actor/audit/reference target counts and native counters paid; recorded process intervals exclude unsaved interruption tails. No complete arithmetic or energy estimate.')
    writing=out.with_suffix('.writing');writing.write_text(json.dumps(result,indent=2)+'\n');os.replace(writing,out)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','source_sha256']}),flush=True)


if __name__=='__main__':main()
