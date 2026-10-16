"""R1 numerical integration admission; no acquisition or competitive scoring."""
import argparse,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    assert os.uname().nodename=='ip-172-31-47-132'
    active=json.loads((PLAN/'worker_recovery.aws6.status.json').read_text())['active'];assert '5' not in active
    assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))>8*1024*1024
    ap=argparse.ArgumentParser();ap.add_argument('--packed',action='store_true');ap.add_argument('--targets',type=int,default=32);ap.add_argument('--tag',default='aws_r1_queryable_native_gate_20261016_v2');args=ap.parse_args();name=args.tag
    sources=['sleeping_machines/queryable_local_deep_race.py','tests/test_queryable_local_deep_race.py',
        'experiments/hierarchy/queryable_native_language_gate.py','experiments/r1_local_native_deep.py',
        'experiments/r1_cerl.py','sleeping_machines/local_deep_race.py',
        'sleeping_machines/native_receiver_eligibility.py','sleeping_machines/bound_native_composition.py',
        'sleeping_machines/relational_frame_composition.py','sleeping_machines/native_cause_bank.py',
        'sleeping_machines/native_indexed_cause_head.py',
        'experiments/theory/QUERYABLE_NATIVE_LANGUAGE_INTEGRATION.md']
    if args.packed:sources+=['sleeping_machines/packed_queryable_local_deep_race.py','tests/test_packed_queryable_local_deep_race.py']
    purpose=dict(battle='R1',stationarity='Stationary corpus numerical TRAIN integration,not competitive/online comparison.',
        fighter='Existing native8layercore plus actual version-bound lower queries,existing joint role composer and historical own-write read credit;normalized persistent cause head. Zero SSM/attention/MLP carrier.',
        adaptation='Two16targetscore-before-updateAdamWsteps;receiver state,eligibility,episodic bank,inbox and bound lineage persist. Synchronous optimizer;no recurrentBPTT;no exact changing-parameter or arbitrary cross-producer derivative.',
        retained='Native independent event/race delays,complex temporal transport,sparse receiver writes,separate keys/values,selected plus sampled losing proposals,local counterfactual credit,dormant concepts,persistent indexed recall and queryable lower native publications.',
        restrictions='All8layers execute each token,all declared lower publication keys scored,up to2selected distinct versions/one losing value each;16root versions/layer but transitive lineage may grow. Query roles not semantic identity or autonomous emission policy. Numerical reference,no DEV/TEST or acquisition.',
        accounting='Slot5one thread,3MKiBRSS/12MKiBVMS,8GiBhostreserve,900stimeout. Report key scoring,queryreads/losers,transport,pair products,eligibility,RSS and retained publication/trace bytes;no completeFLOP/energyclaim.',
        decision='Can existing queryable hierarchy mechanisms work inside the actual native language core with historical own-write credit and persistent state across optimizer boundaries? Pass admits planning a data-adequate acquisition,not an automatic fit.')
    assert args.targets>=32 and args.targets%16==0
    if args.targets>32:
        purpose['adaptation']=str(args.targets//16)+' successive16targetTRAINscore-before-updateAdamWsteps;all native/query/publication/index/inbox state persists. Own-write credit conditional;changing-parameter history approximate. No recurrentBPTT;optimizer synchronous.'
        purpose['decision']='Does query lineage/retained historical eligibility remain inside the declared memory/time envelope across repeated optimizer boundaries? Stress correctness/resources only,no quality result or automatic acquisition.'
    q=ROOT/f'experiments/queue/r1/{name}/{name}.txt';q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists()
    q.write_text('# R1 versioned lower-memory native integration gate;no acquisition.\n'+name+' experiments/hierarchy/queryable_native_language_gate.py --tag '+name+(' --packed' if args.packed else '')+' --targets '+str(args.targets)+'\n')
    job=dict(name=name,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in sources},
        result=f'experiments/results/diagnostics/{name}.json',requires=[],timeout_s=900,rss_kb=3000000,vms_kb=12000000)
    out=PLAN/'addenda'/('000_'+name+'.json');assert not out.exists()
    out.write_text(json.dumps(dict(host=os.uname().nodename,battle='R1',purpose=purpose,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'5':[job]}),indent=2)+'\n')
    print('Admitted R1 queryable-native integration gate,slot5;no acquisition.')
if __name__=='__main__':main()
