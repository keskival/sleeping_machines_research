"""Resume frozen B4 scoring,gate the memory repair,and diagnose native hierarchy."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    common=dict(task_stationarity='Official stationary fixed-weight B4 protocol;hierarchy is independent stationary scenes.',state_history='B4 retains complete within-sequence causal histories. Hierarchy reuses saved features;no resets or actor changes.',online_adaptation='No online adaptation in official frozen-test scoring. Diagnostic optimizer gate uses TRAIN only;no new model acquisition.',resources='Named AWSsix-slot coordinator,one thread/physical core,RSS3MKiB,VMS12MKiB,8GiB reserve,watchdogs. No external model or purchase.',restrictions='Official fixed-weight comparisons do not imply online wins. Microbatch changes stochastic coupling but preserves dropout/jitter laws;deterministic full-update parity tested with both disabled. No context truncation or per-split test selection.')
    slots={}
    specs=[('4','aws_b4x_wiki_test_20261016_v2','experiments/tpp/b4x_score_frozen_20261016.py','--dataset wikipedia_filtered --variants aws_b4x_wiki_ptr_20261015_v1,aws_b4x_wiki_ptrlc_20261015_v1,aws_b4x_wiki_marks_20261015_v1','tpp_b4',2700),('5','aws_b4_github_microbatch_gate_20261016_v1','experiments/tpp/b4_microbatch_gate_20261016.py','','tpp_b4',900),('3','aws_g_hier_grounding_diagnosis_20261016_v1','experiments/hierarchy/grounding_completed_diagnosis.py','','hierarchy',300)]
    for slot,tag,driver,args,folder,timeout in specs:
        bindings={driver, 'scripts/admit_benchmark_continuation_20261016.py'};requires=[]
        if 'wiki_test' in tag:
            assert not (ROOT/'experiments/results/tpp_b4/aws_b4x_wiki_test_20261015_v1.json').exists()
            assert not (ROOT/'experiments/queue/logs/aws_b4x_wiki_test_20261015_v1.log').exists(),'original TEST may have executed'
            val={}
            for variant in ['ptr','ptrlc','marks']:
                records=[]
                for split in range(5):
                    name=f'experiments/results/tpp_b4/aws_b4x_wiki_{variant}_20261015_v1_s{split}.json';r=json.loads((ROOT/name).read_text());assert r['status']=='completed'
                    for file,digest in r['source_sha256'].items():assert sha(ROOT/file)==digest,(file,'model source drift')
                    bindings.update(r['source_sha256']);bindings.add(name);bindings.add(r['checkpoint']);records.append(r)
                    requires.append(dict(result=name))
                val[variant]=sum(r['val']['total'] for r in records)/5
            declaration=dict(common,battle='B4',strongest_fighter='Select pointer/locally-clipped-pointer/mark-memory variant by the frozen five-split validation mean. All15development fits completed,model sources unchanged.',mechanisms='Log-domain clocks,recording-cell rules,persistent addressed mark state,episodic pointer race if validation selects it. Auxiliary dense work included;not a full sparse/counterfactual-learning demonstration.',decision='Restore already-authorized once-only developed TEST after original scorer failed its source preflight without execution. No recipe,variant-list,split or selection change.',validation_means=val,original_blocker='Old test job reports b4x_score.py source mismatch;no old execution log or result. Separate frozen scorer preserves unrelated current edits.')
        elif 'microbatch' in tag:
            source=json.loads((ROOT/'experiments/results/tpp_b4/aws_b4x_github_ptr_20261015_v1_s0.json').read_text())
            bindings.update(source['source_sha256']);bindings.update(['experiments/tpp/race_tpp_b4x2.py','tests/test_b4_microbatch_update.py'])
            declaration=dict(common,battle='B4',strongest_fighter='Existing native logspread/addressed-mark/episodic-pointer actor on actual longest32Github TRAIN sequences;row-microbatch2048padded-event budget.',mechanisms='Full sequence state and timing history,mark memory,pointer andsilence likelihood retained. Global clipping after accumulated gradients.',decision='Repair measured3.389GB RSS failure at a3MKiB cap without shortening histories. One actual update gate only;no replacement fits automatically admitted.')
        else:
            bindings.update(['experiments/hierarchy/grounding_decoder_calibration.py','sleeping_machines/sparse_geometry_teacher.py','sleeping_machines/grounded_native_composition.py','experiments/hierarchy/part_combination_task.py'])
            for arm in ['joint','continued','decoder_only']:
                name=f'experiments/results/hierarchy/aws_g_hier_grounded_{arm}_20261016_v1.json';r=json.loads((ROOT/name).read_text());assert r['status']=='completed';bindings.update([name,r['feature_packet']['path']]);requires.append(dict(result=name,stage='grounded_composition_learning'))
            bindings.add('experiments/results/hierarchy/aws_g_hier_grounding_variance_20261016_v1.json')
            declaration=dict(common,battle='G-HIER',strongest_fighter='Completed three native arms with the same existing linear geometry decoder,TRAIN-fitted at fixed ridge0.01,using saved root representations.',mechanisms='No actor fitting;native mechanisms unchanged. TEST excluded.',decision='Distinguish newly learned geometry from decoder conditioning and held-out class transfer before choosing a repair.')
        admission=ROOT/f'experiments/{"tpp" if folder=="tpp_b4" else "hierarchy"}/{tag}_admission.json';admission.write_text(json.dumps(declaration,indent=2)+'\n');bindings.add(str(admission.relative_to(ROOT)))
        q=ROOT/f'experiments/queue/{"b4" if folder=="tpp_b4" else "hierarchy"}/{tag}/{tag}.txt';q.parent.mkdir(parents=True);q.write_text('# '+declaration['decision']+'\n'+tag+' '+driver+' --tag '+tag+(' '+args if args else '')+'\n')
        slots[slot]=[dict(name=tag,kind='b4_job' if folder=='tpp_b4' else 'hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={f:sha(ROOT/f) for f in sorted(bindings)},result=f'experiments/results/{folder}/{tag}.json',requires=requires,timeout_s=timeout,rss_kb=3000000,vms_kb=12000000)]
    marker=PLAN/'addenda'/'000_benchmark_continuation_20261016_v1.json';marker.write_text(json.dumps(dict(host='ip-172-31-47-132',battle='B4/G-HIER',purpose='Unblock strongest native developed attempts and evidence decisions without external training.',parent_manifest_sha256=sha(PLAN/'manifest.json'),slots=slots),indent=2)+'\n');print(marker.relative_to(ROOT))
if __name__=='__main__':main()
