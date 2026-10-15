"""Admit G-HIER contract/data/smoke/deep-fit stages to the existing AWS coordinator.

No direct numerical launch, no change to active jobs, no new external model fit.
"""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
BASE='aws_g_hier_spatial32_20261015_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sources=['sleeping_machines/spatial_race_hierarchy.py','experiments/hierarchy/visual_representation.py',
             'experiments/hierarchy/contracts.py','tests/test_spatial_race_hierarchy.py',
             'experiments/theory/DEEP_REPRESENTATIONAL_COMPOSITION.md','scripts/admit_deep_spatial_hierarchy.py']
    tags={s:BASE+'_'+s for s in ['contracts','prepare','smoke','fit']}
    results={s:f'experiments/results/hierarchy/{tag}.json' for s,tag in tags.items()}
    declaration=dict(battle='G-HIER',objective='Train genuinely deep hierarchical representations at fixed observation evidence',
        task_stationarity='Episode-classification development, fixed subject-disjoint splits; not an online-stream verdict',
        strongest_fighter='New32-block own-family spatial race construction; prior4x4 flattened adapters discard the local arrangements required for this objective. No shallow proxy or external-model fit. This is architecture development; no competitive verdict.',
        adaptation='Weights update on fit users only; selection users17..19; DEV20..23 scored after selection. Distinct gesture episodes reset evidence; no persistent stream comparison claimed.',
        state_history='Same first1s,8bins,128x128polarity inputs for every stage;32learned blocks/eight spatial scales. Temporal horizon not expanded.',
        retained=['temporal input decay/rotation','separate keys/values','hard exponential feature races','two parallel race groups','global deep value differentiation','all-block local losing-candidate credit'],
        missing=['raw-event sparse spatial dispatch','per-node carried write state across event chunks','independent timer emissions','whole-downstream alternative-state-write credit','cat/part labelled validation'],
        restrictions='Dense tensor reference backend and fixed-bin input are explicit development implementations. They isolate learned spatial composition and preserve128resolution. No asynchronous hardware/cat claim.',
        resources='CPU-only; one coordinator slot/physical core, one thread, RSS3M KiB,8GiB host reserve; finite-step smoke gates full-fit runtime/RSS; source hashes and exact epoch-boundary checkpoints retained.',
        decision='Does32-block spatial composition learn useful held-out representations, and which stages contribute? Diagnose before any architecture extension; no automatic grid.',
        publication='First-pass development result only. Do not promote active stage scores into wins or call gesture classification cat recognition.')
    admission=ROOT/f'experiments/hierarchy/{BASE}_admission.json'
    with admission.open('x') as f:json.dump(dict(status='validated_for_pipeline',**declaration),f,indent=2);f.write('\n')
    bindings={n:sha(ROOT/n) for n in sources};bindings[str(admission.relative_to(ROOT))]=sha(admission)
    commands={
        'contracts':f'experiments/hierarchy/contracts.py --tag {tags["contracts"]}',
        'prepare':f'experiments/hierarchy/visual_representation.py --tag {tags["prepare"]} --mode prepare',
        'smoke':f'experiments/hierarchy/visual_representation.py --tag {tags["smoke"]} --mode smoke --data {results["prepare"]} --max-wall-s 1800',
        'fit':f'experiments/hierarchy/visual_representation.py --tag {tags["fit"]} --mode fit --data {results["prepare"]} --smoke-result {results["smoke"]} --epochs 20 --max-wall-s 14400'}
    jobs=[]
    for stage in ['contracts','prepare','smoke','fit']:
        tag=tags[stage];q=ROOT/f'experiments/queue/hierarchy/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=True)
        with q.open('x') as f:f.write('# G-HIER: '+declaration['decision']+'\n'+tag+' '+commands[stage]+'\n')
        requires=[]
        if stage=='prepare':requires=[dict(result=results['contracts'],stage='contracts')]
        if stage=='smoke':requires=[dict(result=results['prepare'],stage='prepare')]
        if stage=='fit':requires=[dict(result=results['smoke'],stage='smoke',rss_margin=1.25)]
        jobs.append(dict(name=tag,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),
                    source_sha256=bindings,result=results[stage],requires=requires,
                    timeout_s={'contracts':300,'prepare':7200,'smoke':1800,'fit':16200}[stage],rss_kb=3000000,vms_kb=12000000))
    packet=dict(host='ip-172-31-47-132',battle='G-HIER',purpose=declaration['decision'],
                parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'2':jobs})
    marker=PLAN/'addenda'/('000_'+BASE+'.json')
    with marker.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
    print(json.dumps(dict(admission=str(admission.relative_to(ROOT)),addendum=str(marker.relative_to(ROOT)),jobs=[j['name'] for j in jobs],slot=2)))


if __name__=='__main__':main()
