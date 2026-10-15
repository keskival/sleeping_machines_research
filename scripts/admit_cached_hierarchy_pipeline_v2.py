"""Admit cached no-fit preflight then conditional64-update learning probe."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
BASE='aws_g_hier_cached_20261015_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 declaration=dict(status='validated_for_conditional_pipeline',battle='G-HIER',
  task_stationarity='Independent gesture episodes; bounded FIT memorization diagnostic, no online-stream or competitive verdict',
  strongest_fighter='32-cell/eight-level addressable persistent parent with exact suffix categorical credit; bitwise logit/gradient contracts against full replay; no shallow substitute',
  adaptation='AdamW3e-4,64updates,batch2,training labels from one official FIT recording only; no selection/DEV/TEST; no external-model training',
  state_history='Same first1s,8bins,128x128polarity evidence; all-level states within refinement. Independent episodes reset; returned memory and numerical credit boundaries remain separate.',
  retained=['temporal decay/rotation adapter','persistent input and all-lower-level memory','independent hard role races','separate keys/values','source and pose metadata','relational composition','content transport','delayed top-down context','exact final-task alternative suffix credit','cell-balanced full-support audits'],
  missing=['raw asynchronous dispatch','content-indexed discovery','dynamic part ownership','unsampled losing-value derivative updates','complete FLOP ledger','learned semantic transfer'],
  restrictions='Dense tensor reference and fixed9-address/source stencil.32 learned cells/8abstractionlevels/4refinements, width32. Small FIT recording is explicit learning diagnostic, not strongest-family benchmark verdict.',
  resources='CPU-only existing coordinator slot6;1core/thread,RSS3000000KiB,VMS12000000KiB,8GiBreserve,watchdog. Preflight requires1.25RSSmargin and64update estimate with1.5margin under5400s. Probe timeout7200s with exact8update checkpoints.',
  decision='Can the full deep construction learn its FIT recording with exact route credit inside the measured resource envelope? Select learning/variance/transport repair before a larger fit.',
  supersedes='Unstarted cached_v1 pipeline and full-replay preflight; older local-teacher spatial smoke/fit retired. Active training and reusable preparation preserved. v2 preflight measures batch2 exactly as the probe.')
 admission=ROOT/f'experiments/hierarchy/{BASE}_admission.json'
 with admission.open('x') as f:json.dump(declaration,f,indent=2);f.write('\n')
 names=['sleeping_machines/cached_parent_hierarchy.py','sleeping_machines/addressable_parent_hierarchy.py','sleeping_machines/parent_race_hierarchy.py','sleeping_machines/spatial_race_hierarchy.py','experiments/hierarchy/visual_representation.py','experiments/hierarchy/cached_preflight_v2.py','experiments/hierarchy/cached_learning_probe_v2.py','tests/test_cached_parent_hierarchy.py','experiments/theory/EXACT_HIERARCHICAL_SUFFIX_REPLAY.md','scripts/admit_cached_hierarchy_pipeline_v2.py',str(admission.relative_to(ROOT))]
 bindings={n:sha(ROOT/n) for n in names};jobs=[]
 pretag=BASE+'_preflight';preresult=f'experiments/results/hierarchy/{pretag}.json'
 for stage in ['preflight','probe']:
  tag=BASE+'_'+stage;q=ROOT/f'experiments/queue/hierarchy/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=False)
  command=(f'experiments/hierarchy/cached_preflight_v2.py --tag {tag}' if stage=='preflight' else f'experiments/hierarchy/cached_learning_probe_v2.py --tag {tag} --preflight {preresult} --steps64'.replace('--steps64','--steps 64'))
  with q.open('x') as f:f.write('# G-HIER: '+declaration['decision']+'\n'+tag+' '+command+'\n')
  jobs.append(dict(name=tag,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=bindings,result=f'experiments/results/hierarchy/{tag}.json',requires=[] if stage=='preflight' else [dict(result=preresult,stage='cached_preflight',rss_margin=1.25)],timeout_s=1800 if stage=='preflight' else 7200,rss_kb=3000000,vms_kb=12000000))
 marker=PLAN/'addenda'/('000_'+BASE+'.json')
 with marker.open('x') as f:json.dump(dict(host='ip-172-31-47-132',battle='G-HIER',purpose=declaration['decision'],parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'6':jobs}),f,indent=2);f.write('\n')
 print(json.dumps(dict(addendum=str(marker.relative_to(ROOT)),jobs=[j['name'] for j in jobs])))
if __name__=='__main__':main()
