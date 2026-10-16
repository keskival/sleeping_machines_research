"""Complete only memory-failed B4 pointer splits;reuse completed others."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pre='experiments/results/tpp_b4/aws_b4_github_microbatch_gate_20261016_v1.json';gate=json.loads((ROOT/pre).read_text());assert gate['status']=='completed' and gate['max_rss_kib']*1.5<3000000 and not gate['repairs']
    names=set(gate['source_sha256'])|{'experiments/tpp/b4x_resumable_micro.py','experiments/tpp/b4x_manifest_score.py','tests/test_b4_resumable_micro.py','scripts/admit_b4_github_memory_repair.py','scripts/continue_b4_github_memory_repair.py',pre}
    declaration=dict(battle='B4',stationarity='Official stationary fixed-weight benchmark;no online verdict.',strongest_fighter='Native log-domain clock,addressed mark-memory and episodic-pointer actor. Complete only the3memory-failed splits for global/local clipping;retain the original two completed pointer splits and all five mark-memory splits.',adaptation='Same d32/modes16/two layers,batch32,AdamW.003/wd0,200epoch ceiling/30pass validation patience,seed0. Row-microbatch2048,whole-sequence history retained. Local caps calibrated from32TRAIN batches using unclipped accumulated gradients. Dropout/jitter laws retained,coupling changed.',state_history='Complete per-sequence physical-time state;no temporal chunking/context truncation. Exact pass checkpoints save model,optimizer,all RNGs,order,caps and history.',mechanisms='Native clock densities,silence likelihood,addressed mark memory,episodic pointer races. Auxiliary dense work retained and charged;no claim of full sparse/counterfactual-learning integration.',restrictions='Only numerical/memory repair,no architecture/hyperparameter grid. Validation batch1 computes same per-sequence statistic;deterministic parity and stochastic resume contracts pass. No TEST during fitting. Future variant selection uses all five DEV splits.',resources='AWSgym slots5/6,one thread each,RSS3MKiB,VMS12MKiB,12600s each,max fit wall10800s;8GiB host reserve. Real longest32sequence gate1072656KiB. Keep slots1/3/4 available for primary hierarchy/language work.',decision='Finish the documented strongest native developed attempt,select by five-split validation mean,then one frozen TEST evaluation. No weak memory-failed proxy becomes a family verdict.')
    a=ROOT/'experiments/tpp/aws_b4_github_memory_repair_20261016_v1_admission.json';a.write_text(json.dumps(declaration,indent=2)+'\n');names.add(str(a.relative_to(ROOT)));bindings={n:sha(ROOT/n) for n in sorted(names)};slots={}
    for slot,variant in [('5','ptr'),('6','ptrlc')]:
        jobs=[]
        for split in [1,2,3]:
            tag=f'aws_b4x_github_{variant}_20261016_v3_s{split}';q=ROOT/f'experiments/queue/b4/{tag}/{tag}.txt';q.parent.mkdir(parents=True)
            q.write_text('# B4: '+declaration['decision']+'\n'+tag+' experiments/tpp/b4x_resumable_micro.py --tag '+tag+f' --split {split} --event-budget 2048 --max-wall-s 10800'+(' --local-clip' if variant=='ptrlc' else '')+'\n')
            jobs.append(dict(name=tag,kind='b4_job',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=bindings,result=f'experiments/results/tpp_b4/{tag}.json',requires=[dict(result=pre,stage='b4_microbatch_gate')],timeout_s=12600,rss_kb=3000000,vms_kb=12000000))
        slots[slot]=jobs
    marker=PLAN/'addenda'/'000_b4_github_memory_repair_20261016_v1.json';marker.write_text(json.dumps(dict(host='ip-172-31-47-132',battle='B4',purpose=declaration['decision'],parent_manifest_sha256=sha(PLAN/'manifest.json'),slots=slots),indent=2)+'\n');print(marker.relative_to(ROOT))
if __name__=='__main__':main()
