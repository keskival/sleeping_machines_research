"""Source-bound native fitting pipelines; all long fits must pass their own gate."""
from pathlib import Path
import hashlib,json,os,shlex,subprocess
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    jobs=[(1,'aws_r1_rmr_nativeev_K4_16M_s1_20261017_v1',True),(2,'aws_r1_rmr_nativeev_grow_K4_16M_s0_20261017_v1',True),(3,'aws_r1_rmr_nativeev_K4_4M_s0_20261017_v1',True),(5,'aws_r1_gpu_driver_cpu_e2e_smoke_20261017_v1',False),(6,'aws_r1_rmr_native_e2e_cc_smoke_20261017_v1',False)]
    gate='experiments/results/diagnostics/aws_r1_review_repairs_20261017_v15.json';assert json.loads((ROOT/gate).read_text())['gate']=='PASS'
    slots={}
    for slot,old,full in jobs:
        original=ROOT/f'experiments/queue/r1/{old}/{old}.txt';text=original.read_text();command=shlex.split([x for x in text.splitlines() if x and not x.startswith('#')][-1]);tag=old.replace('_v1','_awsfill_v2');command[0]=tag;command[command.index('--tag')+1]=tag
        if full:command+=['--micro','4']
        folder=ROOT/f'experiments/queue/r1/{tag}';folder.mkdir(parents=True,exist_ok=True);candidate=folder/'candidate.txt';candidate.write_text('\n'.join(x for x in text.splitlines() if x.startswith('#'))+'\n'+shlex.join(command)+'\n')
        if full or slot==6:subprocess.run(['python3','scripts/check_native_coverage.py',str(candidate)],cwd=ROOT,check=True)
        purpose=dict(battle='R1',stationarity='One-pass chronological training on stationary FineWeb; fixed-weight, persistent-state and online final scores labelled separately. Deployment-driver job is a CPU-only diagnostic.',fighter='Product-ordered native slope, seed reproducibility, depth growth and cross-lane integration. Main native fits retain every implemented inference mechanism; --micro4 preserves batch/objective. Deployment-driver smoke is diagnostic, not flagship.',online='Pre-update training forecasts; forward/future-trained credit and event-local learning remain declared gaps. Cross-lane smoke retains carry-credit1 with micro1; ordinary fits credit within lanes.',state_history='Carry256, contiguous streams; microbatching does not reset numerical state. No dense-model fit.',restrictions='Native full fits gated at16updates and2.9MKiB RSS; same checkpoint continued. Micro4 for main fits; no model/history reduction. Cross-lane full fit awaits compatible microbatch engineering; current small smoke is its ordered integration gate.',accounting='One thread,3MKiB RSS watchdog,12MKiB VMS,8GiB available host reserve,source-bound run_safe six-slot coordinator. Raw logs/metrics/checkpoints retained.',decision=old+': advance the ordered native scaling evidence or remove its driver integration blocker. No new ablation fits.')
        q=folder/f'{tag}.txt'
        if full:line=f'{tag} experiments/credit/native_gated_fit.py --candidate {candidate.relative_to(ROOT)}'
        elif slot==5:line=shlex.join([tag,'experiments/credit/cpu_driver_smoke.py',*command[2:]])
        else:line=shlex.join(command)
        q.write_text('# '+json.dumps(purpose)+'\n'+line+'\n')
        names=['experiments/r1_rmr.py','experiments/r1_cerl.py','experiments/r1_deep_keyed_lm.py','experiments/native_slot_memory.py','experiments/credit/native_gated_fit.py','scripts/check_native_coverage.py',str(candidate.relative_to(ROOT))]
        if slot==5:names+=['experiments/r1_cerl_gpu.py','experiments/credit/cpu_driver_smoke.py']
        result=f'experiments/results/{"gpu_ladder" if slot==5 else "token_language"}/{tag}.json'
        slots[str(slot)]=[dict(name=tag,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in names},result=result,requires=[dict(result=gate,stage='review repair integrated numerical gate',source_sha256={})],timeout_s=259200 if full else 7200,rss_kb=3000000,vms_kb=12000000)]
    packet=dict(host=os.uname().nodename,battle='R1',parent_manifest_sha256=sha(PLAN/'manifest.json'),slots=slots);p=PLAN/'addenda/000_aws_native_fill_20261017_v2.json';assert not p.exists();t=p.with_suffix('.writing');t.write_text(json.dumps(packet,indent=2)+'\n');t.replace(p)
if __name__=='__main__':main()
