"""Source-bound replication of B10 online learned credit, not a benchmark claim."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import torch
import real_learned_boundary_v2 as control
import real_learned_boundary_v3 as learned
ROOT=Path(__file__).resolve().parents[2]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--mode',choices=['smoke','measure'],required=True);a=ap.parse_args();a.targets=8192;a.context=128;a.group=32;a.audit_probability=.25
    if a.mode=='smoke':a.targets=64;a.context=32;a.group=8
    torch.set_num_threads(1);out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();start=time.monotonic()
    rows=[control.run(a,arm,a.seed) for arm in ('bptt','untrained_audit')]+[learned.run(a,'calibrated_audit',a.seed)]
    assert all(r['targets']==a.targets for r in rows);assert rows[1]['audits']==rows[2]['audits']
    sources=['experiments/credit/real_boundary_confirmation.py','experiments/credit/real_learned_boundary_v2.py','experiments/credit/real_learned_boundary_v3.py','experiments/credit/real_online_episode_bridge.py','experiments/market/episode_state.py','experiments/market/b10_tpp.py','experiments/tpp/race_tpp_v5.py']
    out.write_text(json.dumps(dict(status='completed',tag=a.tag,battle='B10/R1',args=vars(a),metrics=dict(rows=rows),whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},scope='Replicated initializations/audit draws on same 8192-event real TRAIN fragment. Online actor and credit fitting; pre-optimizer unbiased residuals; replay and all dense local derivatives charged. Shared unused predictor scaffold in controls prevents optimized BPTT efficiency claim. No TEST or published benchmark verdict.'),indent=2)+'\n')
if __name__=='__main__':main()
