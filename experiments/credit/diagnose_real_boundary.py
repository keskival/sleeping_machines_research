"""B10 replay diagnosis: observe original scores without changing parameters or data."""
import argparse,hashlib,json,heapq,resource,time
from pathlib import Path
import torch
import real_learned_boundary_v2 as control
import real_learned_boundary_v3 as learned
ROOT=Path(__file__).resolve().parents[2]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();a.seed=183;a.targets=8192;a.context=128;a.group=32;a.audit_probability=.25;torch.set_num_threads(1)
    out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();rows=[];start=time.monotonic()
    original=control.revealed_log_probability
    for module,arm in [(control,'bptt'),(learned,'calibrated_audit')]:
        top=[];count=[0]
        def observe(model,forecast,t,mark):
            value=original(model,forecast,t,mark);count[0]+=1;params=forecast['clocks'];score=float(-value.detach().mean())
            packet=dict(scored_index=count[0]-1,parameter_version=forecast['parameter_version'],nll=score,gap_s=float((t-forecast['time']).detach()[0]),mark=int(mark[0]),log_rate_max=float(params[0].detach().max()),weighted_log_rate_max=float((params[0]+params[3][...,:model.n_exp]).detach().max()),mu_min=float(params[1].detach().min()),mu_max=float(params[1].detach().max()),sigma_min=float(params[2].detach().min()),sigma_max=float(params[2].detach().max()))
            heapq.heappush(top,(score,count[0],packet))
            if len(top)>12:heapq.heappop(top)
            return value
        module.revealed_log_probability=observe
        row=module.run(a,arm,a.seed);module.revealed_log_probability=original;row['worst_forecasts']=[x[2] for x in sorted(top,reverse=True)];rows.append(row)
    sources=['experiments/credit/diagnose_real_boundary.py','experiments/credit/real_learned_boundary_v2.py','experiments/credit/real_learned_boundary_v3.py','experiments/credit/real_online_episode_bridge.py','experiments/market/episode_state.py','experiments/market/b10_tpp.py','experiments/tpp/race_tpp_v5.py']
    out.write_text(json.dumps(dict(status='completed',battle='B10/R1',tag=a.tag,rows=rows,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},scope='Deterministic original seed183 TRAIN replay; forecast observer only; no repair, TEST or headline replacement.'),indent=2)+'\n')
if __name__=='__main__':main()
