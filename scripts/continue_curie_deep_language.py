"""Curie R1 actual depth8/16 and8M/32M acquisition, sequential run_safe jobs."""
import json
import math
import os
import subprocess
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PREFIX='curie_allocated_deep'
BASE='--exposure fresh --train-tokens 262144 --dev-tokens 2048 --chunk 64 --credit-window 16 --payload 24 --heads 2 --pool 4 --lanes 8 --eval-lanes 8 --readout-init frequency --input-init balanced --future-score-positions 4 --seed 6 --teacher-probability 0.25'


def run(tag,timeout):
    env=dict(os.environ,MEM_CAP_KB='12000000',MEM_CAP_RSS_KB='3000000',MIN_AVAIL_MB='8192',
             CGROUP_RESERVE_MB='2048',JOB_TIMEOUT_S=str(timeout),WAIT='1',WAIT_TIMEOUT_S='120')
    subprocess.run(['bash','experiments/queue/run_safe.sh',f'experiments/queue/r1/{tag}.txt'],cwd=ROOT,env=env,check=True)


def load(tag,kind='token_language'):
    packet=json.loads((ROOT/f'experiments/results/{kind}/{tag}.json').read_text())
    assert packet['status']=='completed'
    return packet


def main():
    # Finish the already running fixed market evidence, including publication.
    while subprocess.run(['tmux','has-session','-t','curie_b10_credit_confirmation_20261013_v1'],capture_output=True).returncode==0:
        time.sleep(20)
    publication=ROOT/'.git/credit-report-preview/curie_b10_addressed_clock_20261013_confirmation_v1_publish/publication.json'
    if not publication.exists():raise RuntimeError('Market cohort publication failed; inspect before taking host')
    rates={}
    for depth in (8,16):
        tag=f'{PREFIX}_d{depth}_smoke_20261013_s6_v1';run(tag,5400);packet=load(tag)
        assert packet['args']['depth']==depth and packet['max_rss_kb']<2500000
        norms=packet['curve'][-1]['receiver_gradient_norms']['banks.input']
        assert len(norms)==depth
        assert all(sum(v for head in layer for v in head)>0 for layer in norms),'Missing actual-stage gradient'
        assert all(math.isfinite(row['dev_nll']) for row in packet['curve'])
        assert sum(packet['depth_credit_allocation']['layer_audits'])==packet['teacher_work']['admitted_calls']
        rates[depth]=packet['train_wall_s_total']/packet['presentations_total']
        print(f'PASS D{depth}: every-layer gradients, finite updates, RSS; seconds/target {rates[depth]}',flush=True)
    for depth,size,steps in ((8,'8m',16384),(16,'8m',16384),(8,'32m',65536)):
        tag=f'{PREFIX}_d{depth}_{size}_20261013_s6_v1'
        timeout=math.ceil(2*rates[depth]*steps*512+3600)
        if timeout>172800:raise RuntimeError('Measured deep fit exceeds48h envelope; optimize execution without shrinking depth')
        run(tag,max(timeout,5400));packet=load(tag)
        print(f'Completed {tag}: finalDEV {packet["curve"][-1]["dev_nll"]}; full training {packet["train_wall_s_total"]}s',flush=True)
        quality=tag+'_quality';run(quality,5400);load(quality,'diagnostics')
        # Preserve evidence without touching another host's report files.
        paths=[f'experiments/results/token_language/{tag}.json',f'experiments/results/diagnostics/{quality}.json']
        subprocess.run(['git','add',*paths],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m',f'Complete native allocated-credit depth{depth} {size} language acquisition and DEV score'],cwd=ROOT,check=True)
    print('Deep acquisition complete. Compare saved same-depth AWS references; public validation remains sealed.',flush=True)


if __name__=='__main__':main()
