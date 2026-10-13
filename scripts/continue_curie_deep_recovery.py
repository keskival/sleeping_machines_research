"""Resume the same deep acquisition after host reserve recovery, then continue."""
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
import continue_curie_deep_language as original
ROOT=original.ROOT


def stable_host():
    consecutive=0
    while consecutive<4:
        memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        consecutive=consecutive+1 if memory['MemAvailable']>=10240*1024 else 0
        print(f'Host stability check MemAvailable={memory["MemAvailable"]//1024}MiB; consecutive={consecutive}/4',flush=True)
        if consecutive<4:time.sleep(20)


def main():
    interrupted='curie_allocated_deep_d8_8m_20261013_s6_v1'
    receipt_path=ROOT/f'experiments/results/token_language/{interrupted}_interruption.json'
    receipt=json.loads(receipt_path.read_text())
    checkpoint=ROOT/f'experiments/results/token_language/{interrupted}.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==receipt['checkpoint_sha256']
    rates={depth:original.load(f'curie_allocated_deep_d{depth}_smoke_20261013_s6_v1')['train_wall_s_total']/32768 for depth in (8,16)}
    stages=((8,'8m',16384,receipt['recovery_run']),
            (16,'8m',16384,'curie_allocated_deep_d16_8m_20261013_s6_v1'),
            (8,'32m',65536,'curie_allocated_deep_d8_32m_20261013_s6_v1'))
    for depth,size,steps,tag in stages:
        stable_host()
        timeout=math.ceil(2*rates[depth]*steps*512+3600)
        if timeout>604800:raise RuntimeError('Measured deep fit exceeds7day operational envelope')
        original.run(tag,max(timeout,5400));packet=original.load(tag)
        if tag==receipt['recovery_run']:
            packet['resource_interruption']=receipt
            packet['whole_fit_process_wall_s']=receipt['runner_interval_s']+packet['process_wall_s']
            packet['whole_fit_work_scope']='Interrupted runner interval (1s timestamp resolution) plus resumed process wall. Successful-trajectory training timer alone excludes discarded work and setup/evaluation.'
            path=ROOT/f'experiments/results/token_language/{tag}.json'
            tmp=path.with_suffix('.work.tmp');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(path)
        quality=tag+'_quality';original.run(quality,5400);original.load(quality,'diagnostics')
        subprocess.run(['git','add',f'experiments/results/token_language/{tag}.json',f'experiments/results/diagnostics/{quality}.json'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m',f'Complete depth{depth} {size} acquisition with interruption-aware work'],cwd=ROOT,check=True)
        print(f'Completed {tag}: finalDEV {packet["curve"][-1]["dev_nll"]}',flush=True)
    print('Deep acquisition finished; same-depth benchmark comparisons remain the objective.',flush=True)


if __name__=='__main__':main()
