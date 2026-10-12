"""One independent world-credit deployment per guarded process; same v1 math."""
import argparse
import copy
import hashlib
import json
import resource
import time
from pathlib import Path
import torch
from native_sensor import ROOT
from world_credit_pipeline import verify_result, fingerprint, ACQUISITION, PARENT, RESULTS, sha
from world_credit_sensor import WorldCreditSensor
from world_credit_run import run


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    ap.add_argument('--arm', choices=['exact', 'learned', 'prior'], required=True)
    ap.add_argument('--case', type=int, required=True); a = ap.parse_args()
    assert 0 <= a.case < 8
    torch.set_num_threads(1); begin = time.monotonic()
    out = RESULTS/(a.tag+'.json'); assert not out.exists()
    source = verify_result(ACQUISITION); assert source['checks']['acquisition_gate']
    checkpoint = ROOT/source['checks']['checkpoint']; assert sha(checkpoint) == source['checks']['checkpoint_sha256']
    data = json.loads(PARENT.read_text()); cases = []
    for split in ['heldout_graph', 'heldout_factors', 'heldout_depth', 'heldout_factors_depth']:
        base = [r for r in data['records'] if r['split'] == split and r['variant'] == 'base']
        rewire = [r for r in data['records'] if r['split'] == split and r['variant'] == 'rewire']
        cases.extend([base[0], rewire[1]])
    row = cases[a.case]; path = ROOT/row['path']; assert sha(path) == row['sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    torch.manual_seed(194)
    model = WorldCreditSensor(saved['confidence']); model.load_state_dict(saved['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    incoming = fingerprint(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
    assert incoming == source['checks']['learner_fingerprint']
    measured = run(model, path, a.arm, optimizer, seed=777+a.case)
    assert measured['targets'] == 128
    measured.update(split=row['split'], variant=row['variant'], depth=row['depth'], width=row['width'],
                    path=row['path'], incoming_learner_fingerprint=incoming)
    sources = dict(source['source_sha256'])
    for name in ['experiments/causal_induction/world_credit_case.py', 'experiments/causal_induction/WORLD_CREDIT_RESOURCE_REPAIR.md']:
        sources[name] = sha(ROOT/name)
    result = dict(status='completed', battle='B5/R1', tag=a.tag, arm=a.arm, case=a.case,
                  rows=[measured], targets=128, source_sha256=sources,
                  parent_sha256=sha(PARENT), data_records=[row],
                  acquisition_result_sha256=sha(RESULTS/(ACQUISITION+'.json')),
                  source_checkpoint_sha256=sha(checkpoint),
                  whole_pipeline_wall_s=time.monotonic()-begin,
                  peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Independent single-case process repair; unchanged v1 native world/actor math, source checkpoint, optimizer, seeds, causal8-feedback update policy and all8declared cases. Completed system evidence persisted immediately. RSS2.2GB/9GiBhostfloor/cgroup2GiB. Failed v1 whole-arm logs retained as execution overhead, not model quality.')
    assert result['peak_rss_kb'] < 2200000
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(tag=a.tag, case=a.case, arm=a.arm, nll=measured['nll'], world_nll=measured['world_nll'], wall_s=measured['wall_s'], peak_rss_kb=result['peak_rss_kb'])), flush=True)


if __name__ == '__main__': main()
