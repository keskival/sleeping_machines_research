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
from persistence_world_credit import PersistenceWorldCreditSensor
from public_stream import persistence
from world_credit_run import run


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    ap.add_argument('--arm', choices=['persistence'], required=True)
    ap.add_argument('--case', type=int, required=True); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    assert 0 <= a.case < 8
    torch.set_num_threads(1); begin = time.monotonic()
    out = RESULTS/(a.tag+'.json'); assert not out.exists()
    source = verify_result(ACQUISITION); assert source['checks']['acquisition_gate']
    smoke = verify_result('curie_persistence_credit_20261012_v1_smoke')
    assert smoke['smoke'] and smoke['rows'][0]['persistence_prediction_parity_error'] == 0.
    checkpoint = ROOT/source['checks']['checkpoint']; assert sha(checkpoint) == source['checks']['checkpoint_sha256']
    data = json.loads(PARENT.read_text()); cases = []
    for split in ['heldout_graph', 'heldout_factors', 'heldout_depth', 'heldout_factors_depth']:
        base = [r for r in data['records'] if r['split'] == split and r['variant'] == 'base']
        rewire = [r for r in data['records'] if r['split'] == split and r['variant'] == 'rewire']
        cases.extend([base[0], rewire[1]])
    row = cases[a.case]; path = ROOT/row['path']; assert sha(path) == row['sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    torch.manual_seed(194)
    model = PersistenceWorldCreditSensor(saved['confidence']); model.load_state_dict(saved['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    incoming = fingerprint(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
    assert incoming == source['checks']['learner_fingerprint']
    measured = run(model, path, 'learned', optimizer, seed=777+a.case, max_queries=16 if a.smoke else None)
    assert measured['targets'] == (16 if a.smoke else 128)
    diagnostic, _ = persistence(path, (model.persistence_correct, model.persistence_total), True)
    expected = {q:(when, confidence if bit else 1-confidence) for when,q,bit,confidence in diagnostic['issued_predictions']}
    receipts = model.last_persistence_receipts
    parity = max(abs(r['probability']-expected[r['query_id']][1]) for r in receipts)
    assert parity < 1e-12 and all(r['issued_time']==expected[r['query_id']][0] for r in receipts)
    measured['persistence_credit_receipts'] = receipts
    measured['persistence_prediction_parity_error'] = parity
    measured.update(split=row['split'], variant=row['variant'], depth=row['depth'], width=row['width'],
                    path=row['path'], incoming_learner_fingerprint=incoming)
    sources = dict(source['source_sha256'])
    for name in ['experiments/causal_induction/world_credit_case.py', 'experiments/causal_induction/WORLD_CREDIT_RESOURCE_REPAIR.md', 'experiments/causal_induction/persistence_world_credit.py', 'experiments/causal_induction/persistence_credit_case.py', 'experiments/causal_induction/persistence_credit_case_v2.py', 'experiments/causal_induction/PERSISTENCE_CREDIT_PROTOCOL.md']:
        sources[name] = sha(ROOT/name)
    result = dict(status='completed', battle='B5/R1', tag=a.tag, arm=a.arm, case=a.case,
                  rows=[measured], targets=measured['targets'], smoke=a.smoke, source_sha256=sources,
                  parent_sha256=sha(PARENT), data_records=[row],
                  acquisition_result_sha256=sha(RESULTS/(ACQUISITION+'.json')),
                  source_checkpoint_sha256=sha(checkpoint),
                  whole_pipeline_wall_s=time.monotonic()-begin,
                  peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Independent persistence-posterior credit intervention; native actor/world/memory math retained, source-choice mean deliberately changed as registered. Same common source, optimizer, seeds and causal8-feedback updates. V1 smoke scope copied a predecessor description incorrectly; its arm, protocol and source hashes correctly identify the persistence intervention. V2 changes only metadata and smoke-admission checking, no learning operations. RSS2.2GB/9GiBhostfloor/cgroup2GiB; no efficiency claim.')
    assert result['peak_rss_kb'] < 2200000
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(tag=a.tag, case=a.case, arm=a.arm, nll=measured['nll'], world_nll=measured['world_nll'], wall_s=measured['wall_s'], peak_rss_kb=result['peak_rss_kb'])), flush=True)


if __name__ == '__main__': main()
