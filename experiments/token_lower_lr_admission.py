"""Admit one recipe repair from completed 256K evidence, never an interim score."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
ROOT=Path(__file__).resolve().parents[1]
BASE='curie_data_growth_tokens_256k_b64_c16_p24_s6_20261005_v1'
TAG='curie_data_growth_tokens_256k_b64_c16_p24_lr001_s6_20261005_v1'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def admit():
    path=ROOT/f'experiments/results/token_language/{BASE}.json'
    result=json.loads(path.read_text())
    selection=json.loads(path.with_suffix('.selection.json').read_text())
    assert result['status']==selection['status']=='completed'
    assert result['presentations_total']==524272 and result['args']['steps']==1024
    assert result['args']['lr']==.003 and result['args']['seed']==6
    chosen=min(result['curve'],key=lambda row:row['dev_nll']);final=result['curve'][-1]
    assert chosen['step']==selection['selected']['step']
    assert abs(chosen['dev_nll']-selection['selected']['dev_nll'])<2e-6
    assert final['dev_nll']-chosen['dev_nll']>.05
    assert final['train_nll']<chosen['train_nll']
    utility_path=ROOT/'experiments/results/diagnostics/curie_data_growth_tokens_256k_p24_utility_20261005_v1.json'
    utility=json.loads(utility_path.read_text());assert utility['status']=='completed'
    assert utility['rows'][0]['tag']==BASE and utility['rows'][0]['selected']==selection['selected']
    assert utility['rows'][0]['matched_rng'] and utility['rows'][0]['partition_parity']
    assert utility['rows'][0]['fit_targets']==524272 and utility['rows'][0]['dev_targets']==2040
    for name,digest in result['identity']['source_sha256'].items():assert sha(ROOT/name)==digest,name
    queue=ROOT/f'experiments/queue/{TAG}.txt'
    old=shlex.split((ROOT/f'experiments/queue/{BASE}.txt').read_text())
    new=shlex.split(queue.read_text())
    expected=[TAG if word==BASE else word for word in old]+['--lr','0.001']
    assert new==expected,'Only tag and learning rate may change'
    assert not (ROOT/f'experiments/results/token_language/{TAG}.json').exists()
    return dict(status='admitted',queue=str(queue.relative_to(ROOT)),control=BASE,candidate=TAG,
        control_sha256=sha(path),utility_sha256=sha(utility_path),queue_sha256=sha(queue),
        final_minus_selected_nll=final['dev_nll']-chosen['dev_nll'],lr_before=.003,lr_after=.001,
        scope='One same-data/seed/two-pass exposure repair, after admitted FAS repeats. Preserve all mechanisms and fixed selection cadence. Changed recipe is outside fixed-recipe scaling packet. Candidate cost must be audited independently; no quality or FLOP improvement assumed.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    record=admit();path.write_text(json.dumps(record,indent=2)+'\n');print(record['queue'])
