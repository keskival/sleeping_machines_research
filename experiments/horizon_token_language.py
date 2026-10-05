"""Preserve the initial model as an eligible development-selected checkpoint.

Delegates learning to the source-pinned sampled driver. Selection is recorded
beside its original result; existing evidence and checkpoints stay unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
import horizon_token_language_engine as lab


def select_checkpoint(curve, initial_path, trained_path):
    initial = curve[0]
    trained = min(curve[1:], key=lambda row: row['dev_nll'])
    if initial['dev_nll'] <= trained['dev_nll']:
        return dict(step=0, dev_nll=initial['dev_nll'], checkpoint=str(initial_path))
    return dict(step=trained['step'], dev_nll=trained['dev_nll'], checkpoint=str(trained_path))


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--tag', required=True); p.add_argument('--resume')
    a, _ = p.parse_known_args()
    out = lab.ROOT / 'experiments/results/token_language' / a.tag
    selection = out.with_suffix('.selection.json')
    initial = out.with_suffix('.initial.pt')
    if selection.exists() or initial.exists():
        raise FileExistsError('Selection artifacts must have a unique run tag')
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if a.resume:
        prior = json.loads(Path(a.resume).with_suffix('.selection.json').read_text())
        if prior['selection_source_sha256'] != source_hash:
            raise ValueError('Selection wrapper source changed before resume')
    parent = lab.Model
    class InitialModel(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            initial.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), initial)
    lab.Model = InitialModel
    try:
        lab.main()
    finally:
        lab.Model = parent
    raw = out.with_suffix('.json')
    if not raw.exists(): raw = out.with_suffix('.partial.json')
    result = json.loads(raw.read_text())
    chosen = select_checkpoint(result['curve'], initial, result['best_checkpoint'])
    record = dict(status=result['status'], selection_source_sha256=source_hash,
                  original_result=str(raw), selected=chosen,
                  original_best_trained_dev_nll=result['best_dev_nll'],
                  scope='Same learning trajectory; initialization included in development selection')
    selection.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__': main()
