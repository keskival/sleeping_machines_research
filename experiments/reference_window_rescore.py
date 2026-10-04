"""Source-bound, inference-only E64 checkpoint rescore on native T256 windows.

Preparation uses the standard library. Numerical execution requires the physical
AWS reservation, one-thread guards and immutable parent/weights/data/source bytes.
No fitting, checkpoint selection or parameter updates are performed.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
from sparse_validation_recovery import physical_reservation, verify_guards


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def windows(n, width=256):
    if width < 2 or width % 2 or n <= width + 1:
        raise ValueError('Even context and at least one full window required')
    for start in range(0, n-width-1, width//2):
        yield start, 0 if start == 0 else width//2, width


def model_tree(path):
    classes = [n for n in ast.parse(Path(path).read_text()).body
               if isinstance(n, ast.ClassDef) and n.name in ('LSTMLM', 'TfLM')]
    if {n.name for n in classes} != {'LSTMLM', 'TfLM'}:
        raise ValueError('Exact E64 model definitions required')
    return ast.Module(body=classes, type_ignores=[])


def prepare(parent, checkpoint, tag):
    if Path(tag).name != tag or not tag.startswith('aws_'):
        raise ValueError('Unique AWS tag required')
    parent = Path(parent); checkpoint = Path(checkpoint)
    for path in (parent, checkpoint):
        if path.is_absolute() or '..' in path.parts or not (ROOT/path).is_file():
            raise ValueError('Existing repository-relative parent/checkpoint required')
    saved = json.loads((ROOT/parent).read_text()); a = saved['args']
    if saved.get('status', 'completed') != 'completed' or a['model'] not in ('lstm', 'tf') or a['ctx'] != 256:
        raise ValueError('Completed E64 T256 parent required')
    directory = ROOT/'experiments/queue'/tag
    if directory.exists(): raise ValueError('Preserve prior queue')
    sources = ('experiments/reference_window_rescore.py', 'experiments/sparse_validation_recovery.py',
               'experiments/e64_lm_baselines.py', 'experiments/lm_training_flops.py', 'experiments/queue/run_safe.sh')
    n = a['test']
    with (ROOT/'data/text8/text8').open('rb') as handle:
        handle.seek(95_000_000); data = handle.read(n)
    if len(data) != n or any(c != 32 and not 97 <= c <= 122 for c in data):
        raise ValueError('Exact valid test interval required')
    output = f'experiments/results/reference_window_rescore/{tag}.json'
    record = dict(status='prepared_not_launched', host='ip-172-31-47-132', tag=tag,
                  parent=str(parent), parent_sha256=sha(ROOT/parent), checkpoint=str(checkpoint),
                  checkpoint_sha256=sha(ROOT/checkpoint), result=output,
                  sources={p: sha(ROOT/p) for p in sources},
                  data_sha256=hashlib.sha256(data).hexdigest(), test=[95_000_000, 95_000_000+n],
                  args=a, targets=sum(end-begin for _, begin, end in windows(n)),
                  guards=dict(address_space_kib=3_000_000, rss_kib=1_250_000,
                              min_available_mib=8192, timeout_s=1800),
                  scope='Same saved weights and training work; reset T256 windows, exactly native target positions. '
                        'Original continuous-state/tail scores are retained. No training or test-driven selection.')
    directory.mkdir()
    manifest = directory/'manifest.json'; queue = directory/'job.txt'
    queue.write_text(f"{tag} experiments/reference_window_rescore.py run --manifest {manifest.relative_to(ROOT)}\n")
    record.update(queue=str(queue.relative_to(ROOT)), queue_sha256=sha(queue))
    manifest.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    env = 'MEM_CAP_KB=3000000 MEM_CAP_RSS_KB=1250000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=1800'
    (directory/'README.md').write_text('# Identical-window reference rescore — prepared, unrun\n\n'
        'P0 protocol repair. Physical AWS owner admits this one inference-only job into a free guarded slot, '
        'after the current P0 owners. Never start a fourth job. Check available memory and occupancy first.\n\n'
        f'```sh\n{env} bash experiments/queue/run_safe.sh {queue.relative_to(ROOT)}\n```\n\n'
        '1800 seconds is a conservative ceiling, not a measured ETA. Weights and active drivers remain unchanged. '
        'Source drift requires a new tag and manifest, never weakened pins.\n')
    return record


def preflight(manifest, enforce_host=True):
    record = json.loads(Path(manifest).read_text())
    if enforce_host:
        physical_reservation(record)
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:')))
        verify_guards(record, os.environ, resource.getrlimit(resource.RLIMIT_AS)[0], available)
    for key in ('parent', 'checkpoint', 'queue'):
        path = Path(record[key])
        if path.is_absolute() or '..' in path.parts or sha(ROOT/path) != record[key+'_sha256']:
            raise ValueError('Changed admission input: '+key)
    for path, digest in record['sources'].items():
        if sha(ROOT/path) != digest: raise ValueError('Changed source: '+path)
    with (ROOT/'data/text8/text8').open('rb') as handle:
        start, end = record['test']; handle.seek(start); data = handle.read(end-start)
    if hashlib.sha256(data).hexdigest() != record['data_sha256']:
        raise ValueError('Changed test interval')
    if (ROOT/record['result']).exists(): raise ValueError('Preserve completed output')
    return record, data


def run(manifest):
    manifest = Path(manifest); original_manifest = sha(manifest)
    record, raw = preflight(manifest)
    # Every source, weight, data and host check precedes any numerical import.
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(1)
    namespace = dict(torch=torch, nn=nn, A=27)
    exec(compile(model_tree(ROOT/'experiments/e64_lm_baselines.py'), '<pinned E64 model definitions>', 'exec'), namespace)
    saved = torch.load(ROOT/record['checkpoint'], weights_only=False, map_location='cpu')
    a = record['args']; cfg = saved['args']
    for k in ('model', 'D', 'passes', 'size', 'layers', 'ctx', 'dropout', 'test'):
        if cfg[k] != a[k]: raise ValueError('Checkpoint/parent configuration differs: '+k)
    model = (namespace['LSTMLM'](a['size'], a['dropout']) if a['model'] == 'lstm'
             else namespace['TfLM'](a['size'], a['layers'], a['ctx'], a['dropout']))
    model.load_state_dict(saved['state']); model.eval()
    tokens = torch.tensor([0 if c == 32 else c-96 for c in raw], dtype=torch.long)
    plan = list(windows(len(raw))); total = 0.; targets = 0; calls = 0; started = time.monotonic()
    with torch.no_grad():
        for offset in range(0, len(plan), 32):
            group = plan[offset:offset+32]
            x = torch.stack([tokens[s:s+256] for s, _, _ in group])
            y = torch.stack([tokens[s+1:s+257] for s, _, _ in group])
            logits, _ = model(x)       # no state passed: every lane starts cold
            losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none').reshape(len(group), 256)
            for i, (_, begin, end) in enumerate(group):
                total += float(losses[i, begin:end].sum()); targets += end-begin
            calls += 1
    if targets != record['targets']: raise ValueError('Scored population differs')
    if any(not torch.equal(model.state_dict()[k], v) for k, v in saved['state'].items()):
        raise ValueError('Rescoring changed model weights')
    preflight(manifest)
    if sha(manifest) != original_manifest: raise ValueError('Manifest changed during rescore')
    result = dict(status='completed', test_bpc=total/targets/math.log(2), original_test_bpc=json.loads((ROOT/record['parent']).read_text())['test_bpc'],
                  test_targets=targets, eval_segment=256, state='reset per T256 window',
                  test=record['test'], evaluated_positions=len(plan)*256, evaluator_calls=calls,
                  parent=record['parent'], parent_sha256=record['parent_sha256'],
                  checkpoint=record['checkpoint'], checkpoint_sha256=record['checkpoint_sha256'],
                  manifest=str(manifest.relative_to(ROOT)), manifest_sha256=original_manifest,
                  source_sha256=record['sources'], data_sha256=record['data_sha256'],
                  optimizer_updates=0, wall_s=time.monotonic()-started, scope=record['scope'])
    if not math.isfinite(result['test_bpc']): raise ValueError('Nonfinite result')
    out = ROOT/record['result']; out.parent.mkdir(exist_ok=True)
    with out.open('x') as handle: handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--parent', required=True)
    prep.add_argument('--checkpoint', required=True); prep.add_argument('--tag', required=True)
    fit = sub.add_parser('run'); fit.add_argument('--manifest', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.parent, args.checkpoint, args.tag), indent=2))
    else: run(ROOT/args.manifest)
