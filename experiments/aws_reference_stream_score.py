"""Bounded CPU scoring of the source-audited published target population.

Run only through a uniquely named run_safe queue after benchmark selection.
Importing the protocol reducer does not import Torch or access public data.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from aws_reference_target_plan import target_windows, validate_plan, REFERENCE_TARGETS


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def reduce_windows(windows, score_window, *, expected_targets=REFERENCE_TARGETS,
                   sequence_targets=262144, batches=5):
    """Enforce exact targets and sequence resets around a model callback.

    Callback returns summed NLL, target count and next numerical state. Never
    average chunk means: a short final chunk must receive its actual weight.
    """
    state = None
    total_loss = 0.
    targets = 0
    count = 0
    previous_batch = -1
    previous_offset_end = 0
    for window in windows:
        if window.batch != previous_batch:
            if window.batch != previous_batch + 1 or window.offset != 0 or not window.reset_sequence:
                raise ValueError('Missing or reordered sequence group')
            state = None
            previous_offset_end = 0
        elif window.reset_sequence:
            raise ValueError('Execution chunk cannot reset sequence state')
        if window.offset != previous_offset_end:
            raise ValueError('Missing or reordered execution chunk')
        loss_sum, scored_targets, state = score_window(window, state)
        if scored_targets != window.scored_targets or not math.isfinite(loss_sum):
            raise ValueError('Every declared target must receive a finite loss')
        if loss_sum < 0:
            raise ValueError('Normalized log-probabilities must have nonnegative NLL')
        total_loss += loss_sum
        targets += scored_targets
        count += 1
        previous_batch = window.batch
        previous_offset_end = window.offset + window.count
    if targets != expected_targets or previous_batch != batches-1 or previous_offset_end != sequence_targets:
        raise ValueError('Public target population incomplete')
    return dict(nll=total_loss / targets, loss_sum=total_loss, targets=targets,
                execution_windows=count, reset_sequence_groups=previous_batch+1)


def native_window(model, data, args, generator, window, state):
    """Common bounded causal operation used by public scoring and contracts."""
    import numpy as np
    import torch
    with torch.no_grad():
        tokens = torch.from_numpy(np.array([data[begin:end] for begin, end in window.lane_load_ranges], dtype=np.int64))
        features, state, _ = model(tokens[:, :-1], state, generator, args,
                                   deterministic=args.deterministic_eval)
        loss = model.readout.nll(features, tokens[:, 1:])
        return float(loss.double().sum()), loss.numel(), state


def validate_selection(result, selection, readiness):
    if result.get('status') != 'completed':
        raise ValueError('Completed native fit required')
    if selection.get('step', 0) <= 0 or not math.isfinite(selection['dev_nll']):
        raise ValueError('Trained development-selected weights required')
    if result['args'].get('unit_gain_scale', 1.) != 1.:
        raise ValueError('This scorer requires the original contracted unit gain')
    if result['args']['state_mode'] != 'carry' or result['args']['compiled']:
        raise ValueError('This scorer implements eager persistent CPU inference')
    if result['args']['dev_offset'] < REFERENCE_TARGETS:
        raise ValueError('Selection development interval overlaps reserved public targets')
    # The larger-data/scaling decision supplies this record after completion.
    # A pilot's .02 learning gain alone does not select a benchmark-ready member.
    if readiness.get('status') != 'completed' or readiness.get('public_scoring_ready') is not True:
        raise ValueError('Completed benchmark-selection record required')
    for gate in ('independent_seed', 'larger_data_quality', 'complete_fit_work',
                 'measured_cpu_inference', 'exact_history_target_protocol'):
        if readiness.get('gates', {}).get(gate) is not True:
            raise ValueError('Missing benchmark-selection evidence: ' + gate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--chunk-targets', type=int, default=64)
    a = p.parse_args()
    manifest = Path(a.manifest)
    packet = json.loads(manifest.read_text())
    pins = packet['source_sha256']
    for name, digest in pins.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Scorer source mismatch: ' + name)
    out = ROOT / 'experiments/results/token_language' / (a.tag + '.public.json')
    if out.exists(): raise FileExistsError(out)
    result_path = ROOT / packet['fit_result']
    selection_path = ROOT / packet['selection_record']
    readiness_path = ROOT / packet['benchmark_selection_record']
    result = json.loads(result_path.read_text())
    selected = json.loads(selection_path.read_text())['selected']
    readiness = json.loads(readiness_path.read_text())
    validate_selection(result, selected, readiness)
    checkpoint = Path(selected['checkpoint'])
    if not checkpoint.is_absolute(): checkpoint = ROOT / checkpoint
    data_path = ROOT / packet['validation_shard']
    for key, path in (('fit_result_sha256', result_path), ('selection_record_sha256', selection_path),
                      ('checkpoint_sha256', checkpoint), ('validation_shard_sha256', data_path)):
        digest = sha(path)
        if packet[key] != digest or readiness.get(key) != digest:
            raise ValueError('Benchmark selection provenance mismatch: ' + key)
    for name, digest in result['identity']['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Selected producer source mismatch: ' + name)
    reference = json.loads((ROOT / 'experiments/references/modded_nanogpt_20250126_metadata.json').read_text())
    if reference['validation_tokens'] != REFERENCE_TARGETS or reference['validation_sequence_length'] != 262144:
        raise ValueError('Published protocol metadata changed')
    # All selection checks above execute before a public shard is mapped/scored.
    import torch
    from types import SimpleNamespace
    import horizon_token_language_engine as lab
    torch.set_num_threads(1)
    args = SimpleNamespace(**result['args'])
    args.compiled = False
    torch.manual_seed(args.seed)
    train = lab.load_tokens(ROOT / args.train_file)
    if sha(ROOT / args.train_file) != result['identity']['train_sha256']:
        raise ValueError('Frequency-initialization training data changed')
    counts = lab.LaneTokens(train, 0, args.train_tokens, args.lanes).counts()
    model = lab.Model(args, torch.argsort(counts, descending=True, stable=True), counts)
    model.load_state_dict(torch.load(checkpoint, weights_only=True)); model.eval()
    data = lab.load_tokens(data_path)
    plan = validate_plan(target_windows(a.chunk_targets, len(data)))
    gen = torch.Generator().manual_seed(args.seed + 100000)
    windows_scored = 0
    started = time.monotonic()

    @torch.no_grad()
    def score_window(window, state):
        nonlocal windows_scored
        # load_tokens is a uint16 memmap; copies only bounded lane slices.
        value = native_window(model, data, args, gen, window, state)
        windows_scored += 1
        if windows_scored % 1024 == 0:
            print(json.dumps(dict(batch=window.batch, offset=window.offset,
                                  windows_scored=windows_scored, wall_s=time.monotonic()-started)), flush=True)
        return value

    measured = reduce_windows(target_windows(a.chunk_targets, len(data)), score_window)
    for name, digest in pins.items():
        if sha(ROOT / name) != digest: raise ValueError('Source changed during scoring: ' + name)
    baseline = reference['reported_final_validation_nll']
    wall = time.monotonic()-started
    record = dict(status='completed', **measured, reference_nll=baseline,
                  quality_difference_nll=measured['nll']-baseline,
                  verdict='pure-accuracy win (single seed)' if measured['nll'] < baseline else 'pure-accuracy loss (single seed)',
                  protocol_plan=plan, seed=args.seed, route_generator_seed=args.seed+100000,
                  deterministic_routes=args.deterministic_eval,
                  final_route_rng_sha256=hashlib.sha256(gen.get_state().numpy().tobytes()).hexdigest(),
                  source_sha256=pins, manifest_sha256=sha(manifest), checkpoint_sha256=sha(checkpoint),
                  validation_shard_sha256=sha(data_path), fit_result_sha256=sha(result_path),
                  selection_record_sha256=sha(selection_path), benchmark_selection_record_sha256=sha(readiness_path),
                  inference_wall_s=wall, targets_per_s=REFERENCE_TARGETS/wall,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Published FineWeb GPT-2 reference target/sequence/EOS protocol, CPU native selected weights. Native causal state may use the full declared sequence/document history; attention topology is not imposed. Historical published data hashes are absent; current shard is content-pinned. Quality comparison reuses published log; no matched-training-compute or measured energy claim. Inference wall includes native core, all-key scoring and exact likelihood; hashing/setup excluded.')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__': main()
