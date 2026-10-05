"""Completed tokenized development evidence; no pending scores enter tables."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def pages():
    folder = ROOT / 'experiments/results/token_language'
    entries = [('Local value', 'curie_integrated_2k_local_20261005_v1'),
        ('First-site full suffix', 'curie_integrated_2k_full_20261005_v1'),
        ('First-site K4 suffix', 'curie_integrated_2k_k4_20261005_v1'),
        ('Uniform-site K4 suffix', 'curie_event_credit_2k_k4_20261005_v1')]
    rows = []
    for label, tag in entries:
        result = json.loads((folder / (tag + '.json')).read_text())
        if result['status'] != 'completed': raise ValueError(tag)
        rows.append([label, f"{result['curve'][0]['dev_nll']:.6f}",
            f"{result['best_dev_nll']:.6f}", f"{result['curve'][-1]['dev_nll']:.6f}",
            f"{result['train_tokens_per_second']:.2f}"])
    audit_folder = ROOT / 'experiments/results/diagnostics'
    work = []
    for label, name in [('Full scoring', 'curie_integrated_token_work_audit_20261005_v3'),
                       ('K4 scoring', 'curie_sampled_token_work_audit_20261005_v1')]:
        result = json.loads((audit_folder / (name + '.json')).read_text())
        if result['status'] != 'completed' or not result['complete']: raise ValueError(name)
        arithmetic = sum(v['arithmetic_flops'] for v in result['stages'].values())
        work.append([label, str(result['targets']), f'{arithmetic/1e6:.6f}',
                     f"{arithmetic/result['targets']/1e6:.6f}"])
    return [[('h1', 'Appendix. Tokenized language: integrated learning development'),
        ('p', 'Sleeping Machines pursues a general-purpose substrate for language and reasoning, multimodal world models, embodiment, event-native analytics, continual learning, communication, self-design and hardware. These language diagnostics test constructions within that family.'),
        ('p', 'Seed6; GPT-2 FineWeb; 2,048 training tokens, 8,160 presented targets over64updates, 1,024 disjoint development targets. P16/D2/H2/U4:256persistent memory scalars per lane, four selected writes and16scored keys per target. Public validation remains reserved. Loss is natural-log NLL; lower is better. Throughput includes training work, excludes evaluation.'),
        ('table', (['Member','Initial dev','Best trained dev','Final dev','Targets/s'], rows, [150,80,90,80,75])),
        ('p', 'Uniform-site K4 wins the trained development comparison against first-site K4 by0.067745NLL, single seed at equal data/presentations. It combines broader site exposure, utility weighting and removal of immediate-only route credit.'),
        ('p', 'All trained members lose to initialization. Uniform-site late dev10.354817 loses to first-site K4late9.844873. Initial weights now remain eligible for selection; historical trained-only fields stay preserved with the correction beside them.'),
        ('p', 'Mechanisms: persistent state, race clocks/transport, sparse hard writes, separate selection/value roles, small messages, depth and actual counterfactual writes. Credit remains chunk-bounded; all keys are scored. The uniform-site learner uses winner-only factual/replay proposals. Memory occupancy alone is not predictive capacity.')],
        [('h1', 'Appendix. Tokenized complete work and scaling prerequisites'),
        ('p', 'Synthetic P16/D2/H2/U4/B8/T16, vocabulary50,257; all adaptive tails exercised, first AdamW update. Arithmetic includes factual computation, readout, alternative sampling, replay, backward, clipping and optimizer. Special functions have a separate ledger; random sampling work is unquantified. These are executed steps, not whole-fit extrapolations or hardware energy.'),
        ('table', (['Scoring','Targets','Step MFLOPs','MFLOPs/target'],work,[160,70,125,125])),
        ('p', 'K4 arithmetic is14.05%lower in this realization. Replay readout falls47.286432to12.713512MFLOPs. Complete fitting and per-target fitting FLOPs on the actual corpus, and winner-only inference FLOPs, are not yet traced; no whole-fit resource win is claimed.'),
        ('p', 'Exact actual-driver interruption/resume passed for integrated, sampled-position and uniform-site learners, including all relevant RNGs. Full-score sampled learning equals the original trajectory. Wider decoder tails preserve normalization and initial token priors; default parameterized learning parity and full-width resume passed.'),
        ('p', 'The public reference target reuses the modded-nanoGPT2025-01-26log:3.2774NLL,695,992,320training presentations,10,485,760reserved validation targets. That protocol/context differs from these development rows. The exact attention/context audit precedes benchmark scoring; no dense reference is retrained.'),
        ('p', 'Prepared AWS8K comparisons test broader credit and full-width tails before selecting a scaling member. They are unrun here and have no predicted scores. Rough scaling laws follow a selected promising member and actual resource measurements. Delivery/admission remains unobserved.')]]
