"""Source-bound native language strengths and completed development decisions."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def page():
    folder = ROOT / 'experiments/results/token_language'
    rows = []
    for label, tag in [('Retained lr .003', 'curie_horizon_2k_batch64_credit64_20261005_v1'),
                       ('Alternative lr .001', 'curie_horizon_2k_b64_c64_lr001_s6_20261005_v1')]:
        d = json.loads((folder / (tag + '.json')).read_text())
        if d['status'] != 'completed': raise ValueError(tag)
        rows.append([label, f"{min(r['dev_nll'] for r in d['curve']):.6f}",
                     f"{d['curve'][-1]['dev_nll']:.6f}", str(d['presentations_total']),
                     f"{d['train_tokens_per_second']:.1f}", str(d['max_rss_kb'])])
    audit = ROOT / 'experiments/results/diagnostics'
    exposure = json.loads((audit / 'curie_token_tail_exposure_20261005_v1.json').read_text())
    if exposure['status'] != 'completed': raise ValueError('exposure')
    tail_rows = [[str(r['train_tokens']), str(r['distinct_tokens']),
                  *[str(v) for v in r['branch_targets']]] for r in exposure['rows']]
    work_rows = []
    for label, name, field in [
        ('Whole fit', 'curie_horizon_2k_b64_c64_work_20261005_v1.work.json', 'fitting'),
        ('Selected evaluation', 'curie_horizon_2k_c64_inference_work_20261005_v1.json', 'inference')]:
        path = audit / name
        if not path.exists():
            work_rows.append([label, 'pending', 'pending', 'pending', 'pending'])
            continue
        d = json.loads(path.read_text())
        if d['status'] != 'completed': raise ValueError(name)
        ledger = d[field]
        work_rows.append([label, str(d['targets']), f"{ledger['arithmetic_flops']/1e9:.6f}",
                         f"{d['arithmetic_flops_per_target']/1e6:.6f}",
                         'complete' if ledger['formula_coverage_complete'] else 'partial'])
    return [('h1', 'Appendix. Language strengths: measured decisions and complete work'),
        ('p', 'The target data region deliberately favors strong Transformer references over LSTMs; tiny native fits diagnose learning before that larger comparison. The native construction chooses temporal computation, persistent memory lifetime and sparse addressed activity independently of Transformer topology or fixed windows. Matched-history and longer-native-history comparisons are labelled separately; causal inputs and scored targets stay explicit. Capacity beyond activity earns promotion through better held-out prediction with discovery and learning fully charged.'),
        ('p', 'Controlled learning-rate test: seed6, GPT-2 FineWeb2K, P16/D2/H2/U4, batch64/credit64, uniform K4 actual alternative-write utility,16updates and1,016development targets. Initial weights eligible. Lower learning rate loses selected quality by0.046604NLL; retain .003. Its better final loss does not replace the better selected incumbent. These are development comparisons.'),
        ('table', (['Recipe','Selected NLL','Final NLL','Fit targets','Targets/s','RSS KiB'],rows,[125,75,75,75,70,70])),
        ('p', 'Decoder exposure uses train-only frequency rank and eight contiguous lanes. At2K every target reaches the head: identical default/full-width development curves do not test learned tail rank. The existing8Kcomparison reaches the first tail. Branch counts below are targets per data pass; head cutoff2,000, tail boundaries10,000/30,000/50,257.'),
        ('table', (['Train tokens','Distinct','Head','Tail1','Tail2','Tail3'],tail_rows,[95,75,85,85,75,75])),
        ('p', 'Actual native work: multiply-add=2FLOPs; arithmetic excludes separately counted special functions and unquantified random sampling. Whole fitting includes factual core, all-key scoring, alternative discovery, suffix replay, exact readout, backward, clipping, optimizer and in-step diagnostics. Initialization/frequency counts, evaluation and serialization are outside that fitting column. Selected evaluation includes scorer reductions, with no gradient/optimizer. Audit instrumentation wall time is not ordinary throughput. Public Transformer work remains in its separate audited protocol; no matched-compute win is inferred here.'),
        ('table', (['Boundary','Targets','Total GFLOPs','MFLOPs/target','Coverage'],work_rows,[140,70,110,105,65])),
        ('p', 'Next: existing AWS8Kpaired seeds/horizons, trained memory utility, informative decoder comparison, then measured width/data and fixed-activity pool scaling. Public validation remains reserved. See experiments/TOKEN_STRENGTH_EXECUTION_20261005.md for the ordered strength tests.')]
