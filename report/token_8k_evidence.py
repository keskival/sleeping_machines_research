"""Completed paired 8K integrated-stage results and diagnostic-driven comparisons."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def page():
    folder=ROOT/'experiments/results/token_language'
    diagnostics=ROOT/'experiments/results/diagnostics'
    stage=json.loads((diagnostics/'curie_fixed_batch_tokens_8k_stage_20261005_v1.json').read_text())
    utility=json.loads((diagnostics/'curie_fixed_batch_tokens_8k_utility_20261005_v1.json').read_text())
    if stage['status']!='completed' or utility['status']!='completed':raise ValueError('Incomplete8Kstage')
    rows=[]
    for r in stage['rows']:
        u=next(x for x in utility['rows'] if x['tag']==r['tag'])
        rows.append([str(r['seed']),str(r['credit_window']),f"{r['initial_dev_nll']:.6f}",
                     f"{r['selected_dev_nll']:.6f}",f"{r['gain']:.6f}",
                     f"{u['context_gain']:.6f}",f"{u['history_gains']['memory']:.6f}"])
    comparisons=[]
    for seed in (6,7):
        control=next(r for r in stage['rows'] if r['seed']==seed and r['credit_window']==16)
        for label,tag in [('Full-width tails',f'curie_fullwidth_tokens_8k_b64_c16_s{seed}_20261005_v1'),
                          ('Memory gain2',f'curie_coupled_tokens_8k_b64_c16_gain2_s{seed}_20261005_v1')]:
            path=folder/(tag+'.json')
            if not path.exists():continue
            d=json.loads(path.read_text())
            if d['status']!='completed':continue
            best=min(r['dev_nll'] for r in d['curve'])
            comparisons.append([label,str(seed),f'{best:.6f}',
                                f"{control['selected_dev_nll']-best:.6f}",str(d['parameters'])])
    work=[]
    for credit in (16,64):
        tag=f'curie_fixed_batch_tokens_8k_b64_c{credit}_s6_20261005_v1'
        path=diagnostics/f'curie_fixed_batch_tokens_8k_c{credit}_work_20261005_v1.json'
        if not path.exists():
            work.append([f'Credit{credit}seed6','16368','pending','pending','pending']);continue
        d=json.loads(path.read_text())
        if d['status']!='completed' or not d['fitting']['formula_coverage_complete'] or not d['inference']['formula_coverage_complete']:
            raise ValueError('Incomplete actual8Kwork')
        work.append([f'Credit{credit}seed6',str(d['fitting_targets']),
                     f"{d['fitting']['arithmetic_flops']/1e9:.6f}",
                     f"{d['fitting_arithmetic_flops_per_target']/1e6:.6f}",
                     f"{d['inference_arithmetic_flops_per_target']/1e6:.6f}"])
    return [('h1','Appendix. Tokenized8K: replicated learning and memory-use decisions'),
        ('p','Sleeping Machines pursues a general-purpose substrate for language and reasoning, multimodal world models, embodiment, event-native analytics, continual learning, communication, self-design and hardware. This integrated stage advances the Transformer-leading language program; tiny fits are engineering diagnostics.'),
        ('p','Both horizons beat initialization in both seeds. GPT-2 FineWeb:8192admitted training tokens,16368fitting target presentations/two passes,32AdamWupdates,2040development targets. P16/D2/H2/U4,batch64; credit16/64 share data/updates/cadence. Initial-inclusive development selection every8updates; public validation untouched. Four selected writes,16scored keys and256persistent memory scalars per lane. The bounded local fits preserve the immutable AWSqueues.'),
        ('table',(['Seed','Credit','Initial NLL','Selected NLL','Gain','Context gain','Memory gain'],rows,[35,40,80,80,75,85,90])),
        ('p','Credit16mean selected8.291959 beats credit64mean8.295183, and is better in both paired seeds; the0.003224mean gap is smaller than seed spread. Ordinary measured fitting throughput is395–422targets/s, peakRSS452456–453252KiB. Retain both completed records. Numeric state carries across credit boundaries; truncating derivatives does not reset state.'),
        ('p','Context gain uses the same frozen readout with a constant mean of8184causal TRAINfeatures. Memory gain is NLL after per-token addressed-state/arrival/seen erasure minus intact NLL; negative means the intervention improves loss. Recurrent-message erasure costs0.017843–0.021955NLL. Matched route RNG, source binding and token partition parity pass. These are frozen interventions, not retrained ablations; the constant-feature control is not an optimally refitted unigram.'),
        ('table',(['Alternative','Seed','Selected NLL','Gain vs credit16','Parameters'],comparisons,[135,40,100,120,90])),
        ('p','Full-width decoder tails lose at seed6 and are effectively tied at seed7. Retain the narrower decoder. First-tail training exposure is695targets/data pass, so this comparison exercises learned tail projections. The memory-gain intervention doubles the existing unit output contribution while preserving all temporal/state/routing/learning mechanisms and parameter count; it is selected only from completed fits. See TOKEN_MEMORY_COUPLING_20261005.md for derivation and contracts.'),
        ('table',(['Native member','Fit targets','Whole fit GFLOPs','Fit MFLOPs/target','Eval MFLOPs/target'],work,[110,65,110,115,105])),
        ('p','Actual whole-fit traces must reproduce each control trajectory. Arithmetic includes factual computation, all-key discovery, suffix replay, exact target readout, backward, clipping, optimizer and in-step diagnostics; initialization/frequency counting,evaluation and serialization excluded. Special functions separate; random-sampling work unquantified. Evaluation includes scorer reductions. Pending cells contain no predicted work or quality. Equal updates/data here are not an iso-FLOP claim. Selected Transformer-reference quality and resource protocol remains the larger benchmark target.')]
