"""Completed causal real-data episode bridge; no benchmark win claim."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TITLE='Appendix. Real market and token episodes: causal online adaptation'

def pages():
    path=ROOT/'experiments/results/credit/curie_real_online_bridge_v1_20261010T2050Z_measure.json'
    if not path.exists():return []
    r=json.loads(path.read_text());assert r['status']=='completed';rows=[]
    for domain,arms in r['metrics']['domains'].items():
        for a in arms:rows.append([domain,a['arm'].replace('stateful_','').replace('online_native_','online '),f'{a["nll"]:.5f}',str(a['updates']),f'{a["wall_s"]:.2f}',f'{1000*a["wall_s"]/a["targets"]:.3f}'])
    return [[('h1',TITLE),
        ('p','Both real streams admit useful within-episode adaptation in this short bridge. Issued forecasts are scored before feedback updates; forward numerical state survives credit boundaries. Stateful frozen weights are the control, and the current learning arm is native online BPTT.'),
        ('table',(['Domain','Arm','NLL/target','Updates','Wall s','ms/target'],rows,[22,42,30,21,23,29])),
        ('small','One run, 1,024 fresh TRAIN targets per domain, 128-context observations, update groups of 32, CPU float64 and dropout disabled for both arms. Market model is initialized using only prior context gap statistics; language uses an available 4M-token pretrained keyed member. Language fast weights/optimizer/state reset after five observed EOS boundaries. Full vocabulary output and dense optimizer work are charged. These different warm starts are not compared across domains.'),
        ('p','Online adaptation lowers market NLL by 0.38668 and language NLL by 0.02497 nats/target. Its increased wall cost motivates an economical learned-credit comparison and model development. Cold market parameter fitting is one possible source of the gain; this bridge does not identify nonstationarity as the cause.'),
        ('small',f'Whole bridge including setup/copies and both domains {r["metrics"]["whole_wall_s"]:.2f} s; peak RSS {r["peak_rss_kb"]/1024:.1f} MiB. Per-arm wall includes context replay/scoring/backward/updates, but excludes model copy and initial optimizer construction. Language warm-start fitting cost {r["metrics"]["warm_start"]["language"]["prior_fit_wall_s"]:.1f} s is separate and must be included in a complete-system comparison.'),
        ('p','Real-market state and fixed-parameter gradient contracts pass at depths 2/4/8, with saved-market score parity error 1.9e-10. A trained proper-token member preserves lazy addressed key transport, scores and every factual parameter gradient in float64 to 3.7e-15. Detachment preserves numbers while truncating credit; issued forecasts cannot be rewritten by later updates.'),
        ('small','No public language scoring interval or market TEST is read. Cached key-response algebra is valid within its fixed-learning-path assumptions; across changed optimizer updates the total effect needs versioned replay or audited transition prediction. Serial event-time CPU processing is not clockless hardware measurement.')]] + learned_credit_pages() + confirmation_pages() + stable_clock_pages() + feedback_credit_pages()

def learned_credit_pages():
    tags=['curie_real_boundary_v3_20261010T2200Z_measure','curie_real_boundary_v2_20261010T2150Z_measure','curie_real_boundary_v1_20261010T2140Z_measure']
    path=next((ROOT/'experiments/results/credit'/f'{t}.json' for t in tags if (ROOT/'experiments/results/credit'/f'{t}.json').exists()),None)
    if path is None:return []
    result=json.loads(path.read_text());assert result['status']=='completed';arms=result['metrics']['rows'];rows=[]
    if 'v3_' in path.name:
        control=json.loads((ROOT/'experiments/results/credit/curie_real_boundary_v2_20261010T2150Z_measure.json').read_text())
        arms=[a for a in control['metrics']['rows'] if a['arm'] in ('bptt','local','untrained_audit')]+arms
    for a in arms:
        rows.append([a['arm'].replace('_',' '),f'{a["nll"]:.6f}',str(a['audits']),str(a['critic_steps']),f'{a["wall_s"]:.2f}'])
    learned=next(a for a in arms if a['arm']=='learned_audit');zero=next(a for a in arms if a['arm']=='untrained_audit')
    direction='improves' if learned['nll']<zero['nll'] else 'harms'
    return [[('h2','Real market: online learning of state credit'),
        ('p','Actor and credit parameters now learn in the same causal real-data episode. A connected predictor sees produced temporal/value state before future feedback, estimates the next window cotangent, and learns only from selected revealed audits. Local factual gradients remain exact; residual correction preserves the boundary expectation before clipping and Adam.'),
        ('table',(['Arm','NLL/event','Audits','Critic steps','Wall s'],rows,[55,33,23,31,25])),
        ('small',f'One development TRAIN fragment, seed {result["args"]["seed"]}, {learned["targets"]} targets, groups {result["args"]["group"]}, audit probability {result["args"]["audit_probability"]}. Lower NLL is better. Raw learned predictor {direction} quality against the untrained predictor under identical audit draws; evolving actor paths differ. No replicated benchmark claim or TEST score.'),
        ('p','V1 learned credit hurt quality on 2048 targets. V2 normalizes state features and adds a causal trust coefficient fitted from prior audit cross-moments; raw versus calibrated rows isolate the trust contribution. V3 adds recency-weighted replay of already-revealed audit records; unchanged BPTT/local/untrained controls reuse V2 evidence. All-audit split gradients match full BPTT to 3.3e-16. Producer graphs are consumed under the same actor version before weights change.'),
        ('small',f'Selected phase setup and its executed arms: {result["whole_wall_s"]:.2f} s; peak RSS {result["peak_rss_kb"]/1024:.1f} MiB. Predictor, local graph, audit VJPs and both optimizers are charged; this prototype does not replace the complete backward pass. Baseline arms also execute an unused predictor to retain the common scaffold, so these wall times do not establish superiority over optimized BPTT. No asynchronous hardware or energy claim.')]]

def confirmation_pages():
    path=ROOT/'experiments/results/credit/curie_real_boundary_confirm_20261010T2210Z_decision.json'
    if not path.exists():return []
    d=json.loads(path.read_text());assert d['status']=='completed'
    rows=[[str(g['seed']),f'{g["over_untrained"]:+.6f}' if g['seed']!=183 else 'unstable',f'{g["over_bptt"]:+.6f}' if g['seed']!=183 else 'unstable'] for g in d['gains']]
    return [[('h2','Real online-credit replication'),('table',(['Seed','Gain over untrained audits','Gain over BPTT'],rows,[25,70,70])),('p','Positive gains mean lower future prediction NLL from calibrated online learned credit. All-seed point gate: '+str(d['positive_all_seeds'])+'. Seed182 favors untrained audits; seed183 suffers catastrophic loss in every arm, so its raw difference is not a useful learning gain.'),('small',d['scope']),('p','Forward and credit weights adapt during the episode. Paid audits train a connected state-credit predictor; causal trust and recency-weighted replay support learning. Local derivatives remain exact and the current implementation retains dense local work. Observer-only seed183 replay reproduces the original scores and locates clock log-rate44–45 at an ordinary6.022ms gap. The next integrated variant compresses positive-slot inputs at the clock readout, retaining state, races and scoring; contracts and a matched three-arm pilot precede wider/deeper tests.')]]

def stable_clock_pages():
    p=ROOT/'experiments/results/credit/curie_real_boundary_stable_v1_20261010T2310Z_measure.json'
    if not p.exists():return []
    d=json.loads(p.read_text());assert d['status']=='completed';arms=d['metrics']['rows'];rows=[[r['arm'].replace('_',' '),f'{r["nll"]:.6f}',str(r['audits']),str(r['critic_steps']),f'{r["wall_s"]:.2f}'] for r in arms]
    by={r['arm']:r for r in arms};gain=by['untrained_audit']['nll']-by['calibrated_audit']['nll']
    return [[('h2','Integrated clock-read repair: positive memory compression'),('p','The original seed183 episode becomes catastrophic in all learners at ordinary gaps. Affine clock outputs read unbounded positive addressed values. The new variant applies log1p at the clock head only: stored values, decay/write law, temporal layers, races, mark readout and recording-cell scoring remain. This is a changed native model shared by all three learner arms.'),('table',(['Arm','NLL/event','Audits','Critic steps','Wall s'],rows,[55,33,23,31,25])),('p',f'Calibrated learned credit gain over matched untrained audits: {gain:+.6f} nats/event. These are completed seed183 development results on8192 TRAIN targets; lower NLL is better. Original pathological result files remain preserved.'),('small',f'Full-stream score parity2.9e-10 and parameter-gradient parity3.1e-11; split all-audit BPTT parity2.2e-16. Complete phase {d["whole_wall_s"]:.2f}s; peak RSS{d["peak_rss_kb"]/1024:.1f}MiB. All local derivatives, audits, replay and optimizer work charged. Common unused predictor scaffolding remains in controls; no optimized efficiency, depth scaling or sealed TEST claim.')]]

def feedback_credit_pages():
    p=ROOT/'experiments/results/credit/curie_feedback_credit_v1_20261010T2330Z_measure.json'
    if not p.exists():return []
    d=json.loads(p.read_text());assert d['status']=='completed';base=json.loads((ROOT/'experiments/results/credit/curie_real_boundary_stable_v1_20261010T2310Z_measure.json').read_text());a=d['metrics']['rows'][0];arms=base['metrics']['rows']+[a];rows=[[r['arm'].replace('_',' '),f'{r["nll"]:.6f}',str(r['audits']),str(r['critic_steps']),f'{r["wall_s"]:.2f}'] for r in arms]
    zero=next(r for r in arms if r['arm']=='untrained_audit');gain=zero['nll']-a['nll'];mse=sum(a['audited_mse'])/len(a['audited_mse']);zmse=sum(a['audited_zero_mse'])/len(a['audited_zero_mse'])
    return [[('h2','Persistent credit learns from revealed forward errors'),('p','A connected GRU critic consumes produced temporal/value states, revealed gaps/marks and issued prediction errors. Its64-dimensional credit state evolves with learned elapsed-time decay and persists across update windows. It predicts the88-dimensional cotangent after feedback but before the independent exact audit. No audit target enters the current prediction.'),('table',(['Arm','NLL/event','Audits','Critic steps','Wall s'],rows,[55,33,23,31,25])),('p',f'Feedback-conditioned credit gain over untrained audits: {gain:+.6f} nats/event. On its own audited packets, calibrated prediction MSE{mse:.6g} versus zero{zmse:.6g}; the matched-label error comparison uses the same actor path, while utility comparisons follow different evolving actors.'),('small',f'One seed183,8192 real TRAIN targets, stable log-compressed clock model, update groups32, audit probability0.25. Saved controls are reused unchanged. Credit predictor{a["critic_parameters"]:,}parameters, credit state{a["credit_state_elements"]}values; replay/retained records and all actor/credit optimizers charged. Entire new-arm phase{d["whole_wall_s"]:.2f}s, RSS{d["peak_rss_kb"]/1024:.1f}MiB. Dense local derivatives and audited producer credit remain; no full backward replacement, hard-route alternative credit, optimized efficiency or TEST claim.')]]
