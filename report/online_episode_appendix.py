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
        ('small','No public language scoring interval or market TEST is read. Cached key-response algebra is valid within its fixed-learning-path assumptions; across changed optimizer updates the total effect needs versioned replay or audited transition prediction. Serial event-time CPU processing is not clockless hardware measurement.')]] + learned_credit_pages() + confirmation_pages() + stable_clock_pages() + feedback_credit_pages() + producer_credit_pages() + unseen_receiver_pages()

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

def producer_credit_pages():
    p=ROOT/'experiments/results/credit/curie_feedback_metric_v2_20261010T2355Z_decision.json'
    if not p.exists():return []
    d=json.loads(p.read_text());assert d['status']=='completed'
    return [[('h2','Producer-weighted feedback credit'),('p','The critic now trains against the squared parameter-gradient error induced by its cotangent, rather than coordinate error alone. Each audited packet supplies the exact producer Gram at its pre-update actor version; replay retains that metric and the credit-state creator version. Past producer-weighted audit moments calibrate trust.'),('table',(['Completed measurement','Value'],[['NLL/event',f'{d["nll"]:.6f}'],['Gain over untrained audits',f'{d["gain_over_untrained"]:+.6f}'],['Gain over BPTT',f'{d["gain_over_bptt"]:+.6f}'],['Gain over unweighted feedback',f'{d["gain_over_unweighted_feedback"]:+.6f}'],['Audited producer-error / zero',f'{d["producer_error_ratio"]:.4f}'],['Exact metric construction s',f'{d["metric_wall_s"]:.2f}']],[105,62])),('small',d['scope']),('p','The88x88 metric comes from an88x18164 producer Jacobian and is a paid diagnostic, not the final economical implementation. Its quadratic form matches the actual parameter VJP norm to3.6e-15. A stochastic projection needs its own variance/utility comparison. Route-conditioned credit beyond the active pool is specified separately: shared descriptors support predictions, and positive-support exploration supplies corrections without evaluating every dormant route.')]]


def unseen_receiver_pages():
    path=ROOT/'experiments/results/credit/curie_unseen_receiver_v2_20261010T2140Z_pilot.json'
    if not path.exists():return []
    d=json.loads(path.read_text());assert d['status']=='completed';rows=[]
    for a in d['metrics']['arms']:
        z=a['scopes']['unseen_inactive'];u=a['utility']
        rows.append([a['arm'],f"{z['relative_mse_vs_zero']:.4f}",f"{z['role_constant_relative_mse']:.4f}",f"{u['gain_from_expanding_counterfactual_pool']:+.6f}",str(u['unseen_inactive_selected'])])
    result_pages=[[('h2','Shared credit for never-labelled receiver addresses'),
        ('p','A route-conditioned connected critic learns only from receiver addresses0–5, then predicts optional native key-write effects at addresses6–7. Receiver descriptors contain observed state and token role, without raw receiver identity or future feedback. The comparison uses nearly matched structured-response and free heads.'),
        ('table',(['Critic','Error / zero','Role mean / zero','Expanded-pool gain','Inactive selected'],rows,[28,31,35,43,30])),
        ('small','Calibration concerns never-labelled receivers that differ from the factual active address. Lower relative error is better; positive pool gain means lower actual future loss than selection confined to training receivers. One fitted seed170,128 TRAIN and64 fresh DEV prefixes,8/16 synthetic continuations,256 steps. Prefix bootstrap intervals are conditional on this fitted seed.'),
        ('p','Native optional-write values, action derivatives, causal descriptors and receiver relabeling pass at depths2,4 and8. Value agreement is within1.6e-15. This interface preserves temporal/value evolution; the actor is frozen and factual write addresses remain observed marks.'),
        ('small',f"Entire pilot {d['wall_s']:.2f}s, peak RSS{d['peak_rss_kb']/1024:.1f}MiB, supported work{d['supported_flops']/1e9:.3f}GFLOPs including teacher extraction, all eight receiver queries, both critic fits and selected-write replay. Unsupported operations are excluded. This is a synthetic mechanism diagnostic; causal changing-weight integration and sparse discovery are subsequent tests.")]]

    confirmation=ROOT/'experiments/results/credit/curie_unseen_receiver_confirm_20261010T2125Z_decision.json'
    if confirmation.exists():
        c=json.loads(confirmation.read_text());assert c['status']=='completed'
        rs=[[str(r['seed']),r['arm'],f"{r['inactive']['relative_mse_vs_zero']:.4f}",f"{r['utility']['gain_from_expanding_counterfactual_pool']:+.6f}"] for r in c['rows']]
        result_pages.append([('h2','Three-seed receiver-generalization confirmation'),('table',(['Seed','Critic','Inactive error / zero','Expanded-pool gain'],rs,[22,33,55,57])),('p',f"All structured pool gains positive: {c['all_structured_pool_gains_positive']}. All structured inactive errors below TRAIN-role mean: {c['all_structured_inactive_better_than_role_mean']}."),('small',c['scope'])])
    return result_pages + sparse_receiver_pages() + shared_receiver_pages() + shared_receiver_confirmation_pages() + optimizer_aware_pages() + task_valued_pages() + combined_task_pages() + paired_audit_pages()


def sparse_receiver_pages():
    p=ROOT/'experiments/results/credit/curie_sparse_receiver_20261010T2240Z_measure.json'
    if not p.exists():return []
    d=json.loads(p.read_text());assert d['status']=='completed'
    rows=[[str(r['seed']),str(r['budget']),f"{r['gain_vs_train_fixed']:+.6f}",f"{r['policy_supported_flops']/1e6:.2f}",f"{r['policy_wall_s']:.2f}"] for r in d['rows'] if r['critic']=='structured']
    return [[('h2','Native selection with fewer queried receivers'),('p','Saved receiver critics now select native optional key writes using1,2,4 or8 receiver queries. Discovery anchors the TRAIN-best receiver and samples additional implicit indices without scoring unqueried descriptions. Every queried receiver evaluates nine messages; both critics receive an exact zero no-op option.'),('table',(['Seed','Receivers','Gain / target','Policy MFLOPs','Policy seconds'],rows,[22,30,40,41,34])),('small','Each row:32 fresh prefixes,8 independent synthetic continuations, frozen depth4 native actor. Positive gain means lower future loss than the TRAIN-fixed action. Policy work includes the native prefix recomputation, dense eight-mark forward scoring, queried descriptors and critic selection. MFLOPs are whole32-prefix work for every row; no per-target unit mixture. Wall includes FlopCounter instrumentation.'),('small',d['scope']),('p',f"Whole new diagnostic{d['whole_wall_s']:.2f}s, RSS{d['peak_rss_kb']/1024:.1f}MiB. Teacher work{d['teacher_supported_flops']/1e9:.3f}GFLOPs is additional. Earlier critic fits/checkpoints remain charged in their own completed evidence; no new fitting occurs here. Selection replay and per-seed prior-fit costs are recorded in the result file.")]]


def shared_receiver_pages():
    p=ROOT/'experiments/results/credit/curie_shared_receiver_fit_20261010T2300Z_pilot.json'
    if not p.exists():return []
    d=json.loads(p.read_text());assert d['status']=='completed'
    rows=[[r['arm'].replace('_',' '),str(r['parameters']),f"{r['unseen_inactive_relative_mse']:.4f}",f"{r['gain_vs_fixed']:+.6f}",f"{r['query_supported_flops']/1e6:.2f}"] for r in d['rows']]
    return [[('h2','Sharing forward context across receiver-credit queries'),('p','The new critic encodes the common forward prefix once, then combines it with a seven-dimensional receiver-specific recurrent representation. Versioned context activation can be reused across queried addresses. Sparse/batched values and accumulated parameter gradients match the repeated-context reference; stale versions are rejected.'),('table',(['Critic','Parameters','Inactive error / zero','Gain / target','Query MFLOPs'],rows,[50,25,35,32,25])),('small',d['scope']),('small',f"One seed170;128 TRAIN prefixes with8 continuations,64 DEV prefixes with16 continuations,256 steps. Whole64-prefix query work is in the same units for both arms; descriptors are precomputed for this query comparison. New fit{d['fit_wall_s']:.2f}s/{d['fit_supported_flops']/1e9:.3f}supportedGFLOPs; teacher{d['teacher_supported_flops']/1e9:.3f}GFLOPs; complete new phase{d['whole_wall_s']:.2f}s, RSS{d['peak_rss_kb']/1024:.1f}MiB. Fitting graphs, receiver state, optimizer and source packets are additional to the retained context output.")]]


def shared_receiver_confirmation_pages():
    p=ROOT/'experiments/results/credit/curie_shared_receiver_confirm_20261010T2320Z_decision.json'
    if not p.exists():return []
    d=json.loads(p.read_text());assert d['status']=='completed'
    rows=[[str(r['seed']),f"{r['gain_retained']*100:.1f}%",f"{r['query_work_ratio']:.3f}",f"{r['shared']['unseen_inactive_relative_mse']:.4f}"] for r in d['rows']]
    return [[('h2','Shared prefix: seeded development gate'),('table',(['Seed','Gain retained','Query work ratio','Inactive error / zero'],rows,[22,43,48,54])),('p',f"Gate passes: {d['confirmation_gate']}. "+d['gate_scope']),('small',d['scope'])]]


def optimizer_aware_pages():
    p=ROOT/'experiments/results/credit/curie_optimizer_aware_fit_20261011T0010Z_pilot.json'
    paired=ROOT/'experiments/results/credit/curie_paired_learning_effect_20261011T0030Z_fit.json'
    if not p.exists() or not paired.exists():return []
    d=json.loads(p.read_text());c=json.loads(paired.read_text());assert d['status']==c['status']=='completed'
    rows=[[r['arm'].replace('_',' '),f"{r['learning_aware_relative_mse']:.4f}",f"{r['gain_vs_train_fixed']:+.6f}"] for r in d['rows']]
    r=c['rows'][1];rows.append(['paired learning effect',f"{r['learning_aware_relative_mse']:.4f}",f"{r['gain_vs_train_fixed']:+.6f}"])
    return [[('h2','Route credit including subsequent readout learning'),('p','Optional native key writes now influence later predictions through four branch-local Adam updates to the mark readout. Each branch scores feedback before updating and owns its288 readout parameters/optimizer coordinates. The timed/keyed carrier stays fixed and every revealed prefix is replayed causally. Independent-world scores agree with batched replay within1.2e-15; zero-learning-rate response agrees with the frozen formula within2.6e-16.'),('table',(['Credit model','Error / zero','Gain / target'],rows,[90,35,42])),('p',f"Whole-response correction learns better calibration from learning-aware labels, but both corrections reduce task utility relative to the saved model. Paired supervision learns only the learning-mediated difference: effect error/zero{c['learning_effect_relative_mse']:.4f}. Its selected gain relative to the preserved baseline{c['gain_over_preserved_baseline']:+.6f}; keep the saved model for selection."),('small',d['scope']),('small',c['scope']),('small',f"Whole initial phase{d['whole_wall_s']:.2f}s, teacher{d['teacher_supported_flops']/1e9:.3f}supportedGFLOPs; both fits{sum(r['supported_flops'] for r in d['fit_work'])/1e9:.3f}GFLOPs. Paired refit{c['whole_wall_s']:.2f}s/{c['fit_supported_flops']/1e9:.3f}GFLOPs, reusing those paid packets; saved baseline pretraining remains additional. Largest peak RSS{max(d['peak_rss_kb'],c['peak_rss_kb'])/1024:.1f}MiB. TRAIN-only trust and fresh confirmation follow below.")]]


def task_valued_pages():
    p=ROOT/'experiments/results/credit/curie_credit_trust_20261011T0130Z_fresh.json'
    q=ROOT/'experiments/results/credit/curie_cached_effect_20261011T0145Z_contract.json'
    if not p.exists() or not q.exists(): return []
    d=json.loads(p.read_text()); c=json.loads(q.read_text()); assert d['status']==c['status']=='completed'
    m=d['metrics']; k=c['metrics']
    rows=[[r['arm'].replace('_',' '),str(r['alpha']),str(r['correction_prefixes']),f"{r['gain_vs_train_fixed']:+.6f}",f"{r['correction_supported_flops']/1e6:.3f}"] for r in m['rows']]
    work=[[r['arm'].replace('_',' '),f"{r['forward_supported_flops']/1e6:.3f}",f"{r['one_forward_backward_supported_flops']/1e6:.3f}"] for r in k['work']]
    return [[('h2','Task-valued credit trust: fresh locked evaluation'),('p','Fourfold TRAIN-only utility selection chose zero correction before generating fresh episodes. The trusted policy preserves the established baseline. Full correction changes one action and trails baseline by0.000324 nats/target; its bootstrap interval includes zero. No learned-correction utility win.'),('table',(['Policy','Trust','Evaluated prefixes','Gain / target','Extra MFLOPs'],rows,[57,20,32,30,28])),('small',d['scope']),('small',f"Complete fresh phase{d['whole_wall_s']:.2f}s, supported{d['supported_flops']/1e9:.3f}GFLOPs, peak RSS{d['peak_rss_kb']/1024:.1f}MiB. Extra correction work is whole32-prefix work; prior fitting/teacher work remains charged. Norm/SVD bound preparation is not counted by supported FLOP operators."),('h2','Exact shared-context correction execution'),('table',(['Execution','20 forwards MFLOPs','1 forward/backward MFLOPs'],work,[60,52,55])),('p','Context reuse cuts supported forward arithmetic89.3% and forward/backward arithmetic88.8% for the same learned head. Value and parameter gradients agree, and16 AdamW steps differ by less than5e-16, including nonzero-bias decay.'),('small','Both rows use the same16 prefixes,8 receivers and9 actions. Instrumented forward wall0.0513→0.0340s; forward/backward0.00691→0.00467s. These are paired kernel measurements, excluding teacher, baseline, discovery and prior fitting.'),('small',c['scope'])]]


def combined_task_pages():
    p=ROOT/'experiments/results/credit/curie_task_route_20261011T0340Z_fresh.json'
    q=ROOT/'experiments/results/credit/curie_native_policy_audit_20261011T0405Z_contract.json'
    selection=ROOT/'experiments/results/credit/curie_task_route_20261011T0340Z_select.json'
    if not all(x.exists() for x in [p,q,selection]): return []
    d=json.loads(p.read_text());a=json.loads(q.read_text());f=json.loads(selection.read_text())
    assert d['status']==a['status']==f['status']=='completed'
    rows=[[r['policy'].replace('_',' '),f"{r['gain_over_baseline']:+.6f}",f"[{r['bootstrap95'][0]:+.6f}, {r['bootstrap95'][1]:+.6f}]",f"{r['query_supported_flops']/1e6:.3f}"] for r in d['metrics']['rows']]
    return [[('h2','Combined route credit trained for future task value'),('p','Paid-packet diagnosis finds baseline static-response error larger than the learning-mediated effect. Even exact empirical effect addition can worsen route choices through error cancellation. The new combined correction uses expected future loss under a soft policy plus a frozen-prior KL penalty; matched MSE has the same2817 parameters, initialization, TRAIN data and256 updates. Fourfold TRAIN utility selection retains the baseline.'),('table',(['Fresh policy','Gain vs baseline','Prefix bootstrap95%','Extra query MFLOPs'],rows,[49,32,60,26])),('small',d['scope']),('small',f"Fresh32-prefix/8-continuation phase{d['whole_wall_s']:.2f}s, teacher{d['metrics']['teacher_supported_flops']/1e9:.3f}supportedGFLOPs; cross-fit/refit{f['whole_wall_s']:.2f}s/{sum(x['supported_flops'] for x in f['metrics']['fit_work'])/1e9:.3f}GFLOPs. Each query cell is whole32-prefix work; prior baseline fitting/teacher costs remain additional."),('p','Task-valued fitting has a positive fresh diagnostic point gain, while MSE loses utility. The task interval crosses zero and the primary locked policy is unchanged; no adopted or replicated utility win.'),('h2','Sparse utility audits expose a second learning objective'),('p',f"Issued policy-difference residual audits reconstruct native utility within{a['metrics']['native_exhaustive_mean_error']:.2g}. Only two actions require support on each of12 changed prefixes; unchanged policies require no comparison audit. Current learned residual prediction raises conditional action-sampling variance0.000810 versus zero-predictor0.000231. No efficiency promotion."),('small',a['scope']),('p','Theory derives a common-mode residual calibration that leaves route choices and unbiased gain unchanged but minimizes this audit variance. The next integrated learner must train both task contrasts and economical residual transport, issuing its prediction/calibration before future feedback.')]]


def paired_audit_pages():
    folder=ROOT/'experiments/results/credit'
    paired=folder/'curie_paired_audit_20261011T0740Z_measure.json'
    contract=folder/'curie_paired_audit_20261011T0740Z_contract.json'
    confirmation=folder/'curie_causal_offset_confirm_20261011T0650Z_decision.json'
    if not all(p.exists() for p in [paired,contract,confirmation]): return []
    d=json.loads(paired.read_text());c=json.loads(contract.read_text());f=json.loads(confirmation.read_text())
    assert d['status']==c['status']==f['status']=='completed'
    m=d['metrics'];p=m['paired'];w=c['metrics']['work']
    rows=[[str(r['stream_seed']),f"{r['causal_ratio']:.3f}",f"{r['fixed_ratio']:.3f}",f"{r['wall_s']:.2f}"] for r in f['rows']]
    comparison=[['single route + no-op',f"{p['mean_zero_single_variance']:.9f}",f"{w[1]['supported_flops']/1e6:.6f}",f"{w[1]['instrumented_wall_s']:.3f}"],['paired alternatives',f"{p['mean_variance']:.9f}",f"{w[0]['supported_flops']/1e6:.6f}",f"{w[0]['instrumented_wall_s']:.3f}"]]
    return [[('h2','Causal calibration: independent-history gate'),('table',(['History','Causal / zero variance','Fixed / zero variance','Phase seconds'],rows,[25,58,54,30])),('p',f"Confirmation gate passes: {f['confirmation_gate']}. History474 increases variance6.6%; retain zero prediction for single-route audits. Eight per-sample updates give a different learning clock from one update per issued prefix. New packet-normalized additive updates pass replica/order/version contracts; no performance gain is inferred."),('small',f['scope']),('h2','Paired route credit removes the common no-op branch'),('p','Compare future losses of the two alternatives directly. Their common no-op reference cancels, including its independent learning trajectory. Paired and single-route-plus-no-op audits both need two native worlds. For fixed comparisons under the same joint future, paired variance is no greater than the optimally centered uniform single-route estimator; future noise and nonstationarity are allowed.'),('table',(['Audit','Fresh variance','10 replays MFLOPs','10 replays seconds'],comparison,[52,37,41,37])),('p',f"Fresh paired variance is{(1-p['ratio_vs_zero_single'])*100:.1f}% lower at equal measured replay arithmetic. Native score/Adam/no-op cancellation matches within9e-16. This is an estimator-economy result; no model-quality or whole-learning win."),('small','Variance:32 fresh prefixes,8 continuations each,13 changed comparisons; one fixed native actor seed170 with four causal readout Adam updates per continuation. Work/wall columns:10 paired kernel repetitions,2 worlds each, same denominator for both rows. FLOPs count supported operators; scalar/special/unsupported optimizer operations remain outside that arithmetic count but inside wall.'),('small',f"Whole fresh diagnostic{d['whole_wall_s']:.2f}s, RSS{d['peak_rss_kb']/1024:.1f}MiB. Policy{m['policy_supported_flops']/1e6:.3f}MFLOPs; teacher{m['teacher_supported_flops']/1e6:.3f}MFLOPs/{m['diagnostic_worlds']}worlds; TRAIN preparation{m['train_calibration_preparation_supported_flops']/1e6:.3f}MFLOPs. Operational pair{p['operational_worlds']}worlds is a count: third diagnostic worlds and legacy calibration work were paid here. Earlier teacher/critic/actor fits remain additional."),('p','Next: paired forward/learner evidence trains the connected contrast critic; supported Bernoulli residual audits preserve the declared future-effect mean while prediction quality controls audit work. Test actual causal forward utility and complete fitting cost before a scaling claim.')]]
