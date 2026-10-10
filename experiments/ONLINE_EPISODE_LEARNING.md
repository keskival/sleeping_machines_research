# Online learning within real market and language episodes

User direction, 10 October 2026: both domains can benefit from causal within-episode learning under evolving distributions and inductive biases. Improve the forward model, credit model and optimizer together; do not freeze the native architecture as a research constraint. This programme belongs to B10/R1, uses CPU-only bounded queues, and protects sealed market TEST and public language scoring intervals.

## Objective and causal contract

Let H_t contain only observations revealed by event t. Forward numerical state s_t, credit state c_t, fast forward weights theta_t, credit parameters psi_t and optimizer state omega_t are distinct, versioned adaptive objects. Issue p_t(y_next|H_t) before the outcome. When feedback arrives, record loss against that issued forecast; update theta/psi/omega from revealed evidence, then consume the observed event to predict the subsequent one. A later model cannot rewrite an earlier score. Retain state across credit windows; detaching its graph truncates derivatives but must not erase its numerical value.

The conditional law and useful biases may change with history and event time. Do not assume all episodes necessarily yield adaptation gains: measure frozen-versus-adaptive forward performance, update costs, drift and recovery after changes. Topic variation in language can be captured by forward memory alone, so explicit weight/credit adaptation must beat a stateful frozen-weight control.

At optimizer boundaries: score the already-issued forecast, accumulate revealed losses, perform the admitted update, detach/reconcile carried graphs, then compute the next forecast with the new parameter version. Keep optimizer grouping separate from temporal/event windows. Equal timestamps use the recorded event order; tied market events retain the 1 microsecond recording-cell likelihood. Physical event-time processing does not itself prove clockless asynchronous hardware efficiency.

## Model development and comparisons

Models may gain richer keys/values, addressed memory, route pools, persistent credit activations, higher-capacity critics, faster state transport and token heads. Before substantive changes, state the observed failure, theoretical reason, retained/removed mechanisms and inference/learning costs; require numerical contracts and a bounded integrated fit.

For each developed model M, compare stateful frozen weights, causal online BPTT, predicted-plus-audited credit and a local-credit diagnostic. Match that row's warm start, observations/order, forecast targets, precision and update opportunities. Add model improvements as separate rows and compare complete systems at matched quality/work. A fixed-model row isolates learner contribution; the best system may change both model and learner. Reuse compatible completed controls; train no new external architectures.

Separate reset axes: document/day boundaries, numerical forward state, credit state, fast weights and optimizer moments. Frozen/fitted/reset credit controls identify learned credit and persistence benefits. Carrying pretrained weights is a common warm-start cost, not free training. Retained cached projections must bind their parameter version or be recomputed.

## Market episodes: B10

Use the existing Binance BTCUSDT data and chronological TRAIN/DEV/TEST partition, six mark categories, exact recording cell, and declared next-event forecast. A causal development bridge may use TRAIN-prefix-only initialization and later TRAIN events; no future episode statistics or validation-selected checkpoint may be presented as a temporally prior warm start. Existing Aug1-24 fits selected on Aug25-31 can support development diagnostics, but their selection cannot supply a strict prequential claim on those same validation days. Final sealed TEST starts after all permitted development fitting/selection.

Stream contiguous timestamped events across the old 1024-window boundaries. Preserve temporal modes, addressed slots, previous-message inputs and chosen memory state; add keyed ports when their failure diagnosis warrants it. Market episodes may span a session/day, with reset/carry policy declared in advance. Initially compare frozen versus online BPTT to identify genuine adaptation headroom; use development data to improve models before adding compact credit. Later vary update budget, elapsed-time horizon, depth and capacity independently.

Primary endpoints: prequential time/mark/total LL, complete work to fixed future-quality targets, quality at equal resources, wall time, memory and inference activity. Report causal adaptation after observed regime changes and per-day/group errors. Trading/P&L requires its own cost-inclusive protocol.

## Language episodes: R1

Retain exact GPT-2 tokens and existing FineWeb shard identity, chronological token order inside each document and normalized likelihood. Use fresh TRAIN suffixes after a declared warm-start token interval for development; do not touch reserved public validation or once-scored 65,528-target slices. EOS is observed feedback: close/reset the declared episode objects only when it arrives, before predicting the next document. A budget-start fragment is labelled truncated. Avoid selecting episodes by unseen future length/difficulty.

Contiguous document streams carry forward and credit state, with updates after revealed targets. Start with one lane or independent per-document adaptation copies: a shared fast-weight update across unrelated lanes can import feedback that is not in an episode's declared history. Grouped optimizer updates are causal if their predictions are scored before the group update and later tokens alone benefit. Measure token NLL, time-to-quality, post-change recovery, data exposure, key scoring, value deliveries, full normalized-head work and optimizer work.

The existing sparse token path already carries state, but its historical best checkpoint referenced in one gather result is unavailable locally. The first new gather contract stopped before scoring/fitting. Use available trained keyed-token checkpoints for a member-specific stream contract; do not regenerate a missing historical checkpoint or generalize that member into proof of internal hard routing. Main tokenized hard-route engineering remains an integration target.

## Counterfactual credit under online weight changes

The structured exact key-write response is valid when future hidden queries/timing hazards are unchanged by the intervention. Online weight updates can break this: the write changes loss, which changes a later optimizer update, which changes later hidden states. Within a fixed-parameter feedback interval the local response can remain exact; across updates distinguish direct fixed-learning-path response from total future utility under an intervened learning policy. The latter needs versioned learning-policy replay or a learned intervened transition model plus residual audits. Never reuse the cached decoder as if it certified that total effect.

Real streams do not provide independent true-generator continuations at the same prefix. Train from revealed trajectories and paid counterfactual replays; use correlated-window variance accounting and causal calibration rather than copying the synthetic 1/M noise claim. Learned continuation models need their own distribution/utility audits. Buffers retain forward packets, credit activations, actor/critic/state versions, addresses and horizon definitions until feedback. Stored activations alone do not preserve exact historical parameter gradients after updates; replay/sensitivities or explicit approximation must be charged.

## Admission sequence and evidence gate

1. Complete real-data carry/forecast/version contracts. B10 explicit state supports plain/keyed native temporal models; language validates a trained keyed-token member on fresh proper tokens.
2. Run one bounded causal frozen-versus-online baseline bridge per domain, with all updates/initialization charged. Diagnose absence or presence of adaptation headroom and improve the model accordingly.
3. Integrate learned credit at actual producer/state boundaries, retaining deep factual paths and counterfactual audits. Pass changing-weight/version contracts before a fit. Start small on real TRAIN data, then expand contiguous data/horizon.
4. Compare optimized native BPTT against our complete learner at depths 2/4/8, varying capacity/horizon separately. Include credit pretraining, unsuccessful work, optimizer/discovery and serving costs. Win requires equal or better future quality at lower total work; better scaling requires that work ratio improve with size/depth and three-seed uncertainty.
5. Freeze the developed protocol before once-only final market TEST/public language comparisons. CPU latency or graph correctness is not a clockless hardware energy claim.

## First completed real-data bridge

The 1,024-target exploratory bridge completes in each domain. Market stateful frozen NLL 12.97808 versus online native BPTT 12.59141 (32 updates); arm wall 1.36 versus 4.18 s. Language stateful frozen NLL 5.28992 versus online native BPTT 5.26495 (33 updates, five observed EOS resets); wall 5.04 versus 35.30 s. Whole setup/bridge 46.81 s, peak RSS 591.9 MiB. Market initialization uses prior-context-only gap statistics and no trained weights; language uses the saved 4M-token keyed member. This proves adaptation headroom in the measured fragments, not a nonstationarity causal attribution, a learned-credit efficiency result or a larger-data benchmark win. Next controls need stronger causal market warm starts and longer diverse episodes, followed by complete-credit/model comparisons.

Market state/score contracts pass plain and keyed models at depths 2/4/8; actual saved-market score error is 1.9e-10. Trained-token lazy key aggregation passes score/all-parameter gradient parity at 3.7e-15 in float64, numeric carry and suffix causality. Its first float32 check stopped at gradient reduction error 3.1e-5; double precision isolates the algebra. Production float32 requires a declared numerical tolerance and its own measured fit, not silent tolerance relaxation.

## First online learned boundary integration

`credit/real_learned_boundary.py` learns the next feedback-window cotangent of persistent temporal states and addressed value slots. A connected MLP sees only the produced state before future feedback. Local factual gradients remain exact. A Bernoulli audit drawn before feedback supplies exact future boundary credit; the estimator is predicted credit plus audited residual divided by inclusion probability. Only audited revealed targets train the critic. Forward and credit weights both change after the window; all stored producer graphs are consumed under their original actor version before the update. No stale historical VJP is reused.

Compare full-window BPTT, split local credit, untrained zero predictor with audits, and learned predictor with identical audit draws. The all-audit split must reproduce full BPTT gradients. This is an initial learned future-sensitivity teacher, not future-utility meta-learning or full replacement of BPTT. Conditional cotangent unbiasedness precedes clipping and Adam; nonlinear updates are not claimed unbiased. The current diagnostic still builds local factual graphs and predicts dense state credit, so sparse audit frequency alone is no efficiency result. Whole wall time includes actor, critic, audit and optimizer work. Start with a 64-target smoke and 2048-target real TRAIN comparison; use error analysis before expanding depth/size or claiming a win.

### First real learned-credit result and repair

V1, 2048 real TRAIN targets, seed181: full BPTT NLL13.912655 (11.03s); local13.917740 (8.06s); zero predictor with 17/64 residual audits13.903259 (8.82s); online learned predictor with the same audit draws13.936618 (8.98s). The learned predictor harms this variant. Its audited mean-squared error grows from0.00172 (first8 audits) to0.03204 (last8), versus zero0.000493 to0.02409, on each arm's evolving actor path. These different paths prevent interpreting the error difference as a matched-label calibration test.

V2 addresses state-scale sensitivity with LayerNorm on predictor inputs and learns a conservative credit trust coefficient from prior audits: alpha=clip(EMA(pred dot target)/EMA(pred squared),0,1), forgetting0.95, zero prior. This minimizes a past-audit squared-error surrogate over a scalar blend, not a guarantee of future calibration or utility under drift. Credit output is alpha*prediction; full-support residual audits preserve the pre-optimizer expectation for every causal alpha. Actor and credit parameters still update within episodes. Compare raw and calibrated predictors within V2 to isolate trust; zero/audit/BPTT/local controls retain the common normalized predictor setup. Five-arm bounded8192-target TRAIN pilot, then replicated seeds only after diagnosis. Retain final actor/critic/optimizer/state checkpoints. Predictor storage, dense local derivatives, audit VJPs and all optimizer work remain charged.
