# R1: independent lane exploration without changing the one-lane model

15 October 2026. Candidate backend: sleeping_machines/lane_independent_races.py.
This repairs an implementation coupling named in LANGUAGE_IMPLEMENTATION_AUDIT_20261005.md;
no quality gain, trained-model promotion or new fit is claimed.

For issued lane states and parameters, each receiver clock has independent
Exp(1) noise divided by exp(score). Its winner law is softmax(score) and minimum
time law has rate sum exp(score). The pinned backend broadcasts one H×U tape
over N lanes; the new backend uses N×H×U independent factual and proposal tapes.
Each lane's conditional law and the existing declared factorized race-time
surrogate remain unchanged. The loser proposal is still pi_j/(1-pi_w), excluding
the factual winner. Parameter sharing is retained. Thus expected raw per-lane
credit for the declared objective is unchanged; cross-lane covariance changes.
This does not make the surrogate an exact fixed-noise derivative of a hard min.

For batch mean raw gradients g_bar=(1/N)sum_i g_i,

  Cov(g_bar)=N^-2 [sum_i Cov(g_i)+sum_(i!=j) Cov(g_i,g_j)].

Conditional independent lane tapes remove the noise cross terms for separable
lane computations at an issued cut. If identical lanes have variance Sigma,
shared identical noise gives Sigma whereas independence gives Sigma/N. For
heterogeneous lanes common-noise covariance can be negative; independence is
not a universal variance reduction. Data/state variation and the shared
one-site future teacher still contribute variance. Clipping/Adam depend on
batch covariance, so identical raw marginal laws do not imply identical actual
updates or fitted quality. Use common random numbers ACROSS compared worlds
within a lane; independence between training lanes does not justify independent
noise between factual and counterfactual histories.

The module retains sparse writes, every key score, keys/values, clocks, memory,
transport, forced/suppressed route support and local sampled alternative credit.
Both last-stage and all-depth output interfaces are supported. independent_forward
returns an isolated forward function for declared native episode helpers without
patching frozen globals. Unsupported protected parent/private-branch streams
are rejected; they require ownership-aware adaptation and a new contract.
Never switch the policy of an existing source-bound run/checkpoint silently.

Eight numerical contracts pass: exact single-lane outputs/state/all gradients
and both tapes; deterministic multi-lane parity; stochastic chunk/checkpoint
replay; diversity of identical lanes' factual routes; supplied-content gradients;
shape/eager guards; exact all-depth single-lane output/gradient parity; actual
50,257-token native forward/readout backward/replay with a live readout fixture.
The readout fixture is nonzero because frequency initialization intentionally
has no initial contextual input gradient; no fit or optimizer update is involved.
Global model laws are not inferred from a toy performance score.

Random draws/storage increase from HU to NHU per event/stage for both tapes;
score/value/optimizer work remains charged. This candidate is not a reason to
restart the protected-depth fits or postpone their diagnosis. First finish the
controlled depth comparison and branch-versus-parent attribution. Admit a
bounded fixed-TRAIN gradient/covariance and work audit only if it changes the
next learner choice; then use one integrated paired continuation if warranted.
Do not introduce a lane-noise/horizon/width grid. Keep acquired state, tokenizer,
full vocabulary probabilities, chronological observations and online learning
policy explicit under the capability-first admission rule.
