# Normalized clock noise and precision credit

4 October2026. Derivation and an isolated native prototype; native numerical
admission and actual learning outcomes remain pending.

## Concrete failure and what the evidence isolates

The owner’s completed Mackey–Glass tau18 development fits show sampled
p16/D2/pool2 increment prediction at25.330705sMAPE versus18.450703 with greedy
inference, with matching recorded training-loss histories. Pool1 gives16.434570.
Deterministic training/inference gives18.423278, essentially unchanged at this
three-repeat resolution; wider deterministic p32/3000steps gives20.861426.
Keep these positives and negatives. Tau18 development is not the official tau17
30-repeat score. Pool1 removes alternative routing and is a diagnostic variant.

These comparisons show that execution policy matters. They do not isolate
random winner selection from random elapsed clock: sampled inference changes
both. The original first clock is min(E_i/exp(s_i)); greedy clocks are
1/exp(max(s_i)). Their means differ. Those clocks subsequently affect transport,
memory ageing, incoming context and future races. A precision regression task
can be sensitive to clock noise even if winner uncertainty remains useful.

No learning-regression diagnosis follows from this comparison alone. In
particular the deterministic-training result does not establish that suppressing
every form of exploration improves deep learning.

## Derivation: retain routes, control common-clock variance

Condition on the actual entering state, candidate scores s_i and proposed
values. Let lambda_i=exp(s_i), Z=sum_i lambda_i, E_i iid Exp(1),

    winner = argmin_i E_i/lambda_i,
    T = min_i E_i/lambda_i.

The joint density of winning candidate i at t is
lambda_i exp(−Zt)=(lambda_i/Z) Z exp(−Zt). Therefore winner has probability
pi_i=lambda_i/Z and is independent of W=ZT, which has distribution Exp(1).
This is the existing race factorization, not a new categorical relaxation.

For a fixed clock-noise parameter nu in[0,1], define

    T_nu = W^nu / [Gamma(1+nu) Z].

Compute the winner with the original candidate clocks and change only its
common first clock. At **fixed entering scores**, the winner is exactly the same
for every nu and every realized noise draw. Native episodes can subsequently
take different routes because changed clocks affect evolving messages/memories;
we do not claim equality of complete trained trajectories across nu.

The exact unbounded moments are

    E[T_nu] = 1/Z,
    Var[T_nu] = [Gamma(1+2nu)/Gamma(1+nu)^2 − 1]/Z^2.

The coefficient of variation is1 at nu1,0.522723 at nu0.5,0.280544 at nu0.25,
and0 at nu0. nu1 is the original clock. nu0 gives1/Z while keeping sampled
winners. It differs from greedy inference, which also changes the winner and
uses1/max(lambda). The family exposes a clock-precision degree of freedom
without reducing the route-credit support or selecting additional values.

The normalization is conditional on the current state. It does not guarantee
unchanged distributions of future memory, expected prediction loss or learned
representations after the transformation.

## Credit remains coupled to the representation

Holding the independent **W** fixed, rather than the original candidate noises,

    dT_nu/ds_i = −T_nu pi_i,
    sum_i dT_nu/ds_i = −T_nu.

This is the existing factorized clock teacher with its realized first clock
replaced by T_nu. Native message counterfactual credit still uses pi_i and the
same alternatives. Keys, incoming content and persistent state still set Z,
the clocks, the winning route and future representations. Eliminating clock
noise does not eliminate computational time or its derivatives.

This is a conditional distributional reparameterization. Differentiating a
hard sampled winner at fixed original E_i would produce a different pathwise
derivative; the scalar finite-difference check explicitly holds W fixed.
The existing local message credit remains a linearized surrogate: this law
does not repair missing persistent-write utility, future return curvature,
history truncation or categorical boundary credit for the entire core.

For a **fixed unbounded-clock cotangent** c, the score gradient is g=−cT_nu pi.
Its conditional mean is−c pi/Z and its covariance is
c^2 Var(T_nu) pi pi^T. The clock gradient's relative variation can therefore be
reduced exactly in this restricted witness. Actual cotangents depend on the
delay, winner and future state; clipping and historical Adam moments are
nonlinear. No actual optimizer-variance or prediction improvement follows from
this witness alone. Existing deep replay route-site variance findings133/135
concern a different noise source and remain intact. Shared race draws across
lanes also preclude treating the lane gradients as independent samples.

An eventual learnable nu would have
dT_nu/dnu=T_nu[log(W)−digamma(1+nu)]. The current implementation uses a **fixed
protocol parameter**, without that derivative or a learned/annealed schedule.
Do not describe it as demonstrated uncertainty adaptation.

## Native bounded arrival and resource boundaries

The native delay is h(T)=.001+.010T/(1+T). The invariant mean above concerns T,
not h(T). Since h is concave, reducing variation can change mean arrival;
nu0 has h(E[T]) while nu1 has E[h(T)]<=h(E[T]). Thus this is a route-preserving,
unbounded-mean-calibrated control, not a fully mean-matched native-arrival
intervention. A separate bounded-arrival normalization would require a different
calculus and additional work. This limitation is part of the experiment.

All keys are still scored and the prototype fitting/evaluation paths compute
all proposals. Extra logsumexp/log/exp work and Gamma normalization/setup are
paid; compiler/setup, resident state, traffic and energy are not declared zero.
There is no new selected-value count or demonstrated sparse-serving saving.
If a K1 trained control eventually matches an existing K8 quality result, a
measured same-quality serving comparison would then be warranted.

## Implementation and gated experiment

- `sleeping_machines/clock_noise_law.py`: standard-library algebra, exact moment
  formulas and fixed-W clock credit.
- `sleeping_machines/clock_noise_episodes.py`: an eager native event program with
  a private function-global dictionary and custom clock race, plus an explicit
  independent layer port/compile target. Existing batched/compiled/sparse kernels
  and defaults are untouched. nu1 calls the original eager function directly.
- `experiments/clock_noise_admission.py`: source/data/prerequisite/physical-lock
  checks before numerical imports. Native gates cover FP64 custom versus layer
  logits/every gradient, exact nu1 eager endpoint, unchanged forward from local
  credit, actual compilation, FP32 every-gradient/two-Adam tolerances and actual
  serialized next-update recovery. These are **prepared, not passed**.
- A prerequisite-gated integration pilot uses only official **tau19 DEV**,
  repeat4, nu1/0.5/0 with the same initialization and FIT sampling seeds,8lanes,
  64positions and32Adam updates per arm. It forecasts750autonomous targets after
 750teacher-forced warm inputs. All proposals are evaluated and first full eager
  update work is extrapolated. Saved checkpoints bind config/source/data/RNG.
  A tiny fit tests finite integration; it cannot establish benchmark superiority.

Physical owners run these two unique one-job queues only after their existing
assignments.1CPUthread,2,000,000KiB RSS watchdog,6,000,000KiB VMS,8GiB available
floor and900s timeout per stage. Initial guards are conservative, informed by
completed p16/D2 MG1500-step~1100s runs and prior small compiled admissions;
pilot additionally requires measured contract RSS within its cap. Any timeout,
RSS failure or numerical failure is retained and requires a uniquely tagged
revision; no second trainer or container-local lock admission.

**13 standard-library checks pass**, covering endpoint/conditional winner
invariance, mean/variance/score-gauge algebra, fixed-W finite differences,
quantile normalization, the bounded-delay caveat, invalid values, immutable
sources/manifests, completed-output/interrupted-checkpoint preservation,
prerequisite and container rejection. No Torch/NumPy/HDF5 or native model was
executed in this workspace. Numerical contracts, pilot quality and advantage
remain pending. No official tau17 cell is filled from this proposal.
The additional forecast check pins the warm prediction and751..1500 target
alignment and mirrors each FP32 feedback addition; base+cumsum would use a
different rounding order. Native forecast parity still requires numerical
admission; the pilot does not inherit a winner-only serving contract.

## Relation to prior work and allocation

[Gumbel-Softmax](https://arxiv.org/abs/1611.01144) and
[Concrete](https://arxiv.org/abs/1611.00712) supply established precedents for
reparameterizing discrete stochastic learning and temperature-controlled
relaxations. Our prototype leaves the **hard categorical winner law unchanged**;
nu controls the independent common-clock noise. The distinction matters for
interpreting optional route credit, clock precision and paid activity. No claim
of priority over all clock-race literature is made.

Preserve the owner’s §417 observations and official run choices. The AWS
six-session primate r1 queues and tau17from0/10/20 queues, curie MG round5 and
primate/SHD development, and source-frozen streaming replay remain priorities.
This deferred control complements them and does not reselect a public model
after opening its TEST. Deep persistent event representations, temporal
computation, keys/values, sparse writes and counterfactual learning are retained;
full useful-write credit, trained sparse serving, independent replication and
energy remain separate gaps.
