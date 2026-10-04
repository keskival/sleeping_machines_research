# Route identity and computational time must be tested separately

The owner’s completed Mackey–Glass tau18 development diagnostics support a
useful same-weight deterministic-inference improvement. They do not yet isolate
winner randomness from time randomness. This note complements §417 and retains
the original numbers, programs and interpretation history. No official tau17
score or new native fit is supplied here.

For independent exponential clocks with rates r_j=exp(s_j), let R=sum_j r_j,
W be the winning identity and D the earliest delay. Their joint density is

    f(W=j,D=d)=r_j exp(-R d)= (r_j/R) [R exp(-R d)].

Thus W~Categorical(r/R) and D~Exp(R) are independent conditional on the
pre-race state. All-noise-one execution instead chooses argmax r and delay
1/max r. The latter is R/max r times the sampled mean delay 1/R, between one
and U. With equal rates it lengthens that mean by U. This changes decay,
rotation, context alignment and subsequent races as well as winner identities.
The existing deterministic variant remains valid and worth testing; attributing
its entire quality gain to categorical output noise is not established.

For a fixed prefix, four useful inference diagnostics are:

| Winner | Delay | Question |
|---|---|---|
| categorical r/R | exponential R | Original distribution |
| argmax r | exponential R | Remove identity noise only at this conditional state |
| categorical r/R | 1/R | Remove delay noise only at this conditional state |
| argmax r | 1/R | Remove both using the original mean delay |

Keep the historical argmax/1-max-rate program as a fifth arm. Recompute rates
causally at every changed state; a long altered trajectory does not preserve the
original unconditional delay distribution. Factorized exponential/categorical
sampling reproduces the original law, not its original per-unit random stream
or necessarily finite-precision ties. Shared uniforms can pair new arms, but
must not be labelled identical to historical seeded trajectories. No inference
program is changed by this note.

## What squared loss proves, and what it does not

For output Y and fixed target y, E[(Y-y)^2]=(E[Y]-y)^2+Var(Y). Removing output
variance without changing its conditional mean reduces squared risk. Greedy
route selection need not preserve the mean; it can improve or worsen risk.
Likewise E[F(state,noise)] is generally not F(E[state],E[noise]). A deterministic
internal delay or route is not an exact ensemble prediction. sMAPE has different
geometry, so the squared-loss identity is a diagnostic rather than a theorem of
sMAPE improvement. Independent output ensembles have additional inference cost.

For the scalar linearized recurrence e_(t+1)=a e_t+epsilon_t with independent
zero-mean increments of variance sigma² and zero initial error, terminal
variance is sigma² sum_(k=0)^(h-1) a^(2k). It grows with horizon for a>=1,
but real native feedback is nonlinear and errors may be biased or correlated.
Measure actual local prediction bias, variance and paired rollout divergence
on TRAIN-only held-out prefixes before assigning this explanation to a fit.

## Temperature and credit contracts

Scaling scores by 1/tau concentrates categorical mass on a unique maximum:
with margin m>0, losing probability <=(U-1) exp(-m/tau). It also changes
R_tau=sum exp(s/tau), hence the clock scale. A temperature intervention that
preserves a chosen total hazard needs explicit normalized rates
r'_j=R_ref softmax(s/tau)_j. Choosing R_ref and differentiating it are architectural
decisions because time performs computation. The all-noise-one delay is not
automatically the zero-temperature limit of the original complete timed program.

Linear local categorical credit remains an available surrogate for deterministic
execution, but is not thereby the unbiased derivative of hard argmax risk.
Preserve the distinction between sampled expected-risk credit, winner-path
derivatives and deterministic surrogate learning. No core substitution or long
training run is authorized by a numerical claim in this note.

## Prospective admission

First use fixed owner DEV weights and held-out TRAIN prefixes, with no tau17
access: all five programs, matched warm state/reset conventions, full route/time
traces, one-step bias/variance and autonomous rollout scores. Charge candidate
scoring, selected writes, state initialization and every repeat. Contract the
factorized law, dtype/tie conventions and causal state transitions before a
guarded integrated fit. Then compare matched fitting budgets with the same
training/inference clock choice. Owner MG round4 remains prioritized; avoid
duplicating its deterministic-training fits. Official confirmation remains a
separate frozen protocol, not selection on an already observed test outcome.

The stdlib script `analysis/aws_race_clock_factorization.py` checks survival and
joint-density factorization, delay ratios, exact finite output-risk counterexamples
and linearized variance sums. Its saved artifact is constructed mathematics,
not a native contract, measured advantage or additional benchmark result.
