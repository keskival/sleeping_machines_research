# Learned exploration as credit to a learning intervention

User direction,10 October2026; B1/R1 shared enabler. Design/derivation only,
no new fit or measured efficiency claim. Builds on UNSEEN_ROUTE_CREDIT.md and
ECONOMICAL_EVENT_CREDIT.md. Curie owns native route/learned-credit integration;
AWS retains its admitted numerical-development and exact-credit queues.

## Objective and causal state

An exploration action e specifies whether to probe, a route/region proposal,
probe type and budget. e=0 performs no additional exploration. Its consequence
includes the information acquired and subsequent actor/critic/optimizer/state
updates, not merely the immediate loss from forcing a route. Let X include
those versioned states and causal history. Define

Q(X,e)=E[sum_{k=1}^H w_k loss_{t+k}+lambda work | X,do(e), declared subsequent learning policy].

The exploration advantage is Q(X,0)-Q(X,e). Changes in future policy, weights,
persistent state and optimizer state belong to this objective. A helpful
prediction or a large gradient alone is not an exploration reward. Separate
an information-only probe from an intervention that changes the delivered
forward event; their objectives and costs differ.

Factor a learned policy into a probe gate, lazy region/route proposal and
probe-budget distribution. Inputs may reuse persistent credit state, current
prediction error after feedback, route descriptors, race margins, elapsed age,
versioned residual calibration and paid work summaries. No future loss may
condition the action whose outcome it labels. Thresholds/budgets may be learned
subject to a resource constraint; full-support discovery is a structural
identifiability requirement, not evidence that the learned policy is optimal.
A no-probe atom and supported low-cost proposal allow learning both when and how.

## Learning without routinely running two futures

For sampled e~pi_phi(.|X), observe the factual delayed return R=-sum w loss-lambda work.
The score-function estimator (R-b(X))*grad_phi log pi_phi(e|X) learns the declared
policy objective; an action-independent causal baseline changes variance, not
its conditional expectation. A learned no-exploration value can be this baseline
if fixed before the outcome and detached in the actor update. It need not be an
exact paired counterfactual for this estimator to work. Sequential actions need
causal return-to-go/critic bootstrapping; raw same-version contextual comparisons
must not treat intervened future policy as fixed. Randomized supported actions
provide population evidence, not the unobserved outcome of this exact episode.

For a value-model control variate f(X,e), fixed before action/outcome, use

 sum_e grad_phi pi_phi(e|X) f(X,e)
 + (R-f(X,e))*grad_phi log pi_phi(e|X).

Its conditional expectation is the score-function policy gradient. Exact
summation is paid; for a lazy hierarchy its first term needs a supported,
independently sampled estimate or a tractable restricted action distribution.
Do not query every dormant route to compute a supposedly sparse estimator.
Off-policy data require the appropriate recorded propensities and, for changed
sequential policies, trajectory corrections; one route probability alone does
not correct a whole changed continuation. Version stale models/updates explicitly.

## Cheap local utility and its boundary

Let an exploratory probe change the update by delta_theta relative to the
reference learning action. On a later issued loss at the appropriate version,
local estimated benefit is -grad(loss_future)^T delta_theta. Existing factual
backward cotangents can supply that projection when the update producer map and
parameter/state versions match. Sparse addressed updates can limit the
contraction to their support. A persistent utility critic can amortize prediction
from error/credit packets and train on already issued delayed feedback.

This linearization excludes curvature and the transport through intervening
learning/optimizer/persistent-state transitions. Their JVP/VJP or learned
transport is extra work; retained first-order scan adjoints do not automatically
supply higher-order meta-gradients. Sparse delta_theta does not imply sparse
influence on future computation. Occasional paired same-prefix shadow rollouts
with matched exogenous draws audit the surrogate, including all learning/state
updates and alternative downstream paths. They are not routine free labels.

If two audit policies produce the same conditionally unbiased gradient mean,
the expected first-order learning benefit is identical under a fixed linear
update map. Their utility difference can be second order: locally, expected
loss includes eta^2/2 * tr(H P Cov(g_hat) P^T), with locally fixed preconditioner
P and curvature H. Thus residual-risk allocation minimizes an economical
variance surrogate, not necessarily actual future-loss regret. Learned utility
must test downstream quality, curvature/drift and nonlinear optimizer effects.

## Reuse, discovery and first decisive experiment

Reuse issued forecasts/errors, existing exact/counterfactual audits, versioned
cotangents and stored route packets as observations. Adaptively adding promising
routes to the paid counterfactual pool is the learned exploration policy's
output; the pool itself is not the universe. Existing selected-route labels
alone cannot identify utility outside their support. Descriptor generalization
supplies predictions; supported discovery supplies corrective evidence.

Next native B1/R1 pipeline: fixed-cost no-extra-probe, supported uniform probe
and learned gate+proposal, with a declared causal horizon, initial state and
complete work budget. Separate learning utility from gradient calibration.
Measure later prequential loss, complete work-to-quality, support/propensity
correctness and shadow-audit calibration on TRAIN/DEV. Include adaptation drift
and routes absent from the immediate pool. Arithmetic contracts precede a
bounded integrated pilot; no external architecture or sealed TEST is admitted.
This note does not admit an additional queue or displace existing owned work.
