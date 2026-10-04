# Three complete members, one specification

[Overview](model_family_overview.md) · [Formal core](model_family_specification.md) ·
[Design choices](model_family_design.md) · [Composition](model_family_composition.md)

These are **fully specified illustrative members**, not newly trained models,
benchmark recipes or replacements for the current native research architecture.
They use the same twelve fields to show how the family admits contrasting
choices. Their small dimensions make the contracts inspectable; those
dimensions do not define the family. Real-number equations describe reference
semantics; numerical equivalence requires a separately declared precision.

| Axis | C: causal statistic endpoint | R: selective temporal learner | H: mixed reception and dense query |
| --- | --- | --- | --- |
| Evidence / state | Binary sequence; four counters | Binary sequence; two private scalar memories | Two observed sources; shared vector, counts and two recent records |
| Temporal computation | Token order; no learned delay | Decay, exponential race, bounded learned computational delay | Decaying accumulation, silence deadlines, age-dependent query |
| Participation | Addressed increment; full two-class distribution | Two keys/proposals, one committed slot and delivered value | Selective timeout/state operations; all stored values at dense query |
| Learning | Exact causal increments; no gradient TTT | One-step online gradient update with local alternative teacher | Exact finite-option episode utility and shared-map gradients |
| Composition | Statistic → readout | State/key/value programs → race → readout | Sparse arrivals → shared receiver → bounded record bank → dense island |
| Target coverage | Statistical endpoint | Core temporal/selection example; no deep stack or full writer credit | Hybrid capability example; no native race backbone or measured integration |
| Evidence status | Specified example; related causal estimators implemented | Specified example; related native mechanisms implemented | Specified composition; complete project integration remains aspirational |

## C. Causal two-symbol statistic

| Field | Specification |
| --- | --- |
| E | Observe x_k∈{0,1}, k=0,…,n−1, n≤1024. Predict the next symbol using only x_0,…,x_k |
| O | Exact nonnegative integer increment, row sum, addition and division; smoothing constant is one |
| G | Observed-context table → two-class probability readout; no hidden neural branch |
| S | Four counters C[a,y]=0 initially; previous symbol initially absent; stored preceding forecast and validity flag. Reset between streams |
| I | Symbol selects the table row; observing the next symbol selects the increment column |
| P | Increment one addressed cell after its outcome is observed; read both row entries for prediction; no scored learned keys |
| T | Observation and modeled order are k. Each step updates the previous transition before issuing the new prediction; no pending timers |
| Q | After observing x_k, emit the full distribution p_k(y) below. Completion is immediate in modeled time |
| L | Next-symbol negative log probability, scored when x_{k+1} arrives |
| U | First score the stored preceding forecast, then increment C[x_{k−1},x_k] for k>0. No learned parameter or optimizer update |
| B | At most1024 symbols; reject longer streams unless an explicit reset starts a new one. Counts cannot exceed1023 transitions |
| X | Illustrative reference. Related [causal continuation statistics](../sleeping_machines/count_continuation.py) use richer laws; this is not their reported score |

After the causal increment, predict

    p_k(y) = (C[x_k,y]+1)/(C[x_k,0]+C[x_k,1]+2).

For observations 0,1,0, the successive predictions of symbol1 are 1/2,1/2,2/3.
The last prediction uses the already observed transition0→1; it does not
consume the unknown fourth symbol. State adapts during deployment, but no
gradient-based TTT occurs. A learned residual or temporal region can be added
through an explicit fusion path; it would be another member or declared policy.

Work per step is one increment after the first symbol, two counter reads and
constant-size normalization/output. Storage is four bounded counters plus the
previous symbol, validity flag and stored forecast. Their bits and division
cost belong to a deployment measurement; no neural FLOP comparison is claimed.

## R. Two private temporal programs with a hard race

Let the observed binary symbol map to x=2*x_k−1. Parameters are
theta=(b_i,c_i,rho_i,w_i,g_i for i=0,1; alpha,gamma,u). Initialize
b=(.1,.1), c=(.2,−.2), rho=(.5,.5), w=(.3,−.1), g=(.5,.5),
alpha=gamma=1, u=0. Parameters supply reusable rules; memories supply facts.

| Field | Specification |
| --- | --- |
| E | The same causal binary streams and next-symbol scoring as C |
| O | Scalar exp/decay, multiplication/addition, clipped scores, hard minimum, logistic readout and stable logistic loss |
| G | Two private state/key/value programs → one race → scalar readout; a single temporal layer |
| S | m_i=0, last_i=0 and ready=0 initially. Retain a versioned one-step prediction/credit record. Reset state per stream; parameters persist across streams |
| I | Both programs receive the observed x and modeled arrival a; key scores and delivered values have different parameter roles |
| P | Compute both scores and proposals; choose one winner, commit one private write and deliver one value. Other memory/timestamp is unchanged |
| T | a=max(k,ready); independent positive Exp(1) draws epsilon_i assigned by stream/step/slot; lowest slot wins any numerical tie |
| Q | Predict after the selected emission at modeled time a+d, using observed cutoff k. There is no access to x_{k+1} |
| L | Binary logistic loss ell(z,y)=log(1+exp(z))−y*z, y=x_{k+1} |
| U | One-step local content/readout and clock derivatives plus the value teacher below; SGD eta=.01 after observing/scoring y. Incoming carried state is detached; no future-write credit |
| B | n≤1024. rho is projected to [.01,10], other parameters to [−10,10] after each update. Scores clipped to [−12,12]. Finite two-slot/one-layer execution |
| X | Illustrative specialization of temporal/selection ideas. Actual [native units](../sleeping_machines/addressed_event_heads.py) use vector modes, heads and depth; [batched credit](../sleeping_machines/batched_episodes.py) has its own source contract |

With current state and parameters, form

    s_i = clip(c_i*x + b_i*m_i, −12,12); lambda_i = exp(s_i)
    m'_i = exp(−rho_i*(a−last_i))*m_i + w_i*x
    v_i = x + g_i*m'_i
    winner = argmin_i epsilon_i/lambda_i; tau = min_i epsilon_i/lambda_i
    d = .001 + .010*tau/(1+tau)
    z = alpha*v_winner + gamma*d + u; prediction = sigmoid(z).

Commit m_winner=m'_winner and last_winner=a; set ready=a+d. Timing changes
the queried representation, and memory changes later scores and proposals.
This is genuine content/time/state coupling, despite deliberately small maps.
Each chosen delay lies between .001 and .011; finite input and one layer bound
the execution. This example does not provide a clockless hardware realization.

The learner retains the producing inputs/state/parameters/random draws until
the next symbol arrives. It scores the saved prediction before updating theta
and before predicting from the newly observed symbol. No intervening parameter
update occurs; the retained version is therefore current at this update. It
differentiates selected content and readout with carried state held fixed.
With pi_i=lambda_i/sum(lambda), v_bar=sum(pi_i*v_i), and
g_v=alpha*(sigmoid(z)−y), its additional score teacher is

    value_choice_i = pi_i*g_v*(v_i−v_bar).

Proposal values and g_v are constants inside that score teacher. The specified
clock score contribution is

    clock_i = gamma*(sigmoid(z)−y)*.010/(1+tau)^2 * (−tau*pi_i).

Apply these score cotangents through the score maps, including their clipping
rule (zero derivative at and outside the saturation boundary); add ordinary
selected proposal/readout derivatives. The clock rule uses the factorized
conditional exponential-time estimator, not the derivative of the same
per-candidate epsilon realization's hard minimum. Value choice credit is the
local linearized teacher; it is not an exact expected sequence-risk gradient.
The future effects of the selected write and ready time are omitted. Gradients
into prior memories or earlier producing parameters are explicitly disabled.

The reference always pays two key scores and two proposals, one commit and one
delivery per prediction, plus a constant-size saved record and parameter update.
An equivalent winner-only inference implementation could avoid one proposal;
that optimization is a separate state/numerical/resource contract. Sparse
writes do not turn this reference's proposal work into winner-only work.

## H. Sparse reception, a dense query island and finite-option TTT

This example constructs a mixed-world **capability**, with bounded history and
an explicit shared information path. It is not evidence that the existing
native stack has completed joint multimodal training or useful TTT.

| Field | Specification |
| --- | --- |
| E | An episode has N≤8 observed events (source,t,x), source∈{frame,event}, t∈[0,8], x∈[−1,1]²; query cutoff q∈[0,8]. Admit only observations t≤q |
| O | Vector addition/exp decay, counters, finite timeout options, two-record softmax, affine maps and tanh; squared scalar prediction loss |
| G | Both adapters feed one shared silence receiver; emitted vectors enter one bounded record bank; a dense local query combines records and source counts |
| S | Zero accumulator/counts, absent last/deadline, empty two-record FIFO; selected timeout, retained ≤8-event causal prefix and pre-update maps for replay. Reset each episode; parameters persist |
| I | Source IDs increment their own count; both payloads enter the same accumulator. Query explicitly reads both counts and available records |
| P | Once per episode select H∈{.5,1.5} from softmax(zeta). Every observed event updates the receiver; every closed burst stores one record; query recruits all stored records |
| T | Sort by t, then frame before event, then original within-source index. Arrival-before-timer ties. Each admitted arrival renews deadline=t+H; do not flush pending work at query/EOF |
| Q | Process all admitted arrivals and timers through q; inspect only closed records and counts. Complete one dense query at q; discard remaining pending state only at the declared episode reset |
| L | Forecast scalar y_hat, then observe y∈[−1,1] and score (y_hat−y)²/2 |
| U | After scoring, replay the same observed prefix separately under both fixed H options; exact expected finite-option loss and query-map derivatives; SGD eta=.01 once per episode |
| B | At most8 admitted arrivals and8 emissions, two retained records, one query. Two paid replay executions; no internally recurrent event generation. Reject out-of-domain episodes |
| X | Specified hybrid; [silence reference](../sleeping_machines/silence_burst.py) implements the local primitive. The complete shared-source/dense-query/TTT composition here has no project benchmark claim |

For an arrival at t, first fire a pending timer only when deadline<t. At a
firing d, emit r=exp(−.2*(d−last))*accumulator, store (r,d), evict the oldest
record if necessary, and clear accumulator/last/deadline. Then for the arrival
decay the active accumulator by exp(−.2*(t−last)), add x, increment its source
count, set last=t and deadline=t+H. An inactive accumulator starts at zero.
After the final admitted arrival, a deadline≤q fires; a later one remains
pending. Querying never creates an early emission. No later observation is
used to decide whether a prefix timer fires.

Let C be the two-count vector. Query maps are
phi=(B,b,K,V,W,c,a,e,r_age), initially B=K=V=W=identity2,
b=c=zero2, a=(1,−1), e=r_age=0; timeout logits zeta=(0,0). For each
closed record (r_j,d_j), compute

    h = B*C+b; key_j = K*r_j
    score_j = h·key_j + r_age*(q−d_j)
    value_j = exp(−.1*(q−d_j))*V*r_j
    z = sum_j softmax(score)_j*value_j + W*C+c
    y_hat = a·tanh(z)+e.

For an empty bank the sum is zero. With two records both values contribute;
this region is a full-support dense island. Its dense arithmetic and key/value
traffic are part of the system work. The irregular receiver has no periodic
global update requirement. “Frame” is an observed source type here, not proof
of real image understanding or a learned pixel adapter.

Let ell_H be the actual post-query loss under option H, with its own emitted
records/times and state history. Learning uses

    J = sum_H pi_H*ell_H
    dJ/dzeta_H = pi_H*(ell_H−J)
    dJ/dphi = sum_H pi_H*dell_H/dphi.

The replay inputs and query rule never depend on the newly revealed target;
only losses and updates do. Membership depends on the fixed timeout options
and observed times, not phi. This makes the finite-option derivative exact
for this episode objective, including silence/record effects; it is not a
continuous gradient through changing timeout boundaries. SGD uses the frozen
pre-update phi,zeta for both outcomes, then projects entries to [−10,10]. It
preserves those parameters for the next episode and resets only inference
state. This is a fully specified causal gradient-TTT policy, without a measured
adaptation benefit. Target access is never an event-adapter input.

Charge factual execution and query, both complete replay outcomes, their maps
and gradients, optimizer and resets. Replay duplicates receiver/record work.
State capacity, event activity, dense query work and alternative learning
budget are separate entries. The two-record forward bank does not replace the
retained ≤8-event learning prefix; charge that replay storage as well. The two-option scope avoids claiming an
affordable exact learner for arbitrary future routes or graph morphing.

## How these specifications connect to the research system

The examples demonstrate that one specification can express a statistical
endpoint, a selective temporal learner and a mixed composition. They do not
cumulate into a new completed model. The fitted native language member's actual
source and saved recipe remain in the [architectural review](architecture_review.md)
and [result bindings](architecture_evidence.md); it has different vector
operators, reset/horizon and optimizer settings from R. No illustrative update
rule is silently substituted into that model.

Use these twelve fields when proposing a real member. Then supply concrete
dimensions, initializations, data/label protocol, numerical domain, source and
queue versions, and checks matching the claimed forward/learning/resource
contract. This connects the overarching family to reproducible constructions
without defining it by today's particular experiments.
