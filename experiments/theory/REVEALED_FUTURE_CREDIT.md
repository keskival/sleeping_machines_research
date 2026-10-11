# Credit can use revealed futures without giving prediction a future input

B1/R1,11 October2026. Extends [gradient audit economics](CREDIT_AUDIT_WORK_FRONTIER.md) and [the shared world model](SHARED_WORLD_MODEL_AND_RECIPROCAL_CREDIT.md). This distinguishes information available when forecasting from information available when training. It derives an information path for the next integrated critic; no novelty or measured utility claim follows from the identity alone.

## Two causal information sets

At route selection, history H determines pi_theta(A|H). The actor forecasts causally and cannot see future labels. Later, a supervised future segment U is revealed and a credit update is consumed. Suppose its distribution is independent of the internal sampled route A conditional on H. This holds for the current recorded/generator continuation with internal optional writes: a write changes model state and its independent readout learner, not the external sequence. It also describes teacher-forced recorded language/market forecasting, when the model does not intervene in the data process. It does not automatically describe an embodied action, market-impacting trade, on-policy generated text or an environment changed by the model.

Let D(U,a)=L(U,a)-L(U,b(H)) be the declared learner-aware paired contrast. A critic h_psi(H,U,a) can use the already-revealed raw observations when estimating the credit for the earlier action. Keep psi fixed through this packet's prediction/consumption and train it only afterward. The audit probability p(H,a)>0 and its coin are issued before U, independently of U conditional on H,a. The raw update is

    G = sum_a pi(a|H) h(H,U,a) s(a|H)
        + I/p(H,A) [D(U,A)-h(H,U,A)] s(A|H),
    s(a|H)=grad_theta log pi_theta(a|H).

Since A is conditionally independent of U, pi(A|H,U)=pi(A|H). Conditional expectation over action/audit equals

    g(U)=sum_a pi(a|H) D(U,a) s(a|H).

Averaging over U gives the original task gradient, with the same fixed-carrier/continuation-law scope. The original actor activations and policy version must be retained or exactly reconstructed; evaluating a newer pi after feedback changes the packet. The critic receives future observations only on the learning side. No future enters the actor's issued forecast or route choice. This is ordinary supervised causality, not a physical time-symmetry premise.

Do not condition the common model term on the chosen route's loss or hidden activations as though these were route-independent U. Those depend on A. Either use raw exogenous observations and route-independent features, reconstruct typed counterfactual features per candidate, or derive a different estimator with the appropriate posterior/importance terms. This distinction prevents a target-dependent inference shortcut and preserves the audit identity.

## What variance conditioning can remove

Law of total covariance gives

    tr Cov(G|H) = E_U[tr Cov(G|H,U)] + tr Cov(g(U)|H).

With an exact deterministic future-conditioned contrast model, h=D, the supported residual vanishes and G=g(U). Action/audit sampling noise disappears; variation of the true future gradient remains. A prefix-only exact conditional-mean critic q(a)=E[D(U,a)|H] instead leaves audit noise

    sum_a pi(a)||s(a)||^2 Var_U(D(U,a)|H)/p(a).

By Jensen, tr Cov(g(U)|H) is at most sum pi||s||^2 Var_U(D(U,a)|H), hence at most that prefix-only floor for p<=1. Conditioning can therefore reduce noise which is irreducible for a prefix-only critic. The practical future model has approximation error, costs a query/training graph and can overfit; its benefit must be measured with full work. A known oracle mean gradient could bypass sampling altogether; we do not possess it, and the supported estimator is intended to remain correct under imperfect learned credit.

If only part F of U is available at consumption, the same result needs A independent of F conditional on H and a prediction conditioned on H,F. Its remaining future innovation is conditional on that packet. Waiting can buy information at the cost of delay, storage and stale parameters. Asynchronous credit can consume partial observations and later correction packets, but inclusion/duplicate handling and fixed issued coordinates must be explicit.

## Next engineering comparison

The present connected actor–critic predicts contrasts from H only and fits noisy single-continuation labels. Test an equally connected critic which adds a recurrent encoder of the four revealed raw event marks and gaps. The forecast controller retains its prefix-only interface. Preserve the pretrained whole-response structure at initialization by a zero-initialized future projection; train joint contrast predictions, not small disconnected gradient modules. Compare with a capacity-matched zero-future-input critic, the frozen prefix prior, and the same supported audit law. The continuation readout/optimizer and actual route descriptors remain part of the target.

First reconstruct observed future inputs for saved paid packets from their recorded deterministic stream coordinates, verify each reconstructed contrast against independent native replay, and fit only designated TRAIN packets. Hold out future packets chronologically, then evaluate fresh development futures with per-future contrasts rather than averaged labels that erase innovation. Charge reconstruction, all worlds, predictor queries, retained activations, optimizer work and inherited acquisition. Use completed evidence to admit the coupled actor-learning stage. This is a proposed native integration, not an additional benchmark or new external dense control.

## Registered information-path pilot

Use the valid independent512 snapshot's83 paid TRAIN packets, first75% for fitting and last25% chronological holdout. Reconstruct only recorded raw marks/gaps and causal prefix descriptors; three saved contrasts must match native paired replay. Both connected critics add the same GRU9-to24 and24-to38 future projection to the complete shared response model. Zero projection initializes exactly at the saved critic. One receives four revealed raw events, the other zeros; identical14,054 parameters, seed361,256 updates, batch<=16,AdamW lr0.001/decay0.001,clip1 and inherited TRAIN-only scale. No DEV tuning. Fresh32 prefixes/eight continuations each,512 paid native worlds, random common probe/reference and never-labelled receivers6/7 included. All72 credit values are queried per future to charge a potential supported model-gradient term.

Admission to a coupled future-credit actor stage requires lower fresh per-future MSE than both zero-future fit and frozen prefix prior, positive prefix-bootstrap95 lower bound for MSE improvement over zero-future, and lower policy-logit sensitivity-weighted residual second moment than both. This is a one-seed information-path gate, not an actor or native-parameter variance win. Chronological holdout is a separate generalization diagnosis. Report all fitting/reconstruction/verification cost and inherited512 acquisition; no inference route may use these future inputs. If the gate fails, diagnose representations and target law before further coupled acquisition.

The first contract passes prior initialization/sparse query/VJP/version/future-dependence checks. Its smoke aborts before fitting because the readout teacher was placed inside a no-gradient context. v2 explicitly enables the branch learner during reconstruction/verification and repeats contract/smoke under a fresh tag; no settings were chosen from a DEV performance result.
