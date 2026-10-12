# Beyond credit prediction: useful information and discovered options

12 October 2026, B1/R1. A complementary view of the current native credit programme. These are mathematical constructions and test implications, not measured advantages of a new model.

## The blind spot in a fixed candidate pool

An estimator trained on a fixed action set can learn to rank those actions perfectly while missing the best action outside its set. This is not necessarily a prediction error: the execution interface may never offer the useful write, route, memory address or learning intervention. Exploration can therefore improve two distinct objects: knowledge about available actions, and the set of actions that can actually be executed.

Our current common-state language experiments offer stopping, replay and fresh experience. Their positive fresh-data returns do not identify the value of discovering an unoffered intervention. The native credit model similarly samples alternatives from a bounded current candidate list. A supported estimator cannot correct a candidate that never enters its support.

## Decision information differs from reconstruction information

Let b be a belief over unknown future dependencies, and let A be a finite executable action set. Let ell(a,z) include future task loss and declared action cost under a fixed continuation. Define Bayes decision risk

    R(b,A) = min_(a in A) E_(z~b) ell(a,z).

A pure information query produces Y and posterior b_Y, without physically changing the actor or the feasible actions. Its gross decision value is

    V_info = R(b,A) - E_Y R(b_Y,A) >= 0.

Proof: the minimum of linear functions of b is concave, and E b_Y = b. Equivalently, the post-query learner can always ignore Y and take its pre-query action. Query acquisition, processing and storage costs must be subtracted. This is a standard Bayesian decision-value argument, not a novelty claim.

The quantity can be zero even when the query reveals many bits. Let z=(u,v), with independent fair bits, and let actions predict v under 0/1 loss. Revealing u gives one bit of information and zero decision improvement. A noisy signal of v, correct with probability 0.6, gives only 1-h2(0.4) approximately 0.029 bits, but reduces decision loss from 0.5 to 0.4. Thus maximizing information about the full latent world can choose the worse learning investment.

Mutual-information experimental design is an established useful approach, including [Kleinegesse and Gutmann, ICML 2020](https://proceedings.mlr.press/v119/kleinegesse20a.html). Here the criterion is the actual future decision problem. Our earlier Gaussian demand-weighted information formula is one tractable local case; this risk formulation also covers nonlinear, discrete and initially unknown dependencies.

## Discovery has an additional value

Suppose a paid query also reveals executable options A_Y, with A contained in A_Y. Then its gross value decomposes exactly:

    R(b,A) - E R(b_Y,A_Y)
      = [R(b,A) - E R(b_Y,A)]
        + E [R(b_Y,A) - R(b_Y,A_Y)].

The first term is information value over old options. The second is option-discovery value. Both are nonnegative under the stated pure-information, nested-set assumptions. A newly accessible option can have value even if the belief is unchanged. For example, a deterministic known task has loss 1 under every currently executable route; discovery makes an already understood zero-loss route executable. Its value is 1 despite zero information about the latent task. This illustrates availability, not a claim that realistic route effects are already known.

If the query itself writes memory, changes weights, delays a deadline or consumes an option, it is not a pure information query. Include that physical state transition and the changed costs in the full intervention value; the two nonnegativity statements then need not hold. Actual counterfactual worlds must include those effects. This distinction matters for native temporal races: a late useful message and an immediately executable useful message can have different value.

## A credit representation need only preserve relevant decisions

The strongest sufficient representation preserves all future transitions and losses. That can be unnecessarily expensive for a specific intervention interface. If compressed scores satisfy

    sup_(a in A) |q(a)-q_hat(a)| <= epsilon,

then selecting a_hat minimizing q_hat incurs regret at most 2 epsilon relative to minimizing q. Proof: insert q_hat at a_hat and at the optimal a; the estimated-score difference is nonpositive and the two score errors sum to at most 2 epsilon. A representation may discard large common offsets without damaging rankings. Conversely, a small average prediction error can hide a large error on the newly relevant action.

This bound concerns the represented action set and distribution. It says nothing about omitted options or a future regime outside the fitted support. Decision sufficiency must therefore be tested after new dependencies and new options appear, rather than inferred from ordinary forward reconstruction or current loss regression.

The native temporal memory can retain reusable evidence about operation effects, together with uncertainty, freshness and the conditions in which it was useful. Current-operator anchors identify immediate consequences. Neither mechanism should be forced to reconstruct every detail of the world before making a useful credit decision.

## Large-data advantage is repeated opportunity, not token count alone

Suppose one acquired record yields conditional expected decision improvement delta at each relevant reuse. If reuse opportunity at step k has probability rho_k, discounted utility and survival of the acquired relation give

    V_record = sum_(k=1..H) gamma^k rho_k s_k delta_k - C_acquire - C_keep.

For constant rho and delta, and permanent invalidation hazard h, s_k=(1-h)^k. With r=gamma(1-h), gross value is rho delta r(1-r^H)/(1-r), or rho delta H when r=1. These are declared simplified assumptions, not fitted constants. A returning regime needs a recurrence model rather than permanent invalidation.

Consequently, many new tokens can still provide little amortization if the relation never recurs. A smaller stream can expose substantial amortization if a learned dependency recurs many times across distinct targets. Constructed large-domain emulation should vary distinct target count, number of underlying relations, recurrence horizon, discovery burden and drift separately. Repeating identical examples changes a different axis. The existing fresh/replay result establishes exposure value under its fixed protocol; it does not isolate these axes.

## A conditional family-scaling heuristic

Let N count distinct future targets and K(N)=c N^beta count useful relations that must be acquired, with 0<=beta<=1. Suppose acquisition costs a per relation, serving/learning selected work costs w per target, and each target's discovery costs d K(N)^delta. A declared work model is

    W(N) = w N + a K(N) + d N K(N)^delta,
    W(N)/N = w + a c N^(beta-1) + d c^delta N^(beta delta).

This is resource algebra, not a quality scaling law. When beta<1, reusable acquisition tax per target falls with data. But scanning every relation (delta=1) can overwhelm that advantage. Bounded discovery (delta=0), or a separately charged logarithmic index, is consequently a concrete structural requirement for the desired regime. Available capacity alone does not give favorable scaling. Acquiring new relations independently at every target gives beta=1 and no falling acquisition tax.

With drift, replace K by the number of relation acquisitions and repairs actually paid. Frequent repairs can restore a linear tax even when the number of stored relations grows sublinearly. If a learned credit operator transfers across many relations, its acquisition may instead be charged once per reusable operation class; discovering that transfer is the mechanism that changes the work model. Test whether the acquired operator generalizes, not merely whether the same episode is replayed.

This supplies an economical emulation target: vary relation count and recurrence independently, preserve distinct task targets, and measure acquisition/repair/discovery separately. Choose a family member that reduces these measured terms while retaining prospective task quality. It does not predict a Transformer crossover or authorize an unmeasured FLOP claim.

## Closing the training loop

A finite paired future rollout gives the net value target for an issued query or discovery operation. Copy the complete learner state, including optimizer and numerical memory; charge both the discovery and the continuation; permit subsequent decisions to use acquired information/options only in the intervention world. Compare with the no-acquisition world under coupled exogenous futures where possible. The target terminates in observed task loss and resource use, not a higher learned trainer.

Curie's reciprocal records provide a physical path for this evidence to affect future forward decisions. The current task-taught bank supplies a retained task-gradient path, while paired audits can teach discrete and out-of-window consequences. AWS's acquired-state fingerprints supply the reproducibility discipline for the two worlds. Combining these mechanisms is a construction priority; the existing two projects have not yet measured that combined system.

## Next integrated test implication

Do not fit a controller solely on the current always-fresh examples. Once the current owner pipeline is complete, define a changing-dependency task where (i) useful old relations recur, (ii) genuinely new relations arrive, and (iii) irrelevant novelty is present. Include candidate discovery as a charged action rather than silently expanding the pool for free. Fit only from anonymous causal observations, never hidden generator relations. Prepared Curie rewiring/foreign-event data inform this design, but their deferred transfer fits remain deferred.

The decisive endpoint is lower prospective task loss at the same complete learning budget, with a crossover in cumulative paid work as reuse grows. Contrast discovery-enabled native credit with the same native learner restricted to current candidates and with fixed-cost acquisition. Use a predeclared data/horizon ladder; immediate small-data loss is not a universal rejection gate. No new fit is admitted by this note. The current larger language confirmations finish first, and compact/deep credit integration stays with its owner.
