# Early credit requests and the price of information

13 October 2026, B10. The failure addressed is economic: selecting a useful correction after computing it cannot save that correction's derivative work. Preserve native race memory, own-leg traces and exact four-direction sensitivities; move the work request to the parameter/window boundary. This sacrifices access to current-window errors and gradients, which can reduce decision quality. The completed post-window frozen controller is the quality reference, not a promise about early requests.

Let F be information available at the window boundary and G include its completed observations, with F contained in G. Let Delta be next-window own-update loss minus remote-update loss for the same two actual Adam proposals. For a hypothetical fixed pair of proposals and fixed future distribution, the conditional utility estimates are b_F=E[Delta|F] and b_G=E[Delta|G]. By the tower property b_F=E[b_G|F]. With constant incremental price lambda*c, the best expected utility of a binary late decision satisfies

    E[(b_G-lambda*c)_+] >= E[(b_F-lambda*c)_+].

This follows from convexity of the positive part and conditional Jensen. More information helps optimal selection; an earlier request can still be economically better because it avoids paying c on rejected windows. A later refinement after an early request is useful precisely when the new observations change the sign or magnitude of conditional benefit. This is a one-decision statement, not an ordering of complete trained trajectories: choosing updates changes actor state, Adam moments and future proposal distributions.

The economical objective is expected predictive benefit minus complete resource price. If a native request costs a, each computed remote bank costs c, and audit/teacher overhead averages d, work saved relative to always computing is

    (1-p)*c-a-d,

where p counts every computed bank, including audits. Request rate alone understates actual work. Causal online training observes Delta only after both proposals have been scored on the next completed window; its label is a counterfactual forecast comparison under fixed recorded observations, not a causal market-intervention reward.

Implementation: early requests use current numerical actor state and the previous window's state, own-clock gradients, mark frequencies, loss and elapsed time. The initial receipt carries an explicit missing-history flag. No uncomputed remote gradient is represented as an observed numerical zero. Rejected unaudited windows never instantiate the projected bank. A requested bank starts at zero after the optimizer boundary, where the declared truncated-window derivative is initialized correctly.

The first bounded implementation teaches alternating source windows and every16th online deployment window. This avoids overlapping retained controller graphs across parameter updates. These deterministic audits do not have positive selection propensity at every window; they cannot justify inverse-propensity unbiasedness or arbitrary conditional-risk consistency. Future random audits need a declared independent sampling stream and charged derivative work. Frozen deployment pays no audits. The raw v1 scope string's phrase "full-support source audits" describes availability of both endpoint labels at audited windows; it does not mean full-support random sampling. The protocol and analyses use the latter distinction explicitly.

Three actual Adam updates per forced endpoint have zero forecast, parameter and persistent-state error against saved own/projected implementations. The rejected endpoint fails immediately if paired remote differentiation is called. The source and controller retain their private memories and current-frame credit; historical critic-memory write credit and shared actor/critic banks are separate engineering projects.
