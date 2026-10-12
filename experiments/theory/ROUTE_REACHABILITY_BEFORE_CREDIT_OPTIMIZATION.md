# Route reachability comes before credit optimization

Curie, 12 October 2026. B5/R1; diagnosis of unchanged native outcome-law credit. Read with FORWARD_RESPONSE_SUFFICIENCY_FOR_CREDIT.md and DIRECT_ROUTE_CREDIT_DEPTH_AND_DELAYED_UTILITY.md.

## A restricted action space can make a correct critic ineffective

At an issued history H, let A(H) be the routes actually discovered. Their binary predictions have probabilities p_a and logits z_a. Let b be the last observed target bit. Define the signed persistence margin m_a=(2b-1)z_a. If every m_a>0, every available route gives the same binary answer b. Choosing a different hard route cannot change that answer with the decoder and issued states held fixed. Nor can any probability mixture of those routes: each assigns probability greater than one half to b, so their convex mixture does too.

This is an action-space obstruction, not an inaccurate-credit theorem. A perfect route critic can select the best confidence and reduce proper prediction loss while making no binary answer changes. Conversely, an observed fixed-answer trajectory does not prove an obstruction: unchosen alternatives might cross the decision boundary. Evaluating their actual supplied responses distinguishes those possibilities.

The claim is local to one issued history and fixed parameters. Different routes can affect later memory, parameter updates or discovery. It therefore does not rule out answer changes caused by subsequent learning, a different acquired model, or the full model family.

## Three different deficiencies require different repairs

1. Discovery: a useful source response exists outside A(H). Better ranking within A(H) cannot deliver it. Improve addressed discovery while charging its keys and messages.
2. Representation/decoder: all discovered responses remain on the persistence side despite informative public history. Improve state integration or decoder teaching. Do not remove the persistence prior merely to manufacture flips.
3. Assignment: some discovered response changes an answer usefully, but the selected route misses it. The credit interface, exploration and teacher accuracy are then directly relevant.

These deficiencies can coexist. Useful-route counts conditioned on delivered labels are diagnostic upper bounds on the fixed discovered set, not a deployable selector. Only an issue-time rule can realize the opportunity.

## Why proper-loss gains can coexist with a failed capability gate

The existing head is z_a=(2b-1)[beta+r_a(H)], rather than an unconstrained additive absolute-target residual. With M=1{Y=b}, its BCE is softplus(beta+r_a)-M(beta+r_a), and its derivative with respect to the signed margin is sigmoid(beta+r_a)-M. Proper supervision can learn agreement probability without moving the margin below zero. A useful answer change requires the conditional agreement probability to fall below one half for a distinguishable history, not merely a lower average agreement probability.

Outcome forecasts should be evaluated on the same conditional information as the route response. A posterior that calibrates unconditional agreement, while erasing histories with a high flip probability, can guide confidence correctly and still fail to identify useful changes. Response-weighted residuals diagnose credit error; reachable decision-boundary crossings diagnose actor capability. These are different tests.

## Paid source audit and its limits

The frozen-source audit reuses the acquired source194 actor/world on four fixed TRAIN systems and the first64queries of each. It retains native memory reads/writes and delivered outcome records, freezes every parameter and restores numerical memory per system. Hooks capture candidate inputs already read by the issued forward call; extra head evaluations use those same inputs without additional bank reads or random draws. They are charged diagnostic work, not sparse deployment inference.

For every query it records whether any discovered candidate crosses the persistence boundary, whether such a crossing would be beneficial after the outcome arrives, the minimum signed margin, and route-output spread. It checks factual-route parity and parameter equality. This frozen replay differs from the original online training trajectory and cannot attribute the completed target result by itself. It identifies what the acquired actor can already express on source development data, before choosing a repair.
