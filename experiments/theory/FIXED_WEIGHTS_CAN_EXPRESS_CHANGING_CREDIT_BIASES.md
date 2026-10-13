# Fixed weights can express changing credit biases

13 October 2026. B10 observation: acquired-frozen native guidance uses extra clock credit on37/9/7/4updates in successive4096target segments, wins over own-leg traces by0.000101nats/event and beats three activity-matched random schedules (mean gap0.000205). Online controller weight updates lose to frozen guidance. These are one controller initialization/reusedDEV development results, not general confirmation of a particular latent market structure.

The credit policy is not a fixed numeric rule just because its learned parameters are fixed. Let

    h_t = F_phi(h_previous, elapsed_time, observed_features),
    a_t = G_phi(h_t, current_actor_state, observed_errors, local_credit),
    u_t = Optimizer(theta_t, g_own_t + a_t R_t, moments_t).

The effective bias on actor updates changes causally with h_t and context. Parameter learning changes F/G; numerical inference changes their inputs/state. Neither requires the data distribution or preferred credit allocation to remain stationary. This is a general stateful-function principle, not a new theorem exclusive to this family.

The native construction implements that principle using addressed temporal races and persistent memory rather than a GRU. Its utility teacher compares actual own/projected Adam proposals against subsequent observed recording-cell losses. The fitted policy therefore targets the usefulness of transported credit for a future objective, instead of treating a historical full gradient as the only possible target.

The random audit uses the same total57remote adjustments over512windows. It supports a timing contribution in this episode: the acquired controller beats all three random schedules while using contextual/native state. It does not isolate which representation, memory or clock mechanism causes that contribution. A static hand-designed schedule or an ablation could expose other explanations; declare those as specific future comparisons rather than identifying a hidden market symmetry from a scalar score.

Online parameter updates can still help in other conditions; here they make the current policy worse. Distinguish numerical adaptation, parameter adaptation, derivative fidelity and future utility. A single-window teacher has limited horizon, tiny labels and a changing normalization scale. Those are concrete targets for investigation, not proof that frozen credit functions are universally preferable.

Finally, activity in the optimizer and activity in the sensitivity computation differ. Choosing a_t=0 after R_t is computed does not save remote transport. A true economical policy must request its bank before the window from cheaper information, or pay for replay/exploration. Preserve the stronger acquired-frozen policy and the saved standalone own/BPTT resource boundary when engineering that schedule.
