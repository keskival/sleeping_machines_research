# Realize the state path that a credit target describes

13 October2026, B10. Follows forward-taught state-interface credit and optimizer-aware route credit.

Let an observed producer block take a declared incoming state q to s=P_theta(q,x). Its subsequent future block has loss L_theta(s,y). For a clock parameter direction a_j, write E_j=partial P_theta/partial a_j. Forward probes teach b_j=E_j^T gradient_s L_theta. The complete parameter derivative has two terms:

    derivative_a L_theta(P_theta(q,x),y)
      = b + partial_a L_theta(s,y) holding s fixed.

The trained b is only the producer-state component. If the deployed update changes theta but carries the old s unchanged, its immediate future loss follows L_(theta+delta)(s,y): b is absent from that immediate first-order path. Blindly treating b as the exact next-window gradient would misstate the intervention.

A causal refresh uses already observed x to set s_new=P_(theta+delta)(q,x), then forecasts y. This realizes the producer-state term locally. Incoming q is declared conditioned numerical context: older state/version effects are not reconstructed. Exact clip/Adam produces delta from the admitted gradient; b plus the current own-leg gradient is not the exact derivative of the actual future objective, because future direct-parameter credit, clipping, moments and cross-leg current-gradient approximations also matter. Actual proposal losses are the integration test.

Refresh is not free inference. Eight observed events replayed after each update require their write/read/normalization and addressed-state work, and their refreshed forecasts never replace the original scored forecasts. A local state-interface predictor could later approximate this refresh from forward-derived sensitivities, but all updated parameter directions and nonlinear errors need coverage; the existing four clock columns cannot correct arbitrary dense-Adam updates.

Current protocol fixes producer8/future8, actual clip1/Adam.0003 and additive b coefficient1. Own-carry measures existing detached-state updates, own-refresh isolates refresh, learned-refresh isolates the acquired b. Frozen native critic parameters and causal persistent private state retain native computation. Source TRAIN teaching and full deployment work are charged; no future labels are used to choose deployment updates. A new parameter version resets current gradient traces. The learning cut still includes global clipping and common update boundaries.

Short2048event development stage: learned-refresh5.703270448, own-refresh5.703284229, own-carry5.703336053. Learned credit gains0.000013782aggregate against matched refresh, but its immediate future-only NLL is0.000003030higher. This motivates the unchanged16384event continuation, not a claim of material forecasting advantage. Different update schedule/source acquisition from the established32target program; these NLLs cannot replace its5.737391headline. Full teacher/preparation amortization and depth/size remain the practical objective.

Completed16384DEV targets: one-pass learned-refresh5.756599087 wins against own-refresh5.756963281, own-carry5.756862885, TRAIN-mean5.756874830, signed-native5.756818872, exact8event BPTT-carry5.758146736 and BPTT-refresh5.758367769. Own eligibility accounts for most of the BP difference; learned credit adds0.000364194nats/event over matched own-refresh. One critic/actor initialization, new8+8protocol, reusedDEV, no sealedTEST.

Executed512prefix work (includes127warmup events): BPTT-carry56.380MFLOPs, BPTT-refresh66.060, own-carry77.605, own-refresh87.284, learned-refresh88.999. All forecasts reproduce exactly. Nonlinear/Python/random work and uncovered native-history scalar operations are separate; logical payload is not physical traffic. Quality wins permit this extra work under the user13October direction; it is not a complete-work win. Shared source8192teacher targets/original actor fit additionally paid.

Fourpasses over the same128TRAINreceipts lower fixed-actor credit MSE0.001133650→0.001058803, but actual-update NLL worsens5.756599→5.756772. Retain one-pass for current actor utility. This is a concrete reason to train the update objective directly. For output f_psi, scale D, coordinate embedding P, actual clipped Adam A and refreshed producer S, define theta'=A(theta,m,v,g_own+P D f_psi(h)) and J=L_theta'(S_theta'(q,x),y). Its differentiable factual derivative is

    gradient_psi J = J_f^T D P^T J_(A,g)^T
                    [partial_theta' L + J_(S,theta')^T gradient_s' L].

This includes both direct future-parameter and refreshed-state effects, plus optimizer/clipping sensitivity. Hard-route policy derivatives or alternative effects require their own declared estimator; a fixed factual chain does not label them. Finite actual-update world probes can teach the same local output derivative; differentiable audited Adam/refresh can supply a paid training teacher. Neither method requires copying ordinary factual BP gradients as critic ground truth. Stored native decision activations train the credit model from arrived future losses; the resulting model proposes the next actor update. Numerical contracts must compare actual clipped Adam moments/parameters and full future proposals before training this next variant.
