# B1/R1 route credit including subsequent learning

Failure: frozen-actor response algebra omits parameter/optimizer changes caused by interventions. Retain depth4 timed/keyed state, keys/values and optional hard addressed key messages. Begin with controlled mark-readout adaptation only: each forced branch owns its288 readout parameters and Adam moments, initially identical. Carrier, clock, query and writes stay fixed. Score each revealed suffix target before updating its branch readout. Full revealed-prefix replay is causal and charged; fixed carriers are numerically unchanged by readout updates. This is not full deep actor learning or clockless sparse execution.

Target: mean prequential NLL(intervention)-mean prequential NLL(no-op), including four subsequent Adam updates. Critic stays fixed during each continuation. Sum branch losses for backward so independent Adam coordinates have their single-world gradients; batch means would change learning-rate/epsilon semantics. Contracts compare zero-learning-rate replay to the fixed formula, batched versus independent Adam trajectories/final weights, identical no-op worlds and first-prediction independence from later suffixes.

Next fitting gate: freeze the confirmed shared-prefix representation/baseline, train identical zero-initialized residual heads on either frozen-path response or learning-aware utility. Same action grid, causal packets, fitting steps and head capacity; target differs deliberately. TRAIN receivers0–5, never labelled6/7 evaluated on fresh prefixes. Fit32 prefixes×4 continuations, assess16×8,256 steps. Include all teacher, replay, optimizer, baseline pretraining and query work; retain source-bound checkpoints. The total residual function lacks the old kernel derivative/curvature certificate, so actions stay within the discrete bounded grid. A first variant informs development rather than a family verdict. No external fits or TEST.

The pilot fixes readout Adam learning rate0.05 before DEV evaluation (contract witness0.01). Saved shared-prefix weights/representation remain frozen; each equal-capacity correction has2,817 parameters and zero initial output. The correction’s target is total future-loss change after learning, not a copied gradient. Source-bound smoke passes; native readout/replay costs are measured in teacher work.

## Direct state influence and learning-mediated influence separate

Assume this controlled readout experiment: branch carrier states s_k,s'_k depend on the optional write but not the readout weights. Let theta_k,theta'_k be their adapting readouts and theta0 their common initial readout. For the same observed target, define D_k=ell(theta'_k,s'_k)-ell(theta_k,s_k), and D_Fk=ell(theta0,s'_k)-ell(theta0,s_k). Insert ell(theta_k,s'_k) to obtain the exact decomposition

    D_k-D_Fk = ell(theta'_k,s'_k)-ell(theta_k,s'_k)
              + [ell(theta_k,s'_k)-ell(theta_k,s_k)]
              - [ell(theta0,s'_k)-ell(theta0,s_k)].

The first term is branch-specific learning drift. The other two describe how even the common learning trajectory changes the response to the same write. If loss is L_theta-Lipschitz in readout parameters and its mixed theta/state derivative has norm at most M along the relevant segments, the fundamental theorem of calculus gives

    |D_k-D_Fk| <= L_theta||theta'_k-theta_k||
               + M||theta_k-theta0|| ||s'_k-s_k||.

These are conditional smoothness assumptions, not constants inferred from held-out MSE. Average the bound over the declared future horizon. The fixed response is exact at zero learning; at nonzero learning its error depends both on differential branch learning and on common forward adaptation. In a general actor, interventions also change carriers/transitions; additional state/transport terms are needed.

Let X_k contain the readout and optimizer moments, with X'_0=X_0. A local Lipschitz update law can yield

    ||X'_(k+1)-X_(k+1)|| <= kappa_k||X'_k-X_k||
                           + b_k||s'_k-s_k||.

Unrolling gives a finite sum of carrier perturbations transported through products of update sensitivities. It requires no contraction for a finite horizon, but growing products can destroy economy. Small or decaying physical key differences alone do not imply small optimizer/parameter differences. Independent Adam coordinates are therefore part of the replay state, not an optional annotation.

**A route can matter after its state vanishes.** Initially theta0=0. A temporary write delta affects the first loss ell1=(theta-delta)^2/2; SGD gives theta1=eta*delta. The write is then cleared. A later loss ell2=(theta-1)^2/2 differs from the no-write branch by -eta*delta+eta^2*delta^2/2. The direct later state effect is zero, but the learning-mediated effect survives. This is an exact scalar example, not a native performance result. It establishes why an event-native credit process may need memory of learning consequences longer than memory of the original message.

The reusable forward evidence should accordingly include which parameter/optimizer versions were changed, the causal producer, action/update policy and retained learning-state summaries. A learned critic can compress that information; exact total-gradient or intervention audits still require sufficient replay/transport. An arbitrary previous backward activation is not automatically a sufficient learning-history statistic.
