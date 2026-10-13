# Actual optimizer geometry for coupled future credit

13 October2026, R1/B1 shared integration. Connects predictable route residuals with Curie's measured difference between better credit MSE and better clipped-Adam learning. Standard Adam differentiation is the primitive; the contribution is its explicit place and economical boundary in the native future-credit loop.

Let g be the complete issued gradient at one declared parameter/moment version. Clip it as z=k g with k=min(1,R/(r+e_c)), r=||g||. The actual implementation's e_c=1e-6 matters. Away from the clipping boundary,

    J_clip = I                           if r+e_c < R;
    J_clip = k[I − g g^T/(r(r+e_c))]     otherwise.

The familiar tangent projection with r² in the denominator assumes e_c=0. Under the actual stabilizer the radial eigenvalue is k e_c/(r+e_c), not exactly zero. At g=0 the locally inactive clip has derivative I.

At step t, m'=beta1 m+(1−beta1)z and v'=beta2 v+(1−beta2)z². Define a=1−beta1^t, b=1−beta2^t, h=m'/a, u=sqrt(v'/b), d=u+e_A. AdamW's parameter proposal is delta=−eta(h/d+decay theta). At fixed theta/m/v, its z derivative is diagonal:

    D_i = −eta[(1−beta1)/(a d_i)
          − h_i(1−beta2)z_i/(b u_i d_i²)].

The weight-decay term has zero derivative with respect to g at fixed theta. For a consistent never-updated coordinate h=u=0, the composite derivative is −eta(1−beta1)/(a e_A); do not evaluate a spurious0/0 by differentiating sqrt alone. An inconsistent h!=0,u=0 is singular.

Let lambda_theta, lambda_m and lambda_v be future-loss cotangents at the post-update learner cut. The gradient-space future demand is

    lambda_g = J_clip^T [D lambda_theta
                 + (1−beta1)lambda_m
                 + 2(1−beta2)z lambda_v].

This carries future sensitivity through actual parameter and optimizer-memory updates without constructing an n×n matrix. D and moment contractions are coordinate-local. The clipping correction needs one complete-vector inner product in addition to the existing norm; it is a real synchronization/communication cost. Independent bank clipping would change this boundary and requires a quality comparison, not a claim of equivalence to global clipping.

## The missing target alignment

A state-only producer credit and a held-state parameter update have different realized paths. If a proposal refreshes state s'=P_(theta+delta)(q,x), its endpoint parameter demand adds (partial_theta P)^T lambda_s to the direct future parameter demand. It must also include all admitted directions, cross-layer writes and relevant older learner-state paths. A fixed-version neuronal cotangent alone does not represent the whole future learner.

If credit error e perturbs g, its first-order parameter-update error is D J_clip e; its complete learner error additionally contains the moment/state terms. For a chosen future-loss curvature/information metric H on the post-update cut, the relevant quadratic error is

    e^T J_update^T H J_update e.

Raw coordinate MSE ignores this metric. Predictors with lower MSE can have larger errors in high-impact directions or change moments/clipping in ways that harm future learning. The Curie label-reuse result makes this distinction an engineering priority rather than an abstract qualification.

A forward directional probe of the actual update/refresh map supplies the corresponding scalar future-loss response. The learned backward estimator predicts this response or the endpoint cotangent; the known local optimizer transport maps it into the current credit coordinates. Labels remain actual future consequences. The actor and critic can use the existing backward call for their chosen supervised/residual losses; there is no mathematical need for an infinite sequence of trainers. Delayed labels require the exact proposal/version/context to be retained or replayed and all work charged.

## Implementation and integration boundary

`sleeping_machines/learning/adam_credit_transport.py` implements the matrix-free cotangent including bias correction, clip stabilizer, moment demands and untouched-coordinate limit. `experiments/credit/check_adam_credit_transport.py` compares it with dense autograd of the actual fixed-version proposal for4/17coordinates, clipped/unclipped gradients and steps1/13. This R1 contract does not train another critic or displace the running2M→8M acquisition and residual-credit pipelines.

Next use the operator to construct update-aligned receipt metrics or probe demands inside a selected integrated model. Charge its O(n) elementwise work and global contraction; compare actual future loss rather than declaring an advantage from operator rank alone. Nonstationary weights/moments and new states change the conditioned cut and therefore the target. The inexpensive known optimizer transport is reusable; the future cotangent still needs learning and causal consequence labels.
