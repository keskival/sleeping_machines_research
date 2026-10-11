# Native parameter credit taught by future likelihood

B1/Curie, 11 October 2026. The implemented connected credit function updates native temporal-layer and input-producer parameters. Its teacher is actual next-block likelihood after the finite intervention, not an exact-gradient regression target.

## Declared causal objective

Let H_t contain the observed current forward packet, terminal supervised error and numerical actor/optimizer state. A connected function proposes u_ψ(H_t), with a declared norm bound, and issues θ' = θ − η u_ψ. Exact local heads may have their own optimizer update, held fixed relative to this issued ψ derivative. The next block is forecast before its feedback is used. Its local teaching gradient is

    ∂ℓ_future(θ')/∂ψ = −η (∂u_ψ/∂ψ)ᵀ ∇_θ' ℓ_future,

with the complete derivative of any norm cap included. Incoming actor and head-optimizer state are detached: this is the derivative of the issued local finite intervention, not the full historical trajectory. Function activations and its parameter version must remain available until feedback arrives. Updating ψ before consuming that graph would change the meaning of the derivative.

A Bernoulli teacher I with predictable probability p > 0 uses I/p times this raw derivative. Its mean is the declared local future-loss gradient when teacher selection does not change numerical prediction or the issued weights. A sampled non-teacher executes the same numerical weights with detached derivative state. Clipping, Adam and later changing actor states do not inherit an unbiased-optimizer theorem from this raw mean identity. All actual teacher work is charged.

## What future teaching can learn

The optimal update depends on the conditional future distribution, not only the current sample error:

    u*(H) = argmin_{||u||≤r} E[ℓ(θ−ηu;Y_next) | H].

A local expansion gives a conditional future gradient and Hessian. If the conditional Hessian is positive definite, the unconstrained quadratic optimum is its inverse applied to the conditional gradient, with the appropriate step scaling. With indefinite curvature, the bounded problem can favour boundary directions; a PSD noise-allocation theorem cannot simply be applied. Native future-loss audit results already illustrate this distinction.

Connected cross-layer outputs can represent coordinated updates and data-dependent priors. They need not reproduce the current backpropagated gradient. Generalization depends on whether the forward packet carries sufficient predictive information, whether the function learns that structure and how the distribution evolves. Chain-rule reciprocity supplies a teaching signal; it is not a proof of temporal reversibility, grokking or favorable scaling.

## Measured progression

The short32-acquisition/128-reuse native fit loses quality to SGD at depths4/8 while saving CPU. At128/512 acquisition with512reuse blocks, continued credit improves final DEV quality versus frozen credit at both depths. Depth4 gives better quality and lower CPU work than the Python-SGD control, but an optimized native comparison is necessary.

Using the parity-validated compiled scan and exact local AdamW head updates, seed0 depth8 finalDEV NLL is−0.369780 versus nativeBP-AdamW−0.337109, whole learning12.746 versus13.129s. Depth4 favours BP-AdamW. Prequential training loss favours BP-AdamW at both depths. These are distinct endpoints: the result is a single-initialization final development-quality signal, not a full-stream efficiency/scaling result. Two prospectively fixed initialization repeats assess whether it survives seed variation. Only a32-sequence TaxiDEV slice is used, no official TEST.

## Economy and the next representation test

The current critic has144,704/276,752 parameters versus3,056/5,056 native actor parameters atdepth4/8. It proposes every2,112/4,112 core parameter each block. Dense critic decoding, meta-Adam and dense parameter writes are real costs, not sparse hardware activity. The approximate cost condition for amortized benefit is

    A + N_learn (C_forward + C_head + C_credit + p C_teacher + C_optimizer)
       < N_ref (C_forward + C_backward + C_optimizer_ref),

at the same quality target. A is acquired credit investment. The required target counts N_learn/N_ref depend on actual learning quality; replacing them by equal steps conceals bias/generalization effects. Use measured backend costs, not a slower reference implementation.

If the native advantage repeats, test a structured connected parameter field before increasing width blindly. A rank-r matrix update uses factors with r(rows+columns) outputs instead of rows×columns independent outputs. It can reduce decoder/optimizer size while retaining a globally connected context. Zero-initializing both factors blocks first derivatives; initialize one nonzero basis and the other at zero, then let both learn. A sum of successive low-rank updates need not permanently constrain the rank of the numerical actor weights. Representation bias and final quality are measured costs of this compression.

Such factorization alone still writes dense actor matrices. True clockless economy additionally needs addressed active parameter blocks or lazy factor application, recorded selection costs and learned credit to unrealized alternatives. Retain native time computation, deep memory, key/value separation, internal hard-route/counterfactual and silence objectives when integrating that next stage. Current native dense writes are a documented gap, not a replacement for the full substrate.
