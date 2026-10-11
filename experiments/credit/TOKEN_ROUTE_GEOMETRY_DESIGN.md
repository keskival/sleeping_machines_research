# R1 route geometry development rationale

The fixed v1 continuous policy fit completes64 updates/512 TRAIN targets per
arm. Exact fresh-prefix reference-over-zero gain2.239e-7 nats/target has95%
interval[-8.574e-7,9.034e-7]; reference-over-static is-3.328e-8 with interval
crossing zero. This variant has no confirmed utility advantage. Exhaustive
best-route headroom is.00152–.01068 nats/target. Source carrier and candidate
pool have useful responses; raw-input router learning barely changes policy.

Proposed engineering change: add the native forward model's causal immediate
feature mean shift to the shared route descriptor. For one addressed logit
perturbation beta and native full probability p, route probability q gives

    delta gradient_z CE = (q_r-p_r)(e_r-mu_except_r).

The target embedding cancels, so this update-difference feature needs no future
label. It exposes an existing pretrained representation and route interaction
rather than asking a small newly initialized scorer to infer products from a
short stream. Normalize vector blocks, logarithmically encode age, and retain
perturbation beta/probability-shift strength. Reference learning and exact
factual residual correction remain unchanged. Separate keys/values, temporal
carrier, hard writes, persistent numerical learner/optimizer state retained.

The scalar mean shift is a known softmax derivative identity, not the novelty
claim. Its role in the mutually teaching temporal route/credit system is the
proposed integration. Full normalized vocabulary mean computation adds paid
O(Vd) work; no sparse discovery/output or efficiency claim follows.

Required comparison: fixed v2 same source/data/seed/64updates/three arms and
exact private-prefix scoring; geometry added only to causal issued inputs.
Record new actor parameter count, full work and first observed outcomes.
Gate exact independent CE gradient differences at actual vocabulary, then
bounded driver smoke before fit. Keep completed v1 evidence; no benchmark
verdict or blind tuning. Both development rounds share disclosed private
prefixes; an eventual confirmation requires fresh unseen prefixes/seeds.
