# Model and data scaling

## 1. Register the useful decision

Read current orders and completed comparable results before choosing a run. Name the battle, owner, mechanism, strongest saved comparator and decision changed by success/failure. Record acquired targets, optimizer updates, unique data, presentation order, seed, tokenizer, development selection and scored history/reset policy. For language, keep public validation sealed until development selection under the pinned reference protocol.

Do not queue an architecture grid. Select the next useful model/acquisition point from completed evidence. A resource smoke checks correctness and capacity; a small-data loss alone does not veto a registered larger acquisition. Any admission overrides belong in current product orders, not inferred from this runbook.

## 2. Select scale axes explicitly

- Depth: actual temporal stages with final-stage prediction; diagnose useful information/credit, not just nonzero gradients.
- Width/readout: larger payload and normalized token prediction capacity; preserve the parent when testing a growth repair.
- Bank capacity: grow available receivers while recording selected writes, scored keys, value deliveries and alternative learning work separately.
- Learning horizon: distinguish carried numerical memory, factual gradient window, route consequence horizon and consequences across optimizer updates.
- Data: increase fresh acquired targets with streaming shards; keep tokenizer and scored-target protocol fixed.

Change the axis needed by the hypothesis. A combined repair is valid when required by integration, but must disclose the changed axes and avoid attributing the result to one component. Compare growth with ordinary continued fitting of the saved parent when extra acquisition could explain the gain.

## 3. Estimate and measure memory

Build a ledger for parameters, actual optimizer tensors, persistent state per lane, candidate/key intermediates, retained activations, replay/snapshot worlds, decoder intermediates and data cache. Inspect dtypes and storage aliases; count independent allocations, not just tensor names. For example, float32 parameters plus two float32 Adam moments alone consume about12bytes/parameter, excluding gradients, steps, states and temporary allocations. Actual implementations may allocate additional copies.

Run one bounded gate at the intended shape: depth/width/pool, lane count, credit window, replay admission, vocabulary and checkpoint writing. Cover optimizer-state initialization and evaluation/checkpoint peaks. Sample whole-process-group RSS and available memory; process maxRSS alone misses child totals and host page-cache pressure. Record CPU model, physical cores/SMT, library versions, thread settings and local storage. The existing watchdog samples process-group RSS; extremely short peaks require additional measurement or a suitable cgroup limit.

Choose RSS cap above the measured representative peak with explicit uncertainty headroom. Reserve memory for OS/runtime/services separately. A4GiB worker is only a candidate if the intended larger workload fits; a1GiB small smoke does not size a32M/large-width fit.

## 4. Scale data without loading it all

Use immutable ordered token shards with checksums and an explicit split manifest. Stream/memory-map a bounded local working set; prefetch the next shard before exhaustion. Record shard, lane offsets, document/EOS policy, cursor and consumed target count in checkpoints. The current drivers accept a single shard; arbitrary multi-shard continuation is engineering to implement and gate, not a capability this runbook assumes exists.

Across shard changes, preserve each lane's true contiguous causal history or reset at a declared discontinuity. Never carry state between unrelated stream positions. Fit frequency priors/token maps only from declared TRAIN data. Do not silently refit them during continuation; version changes can invalidate saved readout ranks and checkpoint meaning.

## 5. Gate, acquire, evaluate

Numerical gate: causal inputs, normalized full-vocabulary scoring, retained route/time/state mechanisms, growth parity where applicable, finite updates and checkpoint continuation. Resource gate: intended shape and complete teacher/optimizer costs. Then admit the registered acquisition through EXECUTION.md.

Save a bounded set of complete acquisition-stage checkpoints, not only best weights. Measure quality against strongest compatible saved members on identical development targets; report whole fitting work and per-target work in the same units. Include interrupted/repeated acquisition and evaluation costs explicitly. Distinguish measured CPU time, FLOPs/traffic estimates and physical energy. No pending score, MSE improvement or numerical gate substitutes for competitive quality.

Advance a successful quality/work frontier point. For a loss, diagnose protocol, representation, credit, optimization and acquisition before choosing one next repair. Do not automatically double the budget.
