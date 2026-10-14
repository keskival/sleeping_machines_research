# Guarded execution, recovery and evidence

## 1. Preflight

Read HANDOFF/product orders; inspect queues, completed results, active processes, host lock, memory and (when permitted) GPU occupancy. Reuse comparable saved references. Confirm battle/decision, owner, pinned source/data hashes, unique run tag, resource profile and cost envelope. Running work is not cancelled merely to reorder a queue.

The current runner expects /workspace and /workspace/.venv-docker/bin/python. Bootstrap that layout or implement and gate a portable runner before using another path. Do not bypass run_safe with direct benchmark Python. New hosts default to one training job; the named AWS gym's slot exception is not portable authorization.

## 2. Create one immutable job

Create experiments/queue/<battle>/<unique-run>/<unique-run>.txt containing one job line:

    <unique-run> <approved-driver.py> --tag <unique-run> <registered-arguments>

Record queue/source/data hashes, dependency gates, timeout, RSS/virtual-memory caps, available-memory floor and result destination. Validate arguments against the selected driver's current help/source; prototype drivers may have no completed gate. No automatic fit follows a prototype simply because code exists.

Existing AWS addenda must bind the parent manifest and use the bounded coordinator. New workers invoke the guarded runner directly for their one admitted job. Set all resource variables explicitly, with values from the representative smoke. Example invocation structure (replace the measured placeholders first):

```bash
MIN_AVAIL_MB=<worker-reserve-MiB> \
MEM_CAP_RSS_KB=<whole-job-RSS-cap-KiB> \
MEM_CAP_KB=<measured-virtual-cap-KiB> \
JOB_TIMEOUT_S=<measured-timeout-seconds> \
bash experiments/queue/run_safe.sh <one-job-queue.txt>
```

This is a template, not an executable command with literal placeholders. Preserve existing shared-host floor values. Run long jobs under tmux or a durable service/controller and retain exit status, command, runner logs and heartbeats. Timeout derives from measured whole-fit throughput and an explicit allowance for evaluation/checkpoints, not a fixed context/model ceiling.

## 3. Checkpoint the actual learner

Checkpoint model tensors, numerical lane state, optimizer moments/local steps, bank versions, allocator/critic state where used, every private/global RNG, shard/cursor, acquired-target count, cumulative work/time and source/data identity. Include reset/update schedule and stream layout. Best weights alone support inference; they are not an exact learner continuation.

Write to a temporary path, flush/close and atomically rename on the same filesystem. Upload an immutable object and checksum, then publish its manifest last. Test interrupted/resumed parity at the intended policy before a long preemptible fit. For versioned bank execution checkpoint at a declared quiescent cut or serialize all outstanding producer/read obligations consistently; arbitrary mixed-bank capture is not a complete learner cut.

## 4. Handle interruption

Preserve logs/partial records. Confirm the old process exited and release its host lock before replacement. Fetch the latest verified complete cut; restore every learner/RNG/cursor field, validate identity and account for repeated/discarded work. A source/settings change creates a new run tag and is a changed experiment, not exact resume. Missing final work remains explicitly unknown rather than silently zero.

Provisioning/orchestration retries do not make the training program resumable. Preemptible capacity is appropriate only after complete recovery is implemented; compare its effective cost including interruptions with reliable capacity.

## 5. Publish and decide

Require completed status and verified artifacts. Score selected weights using the registered development/reference protocol; keep public test selection sealed. Publish quality, targets/passes, capacity/activity, complete fitting work, per-target work and inference work, with hardware/dtype scope. Aggregate only identical scoring protocols.

Update current report/handoff and push main. Serialize multi-host publication; workers should upload evidence rather than independently regenerate/commit overlapping reports. Use immutable object names and compare-and-set/idempotent run IDs to prevent duplicates. Never overwrite successful results or silently use changed settings under a prior tag.

Withdraw waiting non-decisive work and stop its admission watchers. The legacy AWS coordinator caches pending jobs; deleting their immutable packet/queue makes its hash preflight reject them before training, but records errors. A future orchestrator must have explicit durable cancelled state and check it immediately before launch. Preserve completed evidence and identify withdrawals clearly; do not mistake them for research failures.
