# Best completed 90M checkpoint: source-exact sparse validation

**PREPARED UNRUN.** This is deployment validation, not training or a new score.
Parent: completed p64/D4/H2/pool2, T256 test1.857306269 BPC on the restricted
test[95M:96M] window. The actual checkpoint and original producer aliases are
preserved; current changed numerical sources are not imported.

Read [the recovery theory and contracts](../../theory/153_source_exact_sparse_deployment_recovery.md)
and [the frozen manifest](manifest.json). Existing official MG, primate data/
confirmation, depth8 replay/teacher and Curie queues retain priority. No host
reservation, scheduler, waiter or numerical job was started from this workspace.

After an actual AWS physical-host admission, the first stage is:

```bash
MEM_CAP_KB=3000000 MEM_CAP_RSS_KB=1250000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=420 \
  bash experiments/queue/run_safe.sh \
  experiments/queue/aws_best90m_sparse_recovery_20261004T190000Z/aws_best90m_sparse_recovery_20261004T190000Z_contracts.txt
```

This uses the ordinary one-job host lock. For a bounded AWS slot, the existing
coordinator must pass its ordinary reservation and slot descriptor; do not
invent those environment values or launch a fourth job. The adapter verifies
the inherited lock, physical hostname, one-thread settings, address-space cap,
RSS watchdog setting, available-memory floor and timeout before any numerical
child. A free Docker-local lock is insufficient.

Then run the two prefix queues serially, with the same caps and600s timeout;
after both pass, full DEV/TEST at their corresponding T128/T256 lengths with
the declared21600s ceilings. Reassess actual prefix memory/wall time before
full admission. Inspect [manifest.json](manifest.json) for every filename,
prerequisite and guard. Changing caps/settings requires a fresh tagged protocol.

Passed canonical results and recovery sidecars are mandatory for downstream
admission. The inner sources/manifests run under an isolated `.git` root;
published numerical records keep their original arguments and byte hashes,
and sidecars map them to the new canonical recovery outputs. Missing/failed
predecessors reject before model imports. The new `_original` queues are
provenance for the inner program; use these wrapper queues to run it.

The source-only archive is about200KiB; it contains no trained weights or
dataset. These remain local and are checked against their declared hashes.
The new dataset hash does not retrospectively establish the original training
file's identity. Native parity, saved-quality reproduction and measured
serving advantage are all pending. Diagnostic wall time is not a serving
speed measurement.
