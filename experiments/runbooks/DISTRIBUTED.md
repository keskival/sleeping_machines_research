# Experiment fleets and distributed asynchronous learning

## First distinguish the two scaling problems

Independent experiments need provisioning, admission, scheduling, artifact storage and recovery. A single distributed learner also needs parameter/state ownership, transport, consistency and useful scaling. Running many independent fits is not evidence of asynchronous model training.

Start with a durable priority queue and one worker per admitted experiment. Parallelism serves registered decisions: larger useful acquisitions, seed confirmation or necessary causal comparisons. Do not spend funding on an uncontrolled architecture grid. Prototype cloud adapters only when they remove a measured execution bottleneck.

## 1. Independent experiment fleet

Represent jobs as immutable manifests and durable states: proposed, admitted, provisioning, running, checkpointed, completed, failed, cancelled. The controller holds a lease for each run; heartbeat expiry triggers inspection/recovery, not simultaneous duplicate training. Create idempotent resource tags and upload immutable artifacts. Admission names battle, decision, full budget and predecessors; cancellation is checked immediately before launch.

AWS Batch is an option for an AWS-only fleet. SkyPilot is an option for supported providers and managed recovery. Direct provider APIs are adequate initially for a small European CPU fleet. Evaluate integration/operating cost before adopting Kubernetes. Ray can orchestrate tasks/actors, but need not be introduced solely to launch ordinary independent scripts.

## 2. Establish single-node useful locality

Current building blocks: receiver-local AdamW, per-bank radial clipping, inbox completion sets, immutable snapshot dispatch prototype, bounded forward consequence teaching and exact-mean controlled route credit primitive. Consult TRAINING_LOCALITY_EXECUTION.md for completed versus withdrawn/pending gates. These are not a completed distributed trainer.

Choose an integrated learner with useful prediction and complete work. Eliminate avoidable global reductions and unused operations while retaining normalized scoring and full counterfactual/shared credit accounting. A local raw-gradient cap does not bound Adam's actual update. No blanket substitution of coordinatewise saturation.

## 3. Define ownership and consistency before networking

Assign each receiver bank to one owner. Canonical owners handle tied/head/interface parameters. Credit packets identify bank version, actor/state cut, event, producer and horizon. Declare all required producers, reject stale/duplicate credit and close zero/absent contributions correctly. A ready bank need not wait for unrelated banks, but downstream credit causally required by that bank cannot be discarded.

Forward events must read a declared consistent snapshot/version policy. Do not mutate tensors still used by backward. Whole-model snapshots are a correctness reference with paid copying/storage; economical execution needs restricted read sets, copy-on-write/version retention or contracted native kernels. Memory state is also versioned/owned; parameter tags alone do not make mixed numerical state meaningful.

Classify staleness explicitly: exact issued-version replay, bounded stale updates under a tested rule, or changed learner semantics. No unversioned parameter-server shortcut is presumed equivalent. Preserve clocks, sparse selection, private memory, keys/values and optional-route learning work.

## 4. Gate a distributed implementation

Use one-node and two-node exact reference tests for routes, clocks, state, packet sums, optimizer moments and recovery. Inject reordering, delay, duplicates, worker loss and interrupted checkpoints. Verify no duplicate updates and no old reader observes overwritten storage. Account for outstanding packets/read versions at checkpoint cuts.

Then run the same intended learner and acquisition on1/2/4nodes only where resource measurements justify it. Record quality trajectory, targets/second, total node-hours, packet bytes/messages, snapshot/queue storage, stalls, retry work and wall time. Charge key discovery, losing-value credit and shared decoder/interface traffic. Keep a single-node reference; changed update grouping/staleness is a changed training rule.

## 5. Scale only the useful configuration

Use one region and an appropriate low-latency network for tightly coupled fits; independent fits can span providers. Select placement from measured communication demand, not a presumed advantage of sparse messages. Many tiny messages may be dominated by dispatch/network overhead; test batching without changing event semantics.

Adopt distributed execution when it reduces time or complete cost to useful quality under the declared protocol. More nodes with worse quality or greater paid work are not an efficiency win. Learned compact credit/shared horizon predictors can reduce trace/communication demand only after useful-update and complete-work evidence. GPUs remain outside current language orders; cloud orchestration does not override that policy.
