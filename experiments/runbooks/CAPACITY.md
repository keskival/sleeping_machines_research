# API-managed capacity selection and provisioning

## Objective and workload profiles

Minimize total cost to complete the registered quality/acquisition decision. Cloud brand, advertised vCPU count and hourly price alone do not determine that cost.

Use measured profiles, not fixed small-model limits:
- Development: one short integrated correctness/resource gate.
- Sustained acquisition: reliable CPU throughput, sufficient learner RAM, local shard cache and resumable storage.
- Larger model/credit: increased optimizer/state/activation/replay memory and measured CPU demand.
- Distributed learner: adequate per-node RAM and measured interconnect requirements after DISTRIBUTED.md gates.

Current language is CPU-only. GPU capacity becomes relevant only after the user changes that policy and useful implementation throughput is measured. x86 is the initial portable target; ARM needs package, numerical and throughput verification before claiming equivalent economics.

## 1. Build a fresh offer shortlist

Collect current offers through authenticated provider catalogs where available. Record provider, region, architecture, CPU allocation (dedicated/shared), RAM, local disk, storage charges, traffic/IPv4 charges, billing interval/cap, setup fees, capacity/quota and provisioning delay. Keep a timestamp; old marketing prices are not procurement evidence.

First shortlist Hetzner Cloud dedicated CPU plans and dedicated-server offers, plus a comparable AWS option. Add another European provider only if its actual quote/specifications could improve the choice. Dedicated-server management/ordering and elastic VM provisioning are different APIs; verify ordering permissions, rescue/reimage workflow, cancellation and availability before relying on automation.

Official references:
- Hetzner Cloud API: https://docs.hetzner.com/cloud/api/getting-started/using-api/
- Hetzner dedicated-server Robot API: https://robot.hetzner.com/doc/webservice/en.html
- CPU allocation semantics: https://docs.hetzner.com/cloud/servers/faq/
- OVHcloud public-cloud price catalog: https://www.ovhcloud.com/en/public-cloud/prices/

A shortlist is not a claim that one provider is cheapest. Prices, capacity and single-thread performance require current verification.

## 2. Benchmark the intended native workload

Only purchase provisioned capacity within an explicit spending envelope. Run the same source-bound resource gate on a small shortlist: same architecture, native model shape, token vocabulary, update/teacher policy, precision and targets. Do not launch external-model training for comparison.

Measure useful targets/second, complete RSS, CPU contention/steal, storage throughput, initialization and checkpoint time. For multithreaded execution, measure one job's scaling; unused cores are not useful capacity. On a shared VM, sustained measurements matter more than a short burst. No multi-job packing on new hosts without a separately authorized bounded scheduler.

Compute cost per million successfully acquired targets, including billing/setup amortization, storage, data transfer, downtime and replay after interruption. For fixed monthly capacity include idle hours and expected utilization. Fit projection uses measured throughput at the intended shape, with uncertainty and checkpoint/evaluation overhead. A cheaper VM that finishes materially slower can cost more.

## 3. Provision a worker

Create a provider adapter with create/inspect/tag/delete operations and idempotent resource IDs. Supply cloud-init or an equivalent bootstrap for pinned OS, container/runtime, immutable source revision and dependencies, local cache, logs and checkpoint upload. Do not bake credentials into images or result manifests.

Required labels: project, battle, run ID, owner, expiry, cost envelope and worker profile. Enforce account-level worker count and spending bounds; count creating and stopping workers too. Use a durable controller so client termination does not orphan spend. Reconcile actual resources against desired jobs and delete idle workers after uploading verified artifacts.

Use API/SSH-based provisioning with the guarded queue initially. Kubernetes, SkyPilot or Ray are optional adapters, not prerequisites. Native provider support must be verified; do not assume a framework supports every European provider. Store data once per active region and cache locally to avoid repeated full downloads.

## 4. Size new-worker memory correctly

Existing shared-host safety reserves remain unchanged. New disposable workers use representative workload peak + explicit uncertainty margin + measured OS/runtime reserve. `run_safe.sh` accepts MIN_AVAIL_MB; do not inherit its default blindly. Keep RSS and address-space watchdogs enabled. Set the virtual-memory cap from observed address space: memory-mapped files can occupy much more virtual space than resident RAM.

Provision upward when the intended model needs more memory. Stream larger data; do not shrink the architecture or credit horizon merely to fit the cheapest profile. Conversely, do not impose the8GiB shared-host reserve on every small disposable worker.

## 5. Stop and reclaim

After completion upload complete result/checkpoint/logs and verify checksums remotely. Delete the worker via API, confirm deletion, and reconcile billed disks/addresses/snapshots separately. Keep evidence in durable storage and git as appropriate; compute instances are disposable. Never delete the sole copy of a checkpoint to save instance cost.
