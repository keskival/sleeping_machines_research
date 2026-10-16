# Scaling runbooks

Scale toward competitive quality and complete resource efficiency. Choose infrastructure for the intended larger model/data workload; today's small model is not a ceiling. These runbooks prepare execution, not new experiment admissions or cloud purchases.

0. [GPU scaling ladder (R1)](GPU_SCALING_LADDER.md): rented single-GPU worker, flagship and memory-only variants from 1e16 to 1e18 FLOPs with scaling diagnostics (needs a purchase order).
1. [Model and data scaling](MODEL_AND_DATA.md): select the decision-changing larger configuration, estimate resources, gate the implementation and register the acquisition.
2. [Capacity and provisioning](CAPACITY.md): compare API-managed offers by measured useful throughput/cost and provision a matching worker.
3. [Execution and recovery](EXECUTION.md): immutable queues, watchdogs, complete checkpoints, evidence publication and safe shutdown.
4. [Distributed and asynchronous training](DISTRIBUTED.md): separate independent experiment fleets from one distributed learner; remove coordination only with explicit consistency contracts.

Read ../HANDOFF.md and ../PRODUCT_ORDERS.md first. Current running jobs and completed evidence are preserved. No new dense reference training; language remains CPU-only and properly tokenized under current orders. Prototype gates withdrawn in the14October queue cleanup are not re-admitted by these documents.

Existing shared-host reserves remain in force. For newly provisioned disposable workers, use a measured OS/runtime reserve and workload-specific headroom: the8GiB shared-host reserve is not a universal instance minimum. One training job per host remains the default; the three-slot exception applies only to the named AWS gym. More cores do not authorize arbitrary job packing.

[Current fleet estimate](CURRENT_FLEET_ESTIMATE.md) separates active jobs, reserved stages and approximate published compute rates.
