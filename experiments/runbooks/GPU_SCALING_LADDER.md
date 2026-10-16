# GPU scaling ladder (R1): run the native flagship from 1e16 to 1e18 training FLOPs on a rented GPU

16 October 2026 (Curie). Implements step 4 of [SCALING_SUPREMACY_PROTOCOL.md](../SCALING_SUPREMACY_PROTOCOL.md): take the
strongest native language member to the compute range where the published modded-NanoGPT curves live (5e16–3e17 FLOPs)
and measure our own compute frontier with enough diagnostics to see where scaling breaks. **Provisioning needs the
founder's purchase order**; this runbook prepares execution and admits nothing by itself. **On hold (founder direction 16 Oct):** the ladder fields the native causal event race member (theory note 169) once it passes its CPU gates; the plan-B variants below are diagnostic controls only.

GPUs are a training accelerator here, not the target. The deployment target is clockless event hardware. Guardrails:
the model is defined by DeepKeyedLM's event semantics, and the GPU kernels are an exact parallel rewrite of them (contract:
loss and every gradient agree to 3e-6). Architecture is judged on prediction quality and work the target hardware
pays (FLOPs, state, traffic, causal depth); GPU wall time and MFU are implementation diagnostics only. Sparse,
asynchronous and hard-routed mechanisms keep their CPU development path and are integrated when they win on that
accounting. GPU speed is not a reason to drop them.

## 1. What runs

| Piece | File |
| --- | --- |
| Driver: flagship and variants, exact GPU kernels, one pass over fresh FineWeb lanes, diagnostics, exact resume, sealed public scoring | `experiments/r1_gpu_ladder.py` |
| Ladder specification: variants, budgets, shapes, LR sweep, depth probe, worker guards, admission declaration | `experiments/gpu_ladder/r1_gpu_ladder_v1.json` |
| Planner/runner: measures FLOPs/token and tokens/s per shape, chooses arms, runs each through `run_safe.sh` | `scripts/gpu_ladder_runner.py` |
| Analysis: IsoFLOP fits, frontier exponent, depth probe, variant gaps, published-curve overlay, per-run scaling flags | `analysis/gpu_ladder_fit.py` |
| Worker bootstrap: driver, PyTorch, data, vocabulary, identity checks, contract | `scripts/gpu_worker_setup.sh` |

**Model.** `DeepKeyedLM` (plan-B): token embedding tied to the output; per block, pre-norm residual temporal memory
(complex decay and rotation per step), per-layer keyed race read (attention with learned per-head distance decay, keys
and values from each event's state and its predecessor's message), MLP; output keyed token-slot bonus (learned cache).
GPU execution:
- temporal memory as a fully parallel chunked kernel in fp32;
- reads through scaled-dot-product attention with the decay as an additive fp32 mask;
- the 50,257-way softmax plus slot bonus in row chunks under activation checkpointing;
- bf16 matmuls, `torch.compile`d blocks, fused AdamW.

Width rule: heads = d/64 (at least 4), key width 64 (d ≥ 256), memory modes = d/2.

**Variants.**
- `flagship`: the full model.
- `memory_only`: no per-layer reads. O(1) state per step, the cheapest member on event hardware. It tests whether
  persistent memory alone carries the scaling.
- `no_bonus`: no output slot bonus, at 1e16 and 1e17 only. It measures how much the cache contributes as scale grows.
- `read_only` (no temporal memory) is a diagnostic control and stays out of the ladder.

**Protocol.**
- Training data: llm.c GPT-2 shards `fineweb_train_*.bin`, in name order, starting at the first shard's offset 0 (the
  same start as the CPU ladder). Lanes of 1024 tokens, each visited once in a seeded permutation.
- Updates: 64 lanes (65,536 tokens) per update; warmup 2%, then cosine decay to 10% of the peak learning rate. Peak LR
  is 3e-3·(256/d)^½ × the sweep multiplier.
- Evaluation: disjoint 1024-token windows of `fineweb_val_000000`. The selection slice (16,384 targets) is scored at
  24 log-spaced steps; the development slice (65,536 targets) at 1/4, 1/2 and the end; the flagship windows at the end.
- The public range (targets 1..10,485,760) is sealed. It is scored once per variant and budget, for that budget's
  compute-optimal arm only (`--mode public`).
- Compute is measured forward+backward FLOPs per training target, without the loss-chunk recompute (the CPU ladder's
  convention).

**Budgets and arms.**
- Budgets: 1e16, 3e16, 1e17, 3e17, 1e18 FLOPs.
- Arms per budget: up to 5 shapes from d128 L4 to d1024 L16 whose tokens per parameter fall in [5, 200]. This makes
  IsoFLOP curves.
- LR sweep: ×{0.5, 1, 2} at 1e16 on d256 L8 and d512 L8. The ladder uses the geometric mean of the best multipliers.
- Depth probe: d512 at L4/8/16/24 at 1e17, for both main variants.
- Size: 69 arms. The largest uses about 5.1B tokens.
- Estimated time: about 40 GPU-hours at 30% MFU on an RTX PRO 6000. The plan phase replaces this with measured numbers.

## 2. Hardware

**Choice: Hetzner GEX131.**
- 1 × NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB GDDR7, 1.8 TB/s, about 439 TFLOP/s dense bf16 by third-party spec).
- Xeon Gold 5412U (24 cores), 256 GB RAM, 2 × 960 GB NVMe, Falkenstein/Nuremberg/Helsinki.
- No minimum term; Hetzner quotes it hourly (launch: €889/month or €1.4247/hour, Hetzner press release 11 Dec 2025).
- Third-party sources report about €1,199/month plus a €599 setup fee from August 2026. Confirm the live price in the
  configurator before ordering.

Why GEX131:
- The 96 GB card fits every planned shape at 64k tokens per update, with no model parallelism.
- Hourly billing suits a bounded ladder.
- It costs about half as much per peak FLOP as an AWS H100.

| Option | GPU | Price (verify) | Peak bf16 | Note |
| --- | --- | --- | --- | --- |
| **Hetzner GEX131** | RTX PRO 6000 Max-Q 96 GB | €1.42–1.92/h, setup €0–599 | ~439 TFLOP/s | chosen; one GPU per server (Hetzner does not offer multi-GPU GEX) |
| Hetzner GEX63 | same GPU, Core Ultra 7, 128 GB | €999/month + **€999 setup** (press release 6 Oct 2026) | ~439 | same GPU; the setup fee dominates a short ladder |
| AWS p5.4xlarge | 1 × H100 80 GB | ~$6.9/h us-east-1, ~$7.4/h eu-north-1 (trackers) | ~989 | ~2× faster per run, ~2× cost per FLOP; use if wall time matters more than cost |
| AWS g7e.2xlarge | RTX PRO 6000 96 GB | ~$3.36/h on demand, ~$1.5/h spot (trackers) | ~439 | spot is viable: the driver resumes exactly on SIGTERM |

**Setup fee.** A setup fee is a one-time charge per server order. It is added to the usage charges and is not a month's
rent. It is charged again if a server is cancelled and re-ordered. So keep one server for the whole ladder instead of
ordering and cancelling per batch. For the GEX63 the fee happens to equal one month's price.

**Cost envelope.** About 40–80 GPU-hours, roughly €60–150 of usage at €1.4–1.9/h, plus any setup fee. Two servers (one per
variant shard) halve the wall time at the same usage cost.

## 3. Provision and bootstrap

1. Order GEX131 in Hetzner Robot: Ubuntu 24.04, your SSH key, location FSN1 or NBG1. Name it `hz-gpu-1`.
2. On the worker, as root:
   - `ssh-keygen -t ed25519`. Add the public key to the GitHub repository as a **deploy key with write access**.
   - `git clone git@github.com:keskival/sleeping_machines_research.git /workspace`, then
     `git config user.name/email` (the founder's identity, as on other hosts).
3. Run `bash /workspace/scripts/gpu_worker_setup.sh`, reboot after the driver install, and run it again. It:
   - installs the NVIDIA open driver;
   - creates `/workspace/.venv-docker` (the path `run_safe.sh` expects) and installs the project's PyTorch version from
     the CUDA wheel index, which must ship Blackwell `sm_120` kernels;
   - downloads 60 shards (6B tokens, ~12 GB) plus validation from `kjj0/fineweb10B-gpt2` and the GPT-2 `encoder.json`;
   - checks their SHA-256 against the Curie copies;
   - runs the CPU contract.
4. One training process per GPU. The worker has no shared-host reserve. Its guards come from the spec's `worker` block:
   - MemAvailable floor 16 GiB;
   - RSS watchdog;
   - virtual-memory cap `unlimited`, because CUDA reserves far more address space than it uses.

## 4. Gates before the ladder (about 30 minutes)

```bash
cd /workspace
.venv-docker/bin/python experiments/r1_gpu_ladder.py --mode contract --tag hz_gpu_ladder_contract_v1
.venv-docker/bin/python experiments/r1_gpu_ladder.py --mode gpu_check --d 512 --layers 16 --tag hz_gpu_check_d512_L16_v1
.venv-docker/bin/python experiments/r1_gpu_ladder.py --mode gpu_check --d 1024 --layers 16 --batch 64 --tag hz_gpu_check_d1024_L16_v1
```

Pass criteria:
- fp32 GPU loss within 1e-4 of the CPU reference, and gradients within 1e-2 relative;
- bf16 eager and compiled losses within 3e-2;
- the largest shape fits in 96 GB at batch 64 × 1024.

If it does not fit, set `"batch": 32, "accum": 2` in the spec's `common` block. Tokens per update stay the same, and the
plan must be re-made.

Record eager tokens/s. If SDPA falls back to the math kernel because the mask needs a gradient, the logs show high
memory and low speed. The fix is a fused learned-slope attention kernel (FlexAttention `score_mod`), which is
implementation work. It does not change the model.

## 5. Plan

```bash
tmux new -s ladder
.venv-docker/bin/python scripts/gpu_ladder_runner.py --phase plan
```

This writes `experiments/results/gpu_ladder/hz_r1_ladder_v1_plan.json`: measured FLOPs/token, parameters, tokens/s, MFU,
GPU memory and RSS per shape; the arm list; and estimated GPU-hours. Check it before running:
- Every budget has at least 3 arms that bracket the expected optimum: tokens per parameter around 20, interior to the window.
- Total hours are within the purchase order.
- `rss_cap_kb` is at least 3 × the largest RSS seen.

Commit the plan (section 8).

## 6. Run

One server:

```bash
.venv-docker/bin/python scripts/gpu_ladder_runner.py --phase run --kinds lr_sweep      # ~1 h; sets the LR multiplier
.venv-docker/bin/python scripts/gpu_ladder_runner.py --phase run                        # everything else, smallest budgets first
```

Two servers:
1. Run the LR sweep on `hz-gpu-1`.
2. Commit and push the sweep results; pull them on `hz-gpu-2`.
3. Start `--phase run --shard 0/2` on server 1 and `--shard 1/2` on server 2.

`--max-hours H` stops admitting new arms past H estimated hours. Each arm is an immutable one-job queue,
`experiments/queue/r1/<tag>/<tag>.txt`, carrying its admission declaration. It runs through `run_safe.sh` (host lock, RSS
watchdog, memory floor, timeout = 2 × estimated time + 1 h) and logs to `experiments/queue/logs/<tag>.log`. The analysis
reruns after every arm.

**Interruption.** On SIGTERM, Ctrl-C, timeout or a spot reclaim, the driver finishes the current update, writes an
exact checkpoint (model, optimiser, RNG, step and the complete log state) and exits 75. Rerunning the same runner command
resumes bit-exactly: on CPU, an interrupted-and-resumed run gave the identical final losses. A setting change requires a
new tag. If hardware fails, copy `experiments/results/gpu_ladder/*.resume.pt` to a new worker and rerun.

## 7. Diagnostics: what is collected and how to read it

Each run streams `experiments/results/gpu_ladder/<tag>.diag.jsonl` and ends with `<tag>.json`. After every arm,
`analysis/gpu_ladder_fit.py` writes `hz_r1_ladder_v1_analysis.json`, `_report.md` and `_scaling.png`.

| Signal (where) | Scaling failure it reveals | Action |
| --- | --- | --- |
| Frontier exponent α and local slope per decade (analysis) | flattening frontier: compute stops buying loss | read the rows below for the cause before scaling further |
| IsoFLOP N*(C) and tokens/param at the optimum | optimum at the edge of the grid; data- or capacity-limited regime | add shapes beyond the edge and re-plan |
| Depth probe (d512, L4–24, 1e17) | depth stops paying | check logit-lens gains and update ratios; composition change, as on CPU |
| Logit-lens loss per layer (diag) | dead layers (< 0.005 nats gain) | depth wasted: residual scaling, per-layer read reach, LR per depth |
| Sublayer update ratio and residual max (diag) | late layers rewriting the stream; activation growth | residual init scale, normalisation, lower LR at depth |
| Read entropy and mean attended distance per head (diag) | attention collapse; no long-range use at scale | decay-rate init and range; check history-bucket losses |
| Memory rates, frequencies and gate activity (diag) | memory modes pinned at the range ends, or idle gates | widen the rate range; modes per width |
| Loss by history bucket (milestones) | long context not improving with scale (256–1024 vs 0–16) | memory and read reach |
| Loss by token role (milestones) | gains only on continuations; word-initial (semantic) loss flat | where depth helped on CPU; composition |
| Slot-bonus gain (diag) and the `no_bonus` gap | cache value shrinking or growing with scale | keep or redesign the bonus |
| Loss spikes, non-finite steps, clipped fraction, per-module update/weight ratio (log, diag) | optimisation instability at width or depth | LR multiplier, warmup, clip, wd per module |
| Prequential by eighth, `under-trained` flag | still falling steeply at the end | schedule length; tokens per parameter |
| MFU and tokens/s by shape (log) | implementation bottleneck (small widths launch-bound) | kernel work; never an architecture verdict |

## 8. Publish and compare

- Commit and push from the worker after each budget completes: result JSONs, `.diag.jsonl`, plan, analysis, report, figure
  and the arm queue files. Work on `main`; run `git pull --rebase` first and preserve other hosts' changes.
- Checkpoints (`*.pt`) are gitignored. `rsync` the final ones to Curie or object storage, keeping at least every
  compute-optimal arm.
- Score the sealed public protocol once per variant and budget, for the IsoFLOP optimum only:
  `experiments/r1_gpu_ladder.py --mode public --run <tag>`. It refuses to overwrite.
- The analysis overlays these points on the published record curves. Reference training FLOPs are estimated from the
  GPT-2 124M shape (6N + 6·L·T·d per token).
- **Decision rule** (protocol): claim scaling supremacy only if ours lies below the published Transformer curve across at
  least one decade of compute with a non-shrinking gap. Otherwise report the measured gap, its trend and the
  diagnostics that explain it.
- Update HANDOFF, theory note 5 (depth predictions), Part I §4.4 and the claims registry from completed results only.

## 9. Shutdown

Cancel only after all of the following:
- `gpu_ladder_runner.py --phase status` shows every arm completed (or the cap reached);
- results are pushed (`git log origin/main` shows them);
- checkpoints are copied off the worker.

Then cancel the server in Robot. Billing stops at cancellation, and a later order pays any setup fee again.
