# Hardware thesis: what a clockless, event-driven, memory-local substrate buys

Written 4 October 2026 at the user's request. This is the investor-facing hardware analysis. The engineering detail and
energy algebra are in [HARDWARE_VALUE_PROPOSITION.md](../experiments/HARDWARE_VALUE_PROPOSITION.md). No Sleeping
Machines chip exists yet. Every advantage below is a **mechanism with a physical basis**, not a measurement. The
evidence column says which parts the current software already shows.

## The short version

The biggest hardware lever is **not** removing the clock. It is three properties of the model that clockless,
event-driven silicon can exploit together:

1. **Work tracks information, not wall time.** Only units that receive an event do anything. Idle capacity costs storage
   leakage, not compute.
2. **Small, constant state and few bytes moved per step.** There is no growing KV cache. Each step reads the winning
   slot per head plus cached keys. Per-character traffic stays nearly flat as capacity grows (cost model below).
3. **Time is itself a computational resource.** Delays and arrival order carry values (race attention), so part of the
   computation can be done by signal propagation instead of arithmetic.

Removing the global clock is what lets hardware act on (1) and (3) natively. Memory locality (2) matters most, because
in modern AI chips **moving data costs far more energy than arithmetic**.

## Cost model of the trained models (P2, 4 October 2026)

`experiments/hardware_cost_model.py` produces `experiments/results/diagnostics/hardware_cost_model_20261004.json`. It
counts the weights and state each model reads and writes per character at steady state. The native rows use the
existing winner-only execution with cached key reads (§414). The model is int8 and batch 1, priced with Horowitz 45 nm
anchors. Its p96 count (1.28 MFLOPs/char) agrees with the recorded winner-only trace (1.3). This is a cost model, not a
chip measurement.

| Model (test bpc, 10M) | Params | Weights read / char | State read / char | Bytes / char | pJ / char, local SRAM | pJ / char, DRAM |
|---|---|---|---|---|---|---|
| Native p96/d4/U2 (1.888) | 0.94M | 0.64M (68%) | 2.3K | 0.65M | 1.00M | 104M |
| LSTM-512, 6 passes (1.799) | 1.20M | 1.20M (100%) | 1.0K | 1.20M | 1.86M | 193M |
| Transformer-256×4, 4 passes (1.908) | 3.24M | 3.24M (100%) | 524K (KV) | 3.76M | 5.83M | 603M |
| Native p64/d4/U32 (capacity probe, quality not measured) | 4.43M | 0.29M (7%) | 17K | 0.31M | 0.48M | 49M |

- Against the Transformer it beats, the native model moves **5.8× fewer bytes per character** and has ~5.8× lower
  modeled energy. Against LSTM-512, which is still better in quality, it moves 1.9× fewer.
- **Capacity beyond activity holds at the hardware traffic level:** from pool 2 to pool 32, parameters grow 10.5× and bytes
  per character grow 5%. At pool 32, 93% of the weights stay untouched on a given character. Those weights can sit in
  dense, low-leakage memory and draw no data-movement energy. Quality at pool 32 still has to be shown.
- **Against tuned baselines (5 Oct, `hardware_cost_model_20261005_tuned_refs.json`):** the best tuned Transformer (TF128×4,
  1.996 bpc) moves 1.10M bytes/char. The native model is better *and* moves 1.7× fewer bytes. Tuned LSTM-384 (1.840 bpc,
  better than native) moves 0.70M bytes/char, only 8% more than native's 0.65M. Against tuned LSTMs there is no current
  traffic advantage at this scale. The structural one (flat traffic as capacity grows; no KV cache at longer contexts)
  still needs a quality point.
- The remaining traffic floor is the shared dense mixing between heads, not key scoring
  ([theory 154](../experiments/theory/154_key_scoring_traffic_floor.md)).

## Question by question

| Question | Answer | Why (physics / architecture) | Main caveat |
|---|---|---|---|
| **Less energy?** | **Yes, likely the largest benefit, mostly from locality and sparsity rather than clock removal alone.** | At 45 nm, a 32-bit DRAM read costs ~640 pJ, a small-SRAM read ~5 pJ and a 32-bit float multiply ~3.7 pJ (Horowitz, ISSCC 2014). Dense low-batch inference is dominated by streaming weights from DRAM/HBM. Our current 1.888-bpc model moves ~5.8× fewer bytes per character than the Transformer it beats (cost model below). Its traffic stays nearly flat as capacity grows 10×. Clock-tree power, a significant share of dynamic power in synchronous chips even after gating, disappears. Idle units draw only leakage. | The saving must be measured against a competent clock-gated, SRAM-heavy synchronous design, not only against a GPU. Handshake/timing overhead, routing and leakage of large on-chip memories are real costs. |
| **Faster chips?** | **Lower latency for sparse, streaming, low-batch work, yes. Higher dense-matmul throughput, no.** | Asynchronous logic completes at average-case rather than worst-case delay: no clock margin for the slowest path, and no waiting for the next tick. Events propagate as soon as they are ready, which is ideal for real-time streams and single-user inference. Race computation resolves when the first signal arrives. | GPUs remain excellent for large-batch dense throughput. Our claim is latency and energy per useful result, not raw FLOP/s. |
| **Less cooling?** | **Yes, as a consequence of lower average power.** | Heat ≈ power. Event-driven activity means average power scales with event rate. A mostly-dormant model runs cool. No clock means no always-on switching floor. | Bursty inputs can create local hot spots. Peak (not average) power sets the thermal design for some workloads. |
| **Packed more densely?** | **Yes, potentially the most strategic benefit.** | Today's chips are limited by power density ("dark silicon"), and 3D stacking of logic on memory is limited by heat. Low average activity makes it thermally feasible to stack compute directly on dense memory and to build very large dies or wafer-scale systems. No global clock means no chip-wide timing closure or clock distribution, so modules (chiplets) compose without a shared timing domain. | Interconnect, yield, and memory density per mm² still bound capacity. Stacked or non-volatile memory technologies carry their own cost and endurance limits. |
| **Local memory colocated with compute, for inference and training?** | **Yes for inference, by construction. For training, plausible but not yet demonstrated.** | The model's state is addressed, persistent and updated sparsely. It maps naturally to memory banks sitting next to small compute units (near-/in-memory computing). Weights and state stay where they are used, so the von Neumann bottleneck of shuttling them to a central processor is avoided. With non-volatile local memory (MRAM/RRAM), dormant capacity costs near-zero power. That makes "capacity beyond activity" a hardware property, not just a software one. Route credit and delay/gate updates are local quantities that could be applied in place. | Current training uses autograd, truncated backprop, shared update windows and global gradient clipping, which are not local rules. On-chip learning needs an explicit local/event-triggered learning rule shown to converge (theory §321). Optimizer state and credit traces also need local storage. |
| **Cheaper on-chip communication?** | **Yes, when activity is sparse.** | Messages are sent only when events happen (as in neuromorphic AER), not as full vectors every clock. Wire energy and wiring area scale with event rate. See the section below. | Per-message address overhead, burst buffering; compare against a clock-gated design. |
| **An advantage over GPUs and dense models?** | **Yes, structurally. Not uniquely as a hardware idea.** | A dense Transformer reads all its weights every token, so at low batch it is memory-bandwidth-bound. Its parameter count and its per-token cost scale together. Ours decouples them. That is the precondition for memory-local silicon to pay off. | Dense models can also use SRAM-heavy or near-memory chips (Cerebras, Groq, d-Matrix), and MoE already decouples capacity from activity partially. Our edge is the model's access pattern, which must be shown at competitive quality. |

## On-chip communication: send only when something happens

On a chip, a wire costs energy each time its voltage flips (roughly C·V² per transition, with C growing with wire
length). Long wires and the network between cores are now a major part of chip energy and area. In a dense synchronous
design every layer exchanges full activation vectors at every step, and the global clock line itself flips every
cycle. In our model a module sends a small message only when an event occurs. Neuromorphic chips do the same with
address-event representation (AER) packets. That gives:

- **Energy:** channel energy scales with the message rate instead of the clock rate × vector width. Silent channels
  cost nearly nothing.
- **Fewer wires, so area and density:** links are sized for the *average* event rate, not peak full-width transfer every
  cycle. Narrow packet links replace wide parallel buses, which frees routing area (often the real limit on how densely
  logic can be placed) and eases 3D/chiplet interconnect.
- **No global timing on long wires:** asynchronous handshakes need no matching of clock skew across the die, which
  makes long-distance and cross-chiplet links simpler.

Caveats: each message carries an address/header, so very high event rates lose the advantage. Bursts need buffers and
arbitration. A well-designed clocked chip with clock gating also leaves idle data wires still. The comparison is
against that design, and the gain is largest when activity is genuinely sparse. Which side wins depends on measured
message rates: the activity and locality profiles below provide exactly that number.

## Further hardware benefits

- **Low-voltage operation.** Asynchronous circuits tolerate process, voltage and temperature variation without timing
  failure, so they can run near threshold voltage. Dynamic energy scales with V², so this is a large lever, and it is
  hard for clocked designs, which must margin for the worst case.
- **Native event sensors.** Event cameras (DVS), silicon cochleas, radar, LiDAR, network and financial ticks feed in
  directly with no frame conversion, keeping the sensors' latency and sparsity benefits end to end.
- **Energy proportional to information.** Quiet inputs cost almost nothing; this suits always-on edge, wearables,
  implants, robotics and space.
- **Low electromagnetic interference.** There is no single clock frequency radiating, which matters in medical, automotive and
  RF-sensitive settings.
- **Time-domain (race) computation.** Delay-coded values let comparison/selection be done by first-arrival detection,
  which is cheaper than digital arithmetic for min/max/selection. Prior art: race logic (Madhavan, Sherwood & Strukov,
  ISCA 2014). It is a published primitive we build on, not a claim we invented it.
- **Fault tolerance and yield.** Local, message-driven modules can route around defective units. This matters for wafer-scale
  integration.
- **On-device continual learning.** If training is local, devices adapt in the field without shipping data to a
  datacenter (privacy, bandwidth, autonomy). Our CPU online-learning pilot (3.191 → 3.096 bpc) shows adaptation as a
  model property, not as hardware.

## What this implies for the business and valuation

- **The model is the bottleneck that the neuromorphic field has not solved.** Loihi, SpiNNaker, TrueNorth, BrainChip and
  Innatera show that event-driven, memory-local silicon is buildable. Their commercial reach has been limited mainly
  because spiking models have not matched mainstream accuracy on mainstream tasks. A trainable event model that is
  competitive with Transformers and LSTMs is the missing piece for that whole hardware class. That gives us a licensing
  position toward several chip programs, not only our own.
- **The hardware upside is multiplicative, but conditional.** Software-level compute savings (our matched-compute
  results, pending tuned-baseline confirmation in P0-6) multiply with the hardware savings from locality, sparsity and
  clock removal. The two portions apply to different parts of the energy budget, so combine them through the fraction
  model in HARDWARE_VALUE_PROPOSITION.md, not by naive multiplication.
- **How to value it:** the base case values the model and runtime on existing hardware. The hardware path is a
  **real option** whose value rises sharply once (a) quality is competitive under tuned baselines and (b) a hardware
  cost model of the trained network shows the traffic and activity reductions. We recommend the deck present it
  exactly that way: a large, physically grounded upside with named gates. That is more credible to technical investors
  than a headline energy multiple, and it is what makes the upside defensible in diligence.

## What we can show before any chip exists

1. **Hardware cost model of a trained network** (P2): per-inference counts of events, local memory reads/writes, message
   hops and arithmetic, priced with published per-operation energies (e.g., Horowitz 2014 scaled to a node). Compute the same
   for the tuned dense reference on a GPU-like memory hierarchy and on an SRAM-heavy synchronous accelerator.
2. **Locality profile:** the fraction of state touched per event and the reuse distance. This is the number that decides
   whether colocated memory works.
3. **Activity profile:** events per input and the dormant fraction over real streams (language, primate, DVS).
4. **Local-learning rule:** an integrated convergence comparison of an event-local update against the current trainer.
5. **FPGA semantics prototype:** validates event queues, bank-local state and precision. It cannot measure clockless energy.

## Precedents and sources

- M. Horowitz, "Computing's energy problem (and what we can do about it)", ISSCC 2014: per-operation energy table.
- A. Madhavan, T. Sherwood, D. Strukov, "Race logic: a hardware acceleration for dynamic programming algorithms", ISCA 2014.
- Intel Loihi 2 technology brief (asynchronous neuron cores, on-chip learning), already cited in the deck.
- Further public precedents to cite when adding numbers: Cerebras/Groq (SRAM-centric dense inference), IBM NorthPole/TrueNorth (memory-local digital inference).
