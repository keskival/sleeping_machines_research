# Sleeping Machines: first SystemVerilog circuit definitions

16 October 2026. Digital RTL drafts for the native temporal/race/addressed-memory family. **Four blocks pass executable RTL contracts.** These are reusable datapath/control blocks, not a complete flagship-model accelerator. No FPGA has been programmed and no fabricated chip, FPGA utilization, device frequency or measured energy result is claimed.

Use **SystemVerilog** for the hand-written synthesizable design and assertions. AMD synthesis supports SystemVerilog, Verilog and VHDL ([UG901](https://docs.amd.com/r/en-US/ug901-vivado-synthesis)); an ASIC implementation subsequently maps the design through a selected library/PDK and physical sign-off ([OpenROAD flow](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts)). The choice of HDL is a toolchain decision, not the substance of a patent claim. Keep the first prototype synchronous with explicit event timestamps. A later GALS/asynchronous implementation needs its own handshake, CDC, arbitration and delay characterization; clocked RTL does not establish clockless benefits.

| Circuit | Actual behavior | Model connection / missing work |
|---|---|---|
| [sm_race_select.sv](rtl/sm_race_select.sv) | Streaming earliest supplied due-time selection, stable arrival-order ties, ready/valid backpressure, separate address/tag output | Native receipt and timer selection in `indexed_temporal_object.py` and `pure_race_timer.py`. Clock probability transforms, random variates and candidate key scoring remain upstream. This is neither a complete exponential-race generator nor an EasyTPP likelihood engine. |
| [sm_versioned_state_bank.sv](rtl/sm_versioned_state_bank.sv) | Selected addressed reads/writes, expected-version checks, timestamp availability checks, overflow rejection; no idle payload sweep | Native indexed state and publication ownership. Stores latest values; immutable historical value storage and outstanding version lifetimes require a separate allocator. A write command carries an already computed proposal, not the whole native transition. |
| [sm_temporal_transport.sv](rtl/sm_temporal_transport.sv) | One complex pair per cycle, supplied decayed cosine/sine coefficients, declared signed Q-format, floor shift and saturation | `split_evolve`/native lazy complex transport. Coefficient generation from elapsed time/rate/frequency is not implemented; protected modes use identity coefficients. No claim of floating-point checkpoint parity. |
| [sm_binding_directory.sv](rtl/sm_binding_directory.sv) | Bounded immutable parent-version/child-version references, child identity separate from pose, causal availability, idempotent identical writes, mutation/capacity rejection | Versioned part bindings and identity/role hierarchy. References require a separate retained-value store; no garbage collector, sparse query policy or native learned relation operator is implemented. Search compares all directory slots and is charged. |

The chips do **not** have to be defined in VHDL. A foundry or ASIC service may accept RTL for a turnkey service, or require a signed-off layout/database; establish that handoff, process, SRAM options and test/package requirements before committing to a fabrication route. FPGA BRAM/DSP primitives belong behind wrappers so they do not become the ASIC interface.

## Verification and reproduction

From the repository root:

```sh
.venv-docker/bin/python hardware/scripts/verify_rtl.py
.venv-docker/bin/python hardware/scripts/synthesize.py
```

The first command requires Icarus Verilog (`iverilog`, `vvp`) and checks race selection/ties/stalls, causal/versioned state ownership, fixed-point transport against integer reference arithmetic, and immutable binding identity/version rules. The second requires Yosys and checks synthesizability at default parameters. [RTL contracts](evidence/rtl_contracts.json) and [generic synthesis](evidence/generic_synthesis.json) bind the sources. Generic cells are not FPGA LUTs, physical area, timing or energy.

The integer timestamp interface is unsigned and nonwrapping within an execution epoch. Reset invalidates all state/directory entries and drops outstanding work; it is not a live epoch rollover. Tie order follows the supplied candidate stream and must be matched to the reference execution. Clock-to-time conversion, timestamp precision and finite-race quantization remain part of model validation.

The [acquired hierarchy profile](profiles/native_hierarchy.json) binds the actual255394parameter checkpoint and every tensor shape:8levels,4stages per level,4heads,8coordinates per head,one protected pair and three learned temporal pairs per head. `hardware/scripts/export_hierarchy_profile.py` regenerates this static export without fitting or evaluating a model.

## Current flagship admission boundary

The public-winning TPP family includes heterogeneous/defective clocks, per-clock marks, state-clock quadrature and a causal pointer mixture in selected members. The deep keyed language member includes per-layer temporal state, keyed reads, residual maps, normalization and vocabulary output. The native hierarchy includes learned proposals, protected mode transport, hard receipt races, persistent state, generated views, versioned part access and counterfactual credit. These four RTL blocks cover shared primitives, not every operation of any of those models. The newest identity-role hierarchy is a development candidate; its resource gate passes, not a new prediction win.

Before calling a bitstream a flagship implementation: export a selected acquired checkpoint and actual event trace; specify all numerical formats and nonlinear approximations; implement its remaining operators; compare intermediate states, selected routes, versions and output probabilities; then evaluate task quality under the frozen benchmark protocol. Stochastic races need an explicit common random-word interface. Quantization and finite timing may change winners; count disagreement and prediction loss rather than assuming exact probability laws survive rounding.

## FPGA to ASIC evidence

1. Select a board/part, tool release, clock target and fixed complete model configuration. Use the supplied Vivado out-of-context script for block synthesis/implementation; add board constraints and a real host/stream interface before a system run.
2. Record post-route LUT/FF/BRAM/URAM/DSP counts, timing closure, logical/physical memory, routing and per-event traffic. Report capacity and selected activity separately. Capture counterfactual/optimizer work for learning, rather than inferring training costs from inference.
3. Measure end-to-end throughput, latency distribution, sustained bursts and idle/active board power at the same prediction quality. Record sampling instrument, rails, temperatures, clock, workload and idle baseline. Tool power reports are estimates; board joules include board overhead.
4. Port the same numerical/event contract to the chosen ASIC PDK. Recharacterize SRAM, cells, placement, parasitics, clocks/handshakes, DFT and sign-off. FPGA area/power does not numerically predict ASIC area/power. Manufacture only after a reviewed physical-design/test/package handoff.

A hardware patent leg should describe concrete storage, timing, arbitration, dispatch and learning arrangements and their technical consequences. FPGA measurements can substantiate those consequences; they do not establish novelty. Include enabling hardware embodiments before the relevant priority filing: later additions do not automatically inherit earlier priority ([USPTO provisional disclosure](https://www.uspto.gov/patents/basics/apply/provisional-application), [EPO G-II 3.6](https://www.epo.org/en/legal/guidelines-epc/2026/g_ii_3_6.html)). See [patent LaTeX](../ip/latex/README.md).
