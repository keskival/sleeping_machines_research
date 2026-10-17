# Dead ends and broken constructions: read before designing

Curated 17 October 2026 (Curie, after the flagship review; founder: "Mark things which cannot work as such so that future
agents don't go down those paths").

Each entry names one specific construction, the evidence, the reason it fails, and what to do instead. These are verdicts on
the construction as built, not on the model family; AGENTS.md: a restricted variant is evidence about that variant. Reopen an
entry only by removing its stated cause, and write down which cause you removed.

Status labels:
- **DEAD**: fails for a structural reason; do not rerun or scale it.
- **BROKEN-FIXED**: an implementation defect, now repaired behind a flag; never run the defective form again.
- **CONFOUNDED**: a result produced under a defect; do not cite it as evidence about the mechanism.

## 1. Depth and composition interfaces

| Status | Construction | Evidence | Why it cannot work | Instead |
| --- | --- | --- | --- | --- |
| DEAD | Serial chain of race stages passing only the winner's dense value upward | 1M dev: serial d8 7.176 vs residual 7.103. Any one of 16 content cuts costs the same 1.56 nats. Route entropy 0.33 of 1.39 nats at D16. Deep gradients about 1e-5 (`curie_depth_ladder_*`, RESIDUAL_DEPTH_20261015.md) | A hard winner chain is a single-path bottleneck: one bad route destroys content, and credit to unchosen routes vanishes with depth | Residual memory (theory 172): every level adds a correction to one shared memory |
| DEAD | Growing depth by stacking race stages on the race core, with no memory interface between them | Protected D4 6.095 vs parent 6.066 (`aws_protected_depth_*`) | Same bottleneck as above; added stages only add noise to the readout | Residual memory |
| DEAD | A hierarchy of races over level-1 *attribution events* (CERL-H as built, `r1_cerl_h.py`) | Causes change on 79% of tokens, median duration 1; compression only ρ ≈ 0.75–0.83; concept units are syntactic slots (`curie_cause_stream_typing_cerl4M_20261016_v1`) | The upper level sees almost the same-rate stream of non-durable, non-abstract symbols, so there is nothing slower or more abstract to model | Hand over *memories* (residual memory). Event hierarchies first need durable cause identities (target ρ < 0.5) |
| DEAD | Part-whole summaries at observed word, clause and sentence boundaries | Dense L2 4M: 5.859 vs 5.865 (0.006) | Boundaries are already visible to the token model; summaries add no new information | Learned pooling into standing slots (RMR-2) |
| DEAD | More data or training for the G-HIER whole-object binding construction | Parts 512/512, wholes at chance (NLL 0.693) even after 32x more training (`investment/composition_scale_evidence.json`) | The construction has no path that binds parts into wholes; extra steps do not create one | Change the binding mechanism before spending more training |
| DEAD | Ungated old-slot corrections (no ReZero gate, no tanh) | Exploded to 6.93 at update 300 | Corrections to old keys change what every later reader retrieves; unbounded at init, they destabilise the readout | ReZero gates at 0 plus tanh (the default) |
| DEAD | Old-slot corrections that never decay, scored on lanes much longer than training | K4 16M: 65,528 slice 5.0874 vs CERL 5.0282, while flagship improves 4.9901 vs 5.0634 | A slot accumulates corrections from every later reader: 32x more writes on 8,192-token lanes than seen in training | Learned decay `--old-decay` (bounded algebra), and train on long lanes (`--lane 1024`) |
| DEAD | Local learner whose depth mixture detaches lower-layer messages (AWS native local stream) | 64k: flag 7.047 at about 29 tokens/s | Lower layers get no credit for what upper layers need, so they cannot learn to serve them | Producer-credit replay (AWS repair running); RMR, where credit reaches every level through the shared memory |
| DEAD | Top-down additions that are not gated at init (CERL-H top-down waking) | The equality with the base model at λ = 0 failed | Any branch that enters the race at init changes the base model before it has learned anything | Gate at a large negative value (`td_wake` −12) or ReZero |
| LIMITED | One old-slot correction vector shared by all m slots a level read | Code review, 17 Oct | A level cannot refine different old slots differently | Per-target corrections `--old-per-target 1` (contracts v5; 16M arm queued) |

## 2. Learning setup (all native drivers)

| Status | Construction | Evidence | Why it cannot work | Instead |
| --- | --- | --- | --- | --- |
| BROKEN-FIXED | AdamW weight decay over `model.parameters()` | Over 16M, decay pulls the age slope from 1.5 toward 1.29 (non-integrable side), triples the learned old-slot decay rate, shrinks ReZero gates and biases | Decoupled decay pulls every softplus-parameterised clock toward softplus(0) and erodes capacity meant to wait unused | `param_groups(model, wd, True)` / `--wd-groups 1`: decay matrices only. Never pass `model.parameters()` to AdamW in a native driver |
| BROKEN-FIXED | Concept vectors `unit_out` initialised to exact zeros, with no load balance | 124 of 1,024 units used (cause-stream typing) | Every unit starts as the same word law, so keys get no differential credit; nothing pushes traffic onto dormant units | `--unit-init 0.02 --balance 0.01` |
| BROKEN-FIXED | Training lanes (256) much shorter than scoring lanes (1,024 / 8,192) for mechanisms whose state grows with length | The long-lane regression above | Length-dependent statistics are never seen in training | `--lane 1024` (16M arm queued); carried memory across lanes is the full fix |
| BROKEN-FIXED | Dense [B, T, T] old-write matrices | Two final-scoring memory stops on Curie | Quadratic memory at 8,192 tokens | Sparse (idx, w) storage (default) |
| DEAD | Depth or capacity verdicts at 4M tokens | Every depth and capacity variant within 0.01 of CERL at 4M; K4 depth paid 0.073 at 16M | 4M is data-limited: extra structure cannot pay before the data supports it | Test depth and capacity at 16M or more |
| DEAD | Credit windows of 16 tokens for long-history mechanisms | 64k: window 64 vs 16 = 8.21 vs 8.03; no reward beyond about 64 tokens | The model is never credited for using history beyond the window | Full-lane credit, separate from persistent state |

## 3. Results that do not count as evidence about the mechanism

| Status | Result | Defect | What to run instead |
| --- | --- | --- | --- |
| CONFOUNDED | Capacity beyond activity at 4M: product-key banks of 65,536 / 262,144 units, flag 5.4716 / 5.4715 vs 5.4665 (`aws_r1_cerl_pk*_4M_*`) | Bank collapse (zero-init vectors, no balance), weight decay on dormant tables, data-limited 4M | The capacity ladder at 16M with `--wd-groups 1 --unit-init 0.02 --balance 0.01` |
| CONFOUNDED | Every 16 and 17 October native run (CERL, RMR, RMR-2): magnitude of the concept cause and of learned clocks | Same two defects | Report results as measured; repeat mechanism attributions on the fixed runs |
| QUARANTINED | Target-dependent E63/E79 mixtures | Target leakage | `experiments/archive/invalid_protocol/` (AGENTS.md) |
| ORACLE | FIFO de-interleaving and the v2 timing probe on FAS | Use hidden training identities | Oracle diagnostics only (AGENTS.md) |
