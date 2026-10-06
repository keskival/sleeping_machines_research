# Plan to create value (6 October 2026, curie session; user-directed)

## What went wrong

For two days the dominant pattern was:
1. pick a public benchmark;
2. build a quick draft model;
3. run the official protocol once;
4. record a loss.

That spent one-shot official protocols (Mackey-Glass, primate) on configurations nobody expected to win. It measured
drafts, not the substrate. It put uninformative losses into investor-facing tables. And it drew compute away from the
substrate-level failures that affect every benchmark. It resembled work without creating value.

## What we have that is real

1. **FAS early fault detection on interleaved event logs** (the user's own simulator, our home field):
   - native beats six classical controls at N = 256 over three seeds;
   - the identity-aware oracle shows headroom (.755 against our ~.60).
2. **A measured root cause** (THEORY §§429–431):
   - the route credit is nearly blind on FAS;
   - forcing one choice flips a median of 125 later routes when memory is long;
   - first-order credit works only where route sensitivity is low, where the transported write credit raises fidelity
     from .00 to .49.
3. **A confirmed theory prediction:** asymmetric private/shared decay groks 1.7× sooner (§427.3; single seed).
4. **Inference efficiency against Transformers** (18–28% of their per-character inference arithmetic), and quality
   wins against tuned Transformers at ~equal training compute.

## The plan

**Rules, effective now:**
- **Maturity gate:** no official, public or investor-table comparison is run until validation evidence predicts a win
  with margin, on the same protocol and at matched compute.
- Draft results live in development notes, never as headline verdicts.
- **One flagship at a time, carried to maturity: FAS.** Other benchmarks are frozen unless they test a substrate fix.
- Every development step is chosen by a measured failure (credit audit, route sensitivity, binding diagnostics), not by
  a knob sweep.

**Stage 1. Establish the FAS bar (AWS, now).** The five neural references (LSTM/RMTPP-pattern,
Transformer/THP-pattern, LRU, S5-style, Mamba-style; AWS_FAS_REFERENCES.md) and a tree reference under the frozen
clean-only causal protocol. Their *validation* AUROCs set the development target.

**Stage 2. Fix the measured failure on FAS, on validation only:**
- **2a. Smooth delivery in long-memory routing.** The exact expected-reception member (§418) delivers Σπv, a
  deterministic value with exact gradients, while keeping hard sparse writes. It removes route chaos from the value
  path. Measure route sensitivity S, credit fidelity, learned half-lives and validation AUROC against the race member
  at equal settings.
- **2b. Margin control** for race members: keep long memory, enlarge routing margins, then re-measure S and fidelity.
- **2c.** Transported write credit (§430) wherever S is shown low; rollout credit for high-S races.
- **2d.** Scale the native model to the references' size and compute once a mechanism is validated (the native p32/d4
  is ~111K parameters against references of several hundred K).

**Gate for Stage 3:** native validation AUROC at N ≤ 256 exceeds the best reference's by more than the seed spread, at
no more training and inference compute, on three seeds.

**Stage 3. One clean test-set comparison.** Pre-declared configuration, three seeds, all references, then the
headline. FAS is our own simulator, so the generator and protocol are published with it.

**Stage 4. Carry the fix to language.** Audit credit fidelity and route sensitivity on the tokenized-language models of
the other curie session. Apply the fix that worked on FAS, and compare against tuned references only after the
development gate.

**Investor packaging:**
- Only matured comparisons appear in headline tables: the FAS win (after Stage 3), inference efficiency, and the
  tuned-Transformer comparisons.
- The scientific story — a measured credit blind spot, route chaos, a confirmed grokking prediction, and the fix — is
  presented as evidence that we understand and control our learning dynamics.

## Resources

- Curie is shared by two agent containers with unshared locks. FAS development arms are small (~1–2 GB, ~1 h each)
  and run only under explicit coordination (HANDOFF), never beside another training job.
- Evaluation-only diagnostics (audits) are minutes long and run under the memory guard.
- AWS runs Stage 1 first.
