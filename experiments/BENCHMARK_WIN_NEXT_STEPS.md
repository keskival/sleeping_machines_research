# Next steps toward benchmark wins

5 October 2026 · Decision memo from completed results · No new host admission.

This plan supports PRODUCT_ORDERS.md; it does not cancel active jobs, alter
frozen queues or change their owner order. Physical owners must inspect current
slots, locks, memory and result files before admitting any run through run_safe.sh.

## 1. Complete the credible public comparison first

Finish the fixed six-session NeuroBench primate run. Two sessions currently
average R² .684, against .677 for bigRSNN on those same sessions. The published
six-session leader is AEGRU at .710. The remaining four sessions need an average
above approximately .723 for our complete mean to exceed .710; the current
partial result is not a leaderboard win. Report all six sessions and the five
untouched-session mean, since one session was used for development.

Compute NeuroBench-compatible footprint and operation columns. Our approximately
179KB footprint is far below bigRSNN's 4.83MB, but above AEGRU's 45.5KB and
tinyRSNN's 27.1KB. A potential win against bigRSNN is not automatically dominance
over every entry. Complete the protocol before choosing a follow-up variant;
use validation, not the unfinished test-session ranking, for future choices.

Reference checked 5 October:
[official NeuroBench leaderboard](https://github.com/NeuroBench/neurobench/blob/main/leaderboard.rst).
Existing queue: queue/aws_primate_admission_20261004T184500Z/manifest.json.

## 2. Turn the industrial lead into a strong controlled result

The first FAS native arm reaches .600 AUROC at 256 events versus the best tested
classical control's .559. Repeat that exact arm on seed 7 and complete the five
AWS neural references before claiming a broader advantage. Report their actual
training work and inference work together. The references are specified
LSTM/Transformer/LRU/S5-style/Mamba-style implementations, not a claim to cover
every tuned state-of-the-art method. FAS is our synthetic benchmark, not an
external leaderboard or customer deployment.

The next native mechanism test is existing R8 (long-timescale initialization).
R0/R1/R3 use their slots but learn roughly 7–8-second half-lives. Typical
item-own event gaps are about 29 seconds: a 7-second half-life retains only 6% of
a write across that median gap. A 100-second half-life retains approximately82%.
This motivates testing retention and binding rather than simply adding slots.

Read out clean-validation NLL, learned horizons, occupancy and declared primary
AUROC. If long horizons survive but useful discrimination does not improve,
investigate item binding and original-write credit across truncated segments.
An identity-aware oracle is a diagnostic only; keep its hidden identities out
of model inputs. R0/R1/R3 have one epoch while the first arm has two: their test
scores do not isolate the effects of pool size or tying against the first arm.

Gate: better early AUROC than the completed relevant references, with the chosen
compute axis no greater and confirmation on a second training seed. Select any
new configuration on development data; further tuning on these already inspected
test scores needs a fresh held-out confirmation. Existing paths:
FAS_BENCHMARK.md, AWS_FAS_REFERENCES.md and the curie recruitment/R8 queues.

## 3. Close the language gap through a small controlled mechanism sequence

At 10M, the native budget-B result is 1.955 BPC at 107.2 TF; tuned LSTM-384 is 1.915
at approximately 106 TF. Budget A is 1.888 versus tuned LSTM-512's 1.826. Wider
p128 did not close the gap. Recent-character taps improve 1.955→1.944 for about
1% additional work, below their promotion gate. Local history remains a supported
diagnosis, but write bandwidth is a hypothesis, not a proved cause.

Use already prepared weight-decay and learning-rate arms, followed by the
k-winner contracts/smoke and matched k1/k2 mechanism comparison. Pool 2/k2 updates
all slots: label it a dense-write diagnostic. Pool 4/k2 retains sparse writes and
is the relevant integrated follow-up. Retain computational time, addressed
state, separate keys/values and counterfactual route learning. Charge second
arrivals, extra writes/deliveries, candidate discovery and learning work.

The pool 4 four-pass mechanism tests cost approximately twice the pool 2 fit;
they do not qualify as budget-B wins just by beating 1.915. Promote only after
validation-supported improvement, then perform a newly named, budget-capped
comparison at the reference's actual work and repeat the selected member on a
second seed. Match causal targets and evaluation context. Choose combinations
only after single-change arms identify benefit; avoid a broad grid.

For an immediate strict Transformer training-compute claim, the native retry
must fit the selected Transformer arm's actual budget: today's tuned comparisons
are approximately 2–4% over budget. A quality lead does not remove that excess.
Finish the existing budget-C 90M controls to adjudicate the 1.800/0.965 PF result;
do not infer a matched win from stronger controls at 4–8× more work.

Existing paths: AWS_CAPACITY_PROGRAM.md, TUNED_BASELINES.md, FINDINGS.md;
queue/aws_kwrite_smoke_20261005T093000Z.txt and its k1/k2 arms.

## Resource allocation and evidence delivery

Complete near-finished public protocols and strong reference comparisons;
screen the specific failure hypotheses above; replicate improvements before
scaling them. Current Mackey-Glass result 14.84 loses to 13.37; another official
30-repeat campaign needs a new configuration supported on development data.
Further width/depth/pass sweeps have lower priority than these targeted tests.

Every finished comparison needs a source-bound result, all-run table, same-unit
quality/resource accounting and explicit win/loss/efficiency-point classification.
Measured serving or energy advantages additionally require trained sparse parity
and physical measurement. This review workspace performed no ML runtime or
training; existing owners retain all admissions and active jobs.
