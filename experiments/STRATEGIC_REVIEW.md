# Family-level scaling: current development decision

The objective is a general-purpose temporal/event substrate spanning language and reasoning, world models, embodiment, continual learning and event analytics. Existing public temporal wins establish a strong front. The next strategic extension is integrated learning on properly tokenized, large sequential data with initially unknown and changing structure.

## Evidence and allocation

R1 already has a completed integrated sparse token-language data/width grid. Use it before creating new variants. `analyze_family_scaling.py` verifies identical source/data identities and every non-axis argument for nine completed seed6 fits, then computes the measured quality envelope and loss-floor-sensitive extrapolation scenarios. It reads result JSONs only; it does not score reserved data or train.

The keyed token model's 1M/4M language gates and this integrated grid have different scoring populations and mechanisms. They supply complementary evidence, not an interchangeable ranking. Curie retains compact-credit/depth ownership. AWS concentrates on complete execution/accounting and preparing the integrated language decision; it does not duplicate the critic programme.

The completed AWS changing-stream controller improves cumulative prediction at depths4/8 but costs more. Its depth4 immediate post-change loss fails the registered replication gate. Retain those results as supporting adaptation evidence; do not expand the constructed-regime controller grid.

## Scaling hypothesis and engineering choices

For a family member with available capacity P, selected activity k and alternative teaching r, charge discovery(P), execution(k), credit(r), optimizer and acquisition. The desired construction increases useful P without proportional increases in these terms. Preserve temporal races, deep persistent event state, separate keys/values, hard routes and counterfactual teaching. Dense key scans and dense optimizer updates can consume the proposed capacity advantage and must be measured.

Under the illustrative law E=A*N^(-alpha)+B*P^(-beta), C=N*P^kappa, optimal error scales as C^(-alpha*beta/(beta+kappa*alpha)). This is an algebraic conditional forecast, not an empirical law for our family. Improve information reuse (alpha), useful representational capacity (beta), and complete-work growth (kappa). Acquisition A_credit pays back only when future work savings exceed A_credit and task quality is maintained. Nonstationary evaluations additionally measure tracking and retention; total historical tokens alone do not describe them.

Forward-observed future-loss contrasts teach credit and allocation estimators. Sampled exact checks anchor them. Counterfactual evaluations should serve routing, credit calibration and future-intervention selection together. This finite observed-outcome loop closes the learning hierarchy.

## Next integrated R1 decision

1. Retain the best measured width recipe and its original credit16 member. Do not impose monotonic capacity exponents on a nonmonotonic width grid.
2. Before admitting a long fit, identify the complete-work bottleneck in this member: candidate discovery, adaptive token head, factual/alternative learning or optimizer. Reuse saved exact receipts; profile only a missing decision-relevant component through a uniquely named guarded R1 queue.
3. Make one diagnosed implementation change while preserving the core mechanisms. Compare on the existing private development protocol, with acquisition and unsuccessful work charged. State the failure and retained mechanisms before substitution.
4. Freeze competing forecasts and selection rules before the reserved larger-data point. Existing reservation and owner gates remain in force. A later natural-stream evaluation must preserve causal order, score before feedback and separate document resets from persistent learning state.
5. Compare with a pinned published Transformer only under matching tokenizer, corpus and scoring protocol. No new dense reference fits. Family superiority requires observed quality/work evidence; the scenarios select informative experiments.

## Executed engineering decision

The guarded P24 profile identifies backward execution as the largest stage:2.547s of4.420s training in an eight-update observation. Inspection identifies repeated per-event vocabulary gathers, each creating a dense vocabulary-gradient accumulation. `batched_token_episodes.py` gathers observed token content once per existing credit window. It changes neither temporal recurrence nor routing, values, state persistence, gradient boundaries or future-write teaching. It is an execution variant, not an architectural substitution.

For a window of W events, embedding width d and vocabulary V, per-event lookup backward materializes W dense embedding-gradient contributions. The batched gather produces one contribution plus the selected-row accumulation. This removes repeated V*d work within the window; dense optimizer work and other core/readout learning remain charged. This is a predicted local work reduction, not a new overall scaling exponent.

The three-update integration gate passes (maximum state9.54e-7, gradients1.20e-7, parameter1.06e-7 and optimizer7.46e-9; identical RNG). The32-update paired pilot reduces observed training18.377→16.145s, with selected DEV8.151419→8.140149. Changes in floating-point gradient summation grow into different hard-route trajectories; do not call the long fit bitwise equivalent.

The completed64K comparison ran sequentially on AWS slot2: `aws_integrated_p24_64k_ordinary_20261012_v1`, then `aws_integrated_p24_64k_batched_20261012_v1`, then source-bound quality/work decision. Same seed6/P24D2H2U4/credit16/8lanes/chunk64/two passes/256updates/four DEV checks;131056fitting targets. Each fit has600s/1.2GBRSS caps and8GiB availability floor, derived from the32-update observations. Initial-inclusive selection. Development gate: candidate selected quality at least as good and training wall at least5% lower. Report training, engine and process wall separately. No public validation or reserved4M access. Completed results determine continuation.


**Completed full64K decision:** ordinary selected8.045605/training149.695s, batched8.132647/131.433s. Gate fails:12.2% training-wall saving with0.087041NLL quality loss. Retain the original member; no scale-up. The complete float64 pair agrees within2.26e-10NLL across its DEV curve and reduces training173.704→140.719s; both select8.105794, worse than the retained float32 reference. Precision stabilizes this execution comparison but fails the useful-quality gate. An earlier gate admitted a seed7 ordinary run before the stronger reference gate took effect; it completed, and the three unstarted confirmation queue files were preserved with a.retired suffix to block their launch. Full paired work includes alternative teaching and optimizer; no per-kernel saving promoted to a quality-matched win.


## Distinct sequential experience at fixed learning work

Completed seed6 uses the original float32 model. Two passes over65536tokens versus one pass over131064tokens produce exactly131056fitting targets and256updates. DEV8.045606→7.803405; training149.695→149.410s. Of the0.242201NLL gain,0.086160 is present in frequency initialization and0.156041 is the difference in learning gain after initialization. Contextual gain over the same TRAIN-feature-mean comparator rises0.227745→0.407795. Data diversity, lane partition and corpus-wrap resets change together; this is a practical exposure-policy improvement, not an isolated causal diversity theorem. No Transformer or nonstationarity claim.

Unchanged seeds7/8 now confirm both exposure arms. scripts/continue_unique_language.py publishes every completed decision and, only after allthree quality gains and contextual utilities are positive, mean learning-gain change is positive and mean training-wall increase≤5%, admits one larger paired point: two passes262144tokens versus one pass524280tokens, both524272fitting targets/1024updates. Same source/configuration/seed6,1800s/1.2GBRSS/8GiBfloor per fit from the measured64K costs; one slot, bounded3h pipeline. Existing public validation and reserved4M stay protected. Next architectural decision follows actual larger quality/work and Curie's compact-credit integration.
