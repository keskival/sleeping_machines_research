# Acquired history and data scaling (R1)

The three-seed frozen transfer result establishes better input-dependent prediction after more acquisition. Its zero-feature readout intervention does not isolate persistent history. The next diagnostic separates acquisition gain from reliance on stored history, using completed weights rather than another fit.

Let D denote acquired training targets, f_D the frozen predictor and R_D^h its mean negative log likelihood when persistent state is retained for h observed tokens. Let h=∞ denote the factual stream; a finite-h intervention clears receiver memories, arrival records, seen flags and feedback context at block boundaries, preserves stream position, and uses the same observed tokens and route random sequence. No targets enter the state. This changes subsequent routes through changed keys and context: those changes are consequences of the intervention, not a route-controlled ablation.

Define signed history value V_D(h)=R_D^h−R_D^∞ and acquisition gain G_h=R_small^h−R_large^h. Then exactly

    G_∞−G_h = V_large(h)−V_small(h).

A positive right side means that more acquisition increases the measured contribution of retained history under this intervention. A negative side means the acquisition improvement is stronger in the restricted-history stream. Report both rather than treating either sign as a family verdict. At h=16, the diagnostic tests history carried beyond the current credit-window boundary; h=64 tests beyond the optimizer chunk. Neither numerical-state retention nor its measured usefulness implies differentiation through the entire past.

## Information interpretation

For a genuinely restricted predictor q(Y|Z), where Z contains the current input and allowed recent history, and a full predictor p(Y|Z,H), the population log-loss identity is

    R(q)−R(p) = I(Y;H|Z)
                + E KL(P(Y|Z) || q(Y|Z))
                − E KL(P(Y|Z,H) || p(Y|Z,H)).

This follows by adding and subtracting the two conditional entropies. More data can improve both conditional estimates and their approximation to the task-relevant history statistic. The information term explains why a large-data advantage can become visible through learned predictive use of history rather than merely through more available state. The frozen reset predictor is not separately fitted to restricted histories, so its excess-risk term can increase through the intervention. A positive reset loss is evidence of factual reliance, not an estimate of conditional mutual information.

## Development decision

Score the same first four already-used transfer development panels for three seeds and two acquisition sizes. Reproduce factual panel scores exactly before interpreting resets. If V_large(16) is positive and increases consistently with acquisition, prioritize efficient longer-lived state credit and its forward-teaching interface. If it is near zero or negative, diagnose feedback, memory retrieval and training exposure before investing in a larger memory horizon. Either outcome directs model engineering; the public benchmark stays sealed. Report signed per-seed differences and the acquisition/history interaction, with all diagnostic work charged separately from training.

This experiment does not measure nonstationary-regime adaptation or a scaling exponent. Those require their own causal online protocol; the present test identifies which information path is responsible for the acquired-data improvement.
