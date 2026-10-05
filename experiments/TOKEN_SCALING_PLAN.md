# Size the model from measured token scaling

User direction,5October: once the general setup learns well and has passed
fault checks, obtain rough scaling laws to choose model size for training data.
This follows the small-fit mechanism repair stage; it does not admit a grid now.

Freeze the winning tokenization, input/readout initialization, routing/learning
rule, credit horizon, cadence, optimizer and development scoring protocol.
Check that fits learn beyond the train-frequency baseline and that larger
capacity lowers approximation error when sufficiently exposed. Fix accounting,
streaming data allocation and exact continuation before long scaling fits.

## Quantities kept separate

For every completed cell record vocabulary/lexical parameters, shared core
parameters, private receiver parameters, total learned parameters, available
receiver states, selected writes/value deliveries, scored keys and learning
proposals; persistent-state bytes; unique training-data tokens, actual fitting
target presentations and passes; complete fitting work/per-target work and
inference work, measured CPU wall and peakRSS. Initialization frequency counts
also consume training data and are included in the declared data population.

The first width sweep holds depth/heads/pool/credit fixed, so it is one model
family. A separate pool sweep tests capacity beyond activity. Do not collapse
those axes into one dense parameter count or apply a borrowed tokens-per-
parameter constant. Changes to optimizer/credit/protocol start a new sweep.

## Economical first measurement packet

Choose three widths around the successfully debugged integrated fit, and three
training-data budgets (initially64K/256K/1M tokens if the learning curves support
those scales). Use the same fixed number of data passes and a fixed number of
validation checkpoints per pass, with8 evaluation lanes and one sufficiently
large development interval disjoint from public benchmark validation.

Admit the smallest three cells first. Advance promising widths to the next data
budget, retaining enough crossed cells to identify model/data effects. Aim for
nine completed cells for the joint rough fit; no all-knob search. Repeat one
central cell to estimate variation. Stop a cell only for a predefined numerical,
resource or learning failure, retaining its record. Use frozen small-run
checkpoints where continued data/protocol remain comparable.

Fit an exploratory loss surface

    L(N,D) = L_floor + a*(N/N0)^(-alpha) + b*(D/D0)^(-beta),

where N is learned parameter capacity along this fixed-width family and D is
unique data at the fixed pass count. Report uncertainty and residuals; fit a
simpler separate-slope model if the joint parameters are poorly identified.
An error-balanced size relation is N(D) proportional to D^(beta/alpha), with its
constant fitted from these cells. It is not a universal compute-optimal rule.

Use measured c(N)=complete fitting work per target, including readout,
counterfactual probes, cache refresh and optimizer. Given a CPU/work budget,
scan the measured/interpolated size choices with D_presentations*c(N) within
that budget, and choose predicted validation quality. Available capacity and
active cost are separate for this substrate, so dense C proportional to N*D is
not assumed.

Reserve one next-larger data/size point before fitting the law, e.g.4M tokens
if1M is the last fitted scale. Predict its loss and cost, then run it. Its
prediction error determines whether the rule is trustworthy enough to select
the public-benchmark model. A failed extrapolation triggers diagnosis/new fit,
not an unexamined hundred-million-token investment.

No scaling coefficients have been measured for the new tokenized setup yet.
Current8K pilot variants are mechanism comparisons, not a scaling dataset.
