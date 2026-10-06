# Separate factual and alternative-write credit horizons

The completed P24/64K seed6 comparison changes two learning mechanisms together.
Credit16 selects8.033310714422NLL; credit64 selects initialization8.162279794730.
Both use fixed batch64,256updates,131056targets and fourDEVchecks. Longer credit
loses0.128969080308NLL. This is a failure of this coupled learning recipe.

At the four logged checkpoints, credit16 gradient norms are1.718/1.832/5.527/3.772;
credit64 norms are5.691/1.214/26.662/4.499. Teacher weights are60/24/52/28 versus
124/24/180/28. These are sparse checkpoint observations, not full gradient
statistics or measured teacher variance. Sampled sites/RNG paths differ after
the learning change. No causal attribution follows from these four observations.

The current engine uses `args.credit_window` both in `token_features` to detach
numeric state between factual segments and in fitting to choose `credit_end`
and the teacher's inverse-probability utility weight. Thus the completed loss
cannot distinguish instability from longer factual differentiation versus the
changed continuation teacher. Numeric state survives both policies.

The next bounded diagnosis should independently choose factual credit F and
alternative scoring horizon A, comparing (F,A)=(16,16),(64,16),(16,64),(64,64).
Reuse completed diagonal controls and keep the original implementation unchanged.
A separate wrapper/engine must default to the exact original setting, preserve
selected writes, clocks/races, keys-values, numeric state/EOS resets, readout,
actual alternative writes, optimizer batches and initial-inclusive selection.
A changes the declared utility endpoint and weight, not future-token visibility.
The counterfactual replay already starts from detached numeric initial state;
it must stay explicit which losses receive gradients and which weight supplies
score credit. Never truncate the factual horizon silently to make the fit pass.

Required before fitting: source-derived construction, diagonal numeric/update/
RNG parity; identical inference features/state across F/A choices; explicit
initial-state gradient cut/retained-path contract; causal scoring positions;
importance-weight endpoint contract; interruption/resume; bounded integrated
smoke. Charge every realized counterfactual replay, backward and optimizer
operation through complete accounting. No result or speedup is predicted.

Independent off-diagonal64K fits are not admitted by this note. First finish the
live credit64 exact work replay and implement contracted support. Retain P24/
credit16 as the main language member and the completed two-seed1M results.
The reserved4M extrapolation and public-reference objective remain unchanged.


## 64K diagnosis admission after integrated smoke

The corrected actual-driver contracts pass all12 tiny fits, including exact diagonal/default learning and off-diagonal optimizer/state/RNG resume. Both2K off-diagonal smokes learn against common initialization8.871766: F64/A16 selects8.810150, F16/A64 selects8.798182,4,080 fitting targets/8updates/two DEVchecks. Ordinary wall10.626/11.049s and RSS488668/488816KiB. Selected context/memory utility is negative for F64/A16 and positive for F16/A64 in this smoke; retain both results. F64/A16 exact whole-work replay passes; F16/A64 replay is running.

Source-pinned 64K admission waits for both completed smoke exact-state/selection/formula receipts. Run F64/A16 then F16/A64 at256updates/131,056 fitting targets/four DEVchecks, fixed batch64/lr.003/seed6/P24D2H2U4 and the same token intervals. Reuse completed F16/A16 and F64/A64 diagonal controls. This isolates which coupled credit change causes the measured64K loss; it is not a new architecture or proof of a horizon penalty for the family. Each fit900s/1.2GBRSS/5GBVMS/8GiB floor/one thread, derived from283.56s full64K fit and measured smoke RSS with margin. Follow with selected utility and exact complete-work replay for each arm (5400s audit cap, based on completed46.4-minute64K tracing). All jobs sequential through unique one-job run_safe queues. Larger4M remains reserved; public validation untouched. Do not promote an off-diagonal member before comparable quality and complete work are available.
