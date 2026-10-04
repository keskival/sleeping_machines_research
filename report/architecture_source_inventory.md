# Source navigation for the whole family

AST inventory; file existence and class names do not establish validated capability.
See [semantic family review](model_family_inventory.md) and [source hashes/methods](architecture_source_inventory.json).

| Module | Classes / top-level functions | Declared source summary |
| --- | --- | --- |
| [sleeping_machines/__init__.py](../sleeping_machines/__init__.py) | Package |  |
| [sleeping_machines/addressed_event_heads.py](../sleeping_machines/addressed_event_heads.py) | AddressedEventState, AddressedEventHeads | Native addressed content/time events with persistent independent race heads. |
| [sleeping_machines/affine_packets.py](../sleeping_machines/affine_packets.py) | compose, prefix, pooled_state_scan | Affine state and weighted state-pooling summaries in linear event work. |
| [sleeping_machines/batched_addressed_fit.py](../sleeping_machines/batched_addressed_fit.py) | BatchedTemporalRoute | Vectorize independent dense-packet clips without changing native route credit. |
| [sleeping_machines/batched_episodes.py](../sleeping_machines/batched_episodes.py) | LaneRace | Episode-batched native core with gradients: many independent episodes as lanes of one pass (THEORY §405). |
| [sleeping_machines/batched_evolution_offset_fit.py](../sleeping_machines/batched_evolution_offset_fit.py) | forward | Frozen native batched kernel with reception-phase offset only (theory90). |
| [sleeping_machines/bounded_score_sensitivity.py](../sleeping_machines/bounded_score_sensitivity.py) | bounded_score_bridge, smooth_slope | Bounded monotone score bridge; alpha0 exactly retains the old hard clamp. |
| [sleeping_machines/bridge_addressed_events.py](../sleeping_machines/bridge_addressed_events.py) | BridgeAddressedEvents | Bounded-score native reference; ordinary sparse inference and actual state. |
| [sleeping_machines/bridge_batched_episodes.py](../sleeping_machines/bridge_batched_episodes.py) | LaneRace | Bounded-score sibling of batched_episodes; same physical races and full writes. |
| [sleeping_machines/causal_contexts.py](../sleeping_machines/causal_contexts.py) | word_codes_before | Contexts for predicting x[t], measurable from x[:t] only. |
| [sleeping_machines/causal_language_shadow.py](../sleeping_machines/causal_language_shadow.py) | LaneRace | All-token native shadow lanes with detached persistent entering state (note108). |
| [sleeping_machines/causal_language_shadow_cached_prefix.py](../sleeping_machines/causal_language_shadow_cached_prefix.py) | LaneRace | Unchanged native races with detached causal token-boundary snapshots (118). |
| [sleeping_machines/causal_language_shadow_compact_suffix.py](../sleeping_machines/causal_language_shadow_compact_suffix.py) | compact_suffix_returns | Unadmitted CPU-only compact suffix prototype; no active driver imports this. |
| [sleeping_machines/causal_language_shadow_rng.py](../sleeping_machines/causal_language_shadow_rng.py) | LaneRace | All-token native shadow lanes with detached persistent entering state (note108). |
| [sleeping_machines/causal_language_shadow_winner_reuse.py](../sleeping_machines/causal_language_shadow_winner_reuse.py) | LaneRace | Native causal shadow kernel with factual winner recording (note115). |
| [sleeping_machines/checkpointed_addressed_replay.py](../sleeping_machines/checkpointed_addressed_replay.py) | BatchedTemporalRoute | AWS prefix-checkpoint replay kernel. Arithmetic mirrors the frozen batched core; snapshots are detached no-grad shadow inputs, never factual producer-credit truncation. |
| [sleeping_machines/clock_feature_event_heads.py](../sleeping_machines/clock_feature_event_heads.py) | ClockFeatureState, ClockFeatureEventHeads, ClockFeatureLanguageModel | Native sparse events with fast delay-computed features per independent head. |
| [sleeping_machines/clock_feature_races.py](../sleeping_machines/clock_feature_races.py) | ContentAndClockRaces | Shared matches drive a content race and independent clock-only policies. |
| [sleeping_machines/clock_noise_episodes.py](../sleeping_machines/clock_noise_episodes.py) | ClockRace | Isolated native clock-noise prototype; existing kernels/defaults untouched. |
| [sleeping_machines/clock_noise_law.py](../sleeping_machines/clock_noise_law.py) | check_temperature, log_normalizer, clock_from_unit_noise, race, moments, factorized_clock_credit | Mean-normalized clock-noise control for a hard exponential race (stdlib). |
| [sleeping_machines/clock_preserving_temperature.py](../sleeping_machines/clock_preserving_temperature.py) | ClockPreservingTemperatureHeads | Frozen categorical calibration preserving the exponential first-time law. |
| [sleeping_machines/compact_context_predictor.py](../sleeping_machines/compact_context_predictor.py) | Predictor | Portable dense 33-anchor readout of an unchanged native query context. |
| [sleeping_machines/compact_tied_inference.py](../sleeping_machines/compact_tied_inference.py) | CompactTiedSparseWorker | Prepared CPU inference for tied maps and private temporal state. |
| [sleeping_machines/compiled_episodes.py](../sleeping_machines/compiled_episodes.py) | _rotate, _transport, layer_step, compiled_step, compiled_logits | Compiled episode-batched native core: batched_episodes' arithmetic with each layer step fused (THEORY §412). |
| [sleeping_machines/conditional_branch_content_credit.py](../sleeping_machines/conditional_branch_content_credit.py) | conditional_branch_objective | Single-site branch-content averaging with correctly scaled route credit. |
| [sleeping_machines/context_addressed_memory.py](../sleeping_machines/context_addressed_memory.py) | ContextMemoryState, _ContentWithMemory, ContextAddressedNativeModel | Context-addressed persistent memory for the native temporal core (THEORY §390 addendum). |
| [sleeping_machines/count_carrying_language.py](../sleeping_machines/count_carrying_language.py) | CountCarryingNativeModel | Count-carrying receivers over the native addressed temporal core (THEORY §§376-380). |
| [sleeping_machines/count_composed_stream.py](../sleeping_machines/count_composed_stream.py) | PositionedState, CountComposedModel | Generic count-receiver composition over any stream model (labelled diagnostics; THEORY §§386-387). |
| [sleeping_machines/count_continuation.py](../sleeping_machines/count_continuation.py) | _triples, fit_stream_continuation_counts, eval_stream_continuation_counts | Kneser-Ney continuation statistics for the lower cascade orders (THEORY §393 result). |
| [sleeping_machines/count_escape_gate.py](../sleeping_machines/count_escape_gate.py) | CountMessage, EscapeGate, GatedCountCarryingNativeModel, GatedCountComposedModel | Count-conditioned base and state-dependent escape for count receivers (THEORY §379 caveat, §387.3, §389). |
| [sleeping_machines/decay_timing_reference.py](../sleeping_machines/decay_timing_reference.py) | DecayTimingReference | Generator-informed two-trace diagnostic; no learned routes or supremacy claim. |
| [sleeping_machines/depth_growth.py](../sleeping_machines/depth_growth.py) | grow_event_encoder | Grow an event encoder by identity residual blocks with live output teachers. |
| [sleeping_machines/dilated_delay_taps.py](../sleeping_machines/dilated_delay_taps.py) | TappedNativeState, TappedMix, TappedNativeStreamLanguageModel | Learned dilated delay taps for the native temporal core (THEORY §391). |
| [sleeping_machines/energy.py](../sleeping_machines/energy.py) | Profile | Energy estimates from measured operation counts. |
| [sleeping_machines/episodic_race_language.py](../sleeping_machines/episodic_race_language.py) | EpisodicState, EpisodicRaceLanguageModel | Per-position KV race retrieval inside the sparse temporal backbone. |
| [sleeping_machines/event_history_control.py](../sleeping_machines/event_history_control.py) | EventHistoryControl | Diagnostic learned decoder over causal three-mark history; no race mechanisms. |
| [sleeping_machines/event_memory.py](../sleeping_machines/event_memory.py) | segmented_memory, affine_prefix, linear_memory | Segmented affine event memory: reference and linear-work implementations. |
| [sleeping_machines/event_query.py](../sleeping_machines/event_query.py) | ObservedPrefix | Explicit observed-prefix queries. Targets never enter the event batch. |
| [sleeping_machines/event_state.py](../sleeping_machines/event_state.py) | EventStateBlock, EventStateEncoder, CoalescedEventStateEncoder | Learned signed event-state blocks with nonlinear payloads and winning clocks. |
| [sleeping_machines/evidence_memory.py](../sleeping_machines/evidence_memory.py) | ConditionalEvidence, RelativeRouteMemory | Local sufficient statistics and hard pointer races shared by task adapters. |
| [sleeping_machines/evolution_offset_heads.py](../sleeping_machines/evolution_offset_heads.py) | EvolutionOffsetHeads | Reception phase calibration while retaining physical temporal evolution. |
| [sleeping_machines/exact_pi_race.py](../sleeping_machines/exact_pi_race.py) | ExactPiRoute | Exact-pi linearized race credit (THEORY §400). |
| [sleeping_machines/factorized_race.py](../sleeping_machines/factorized_race.py) | FactorizedRace | Factorized race credit: categorical choice and common first-time clock (THEORY note 92 remedy 2; §402 correction). |
| [sleeping_machines/fast_native_core.py](../sleeping_machines/fast_native_core.py) | FastNativeCoreMixin, FastNativeStreamLanguageModel | Batched training path for the native temporal core: same arithmetic, fewer operations (THEORY §393 scale). |
| [sleeping_machines/folded_polynomial_readout.py](../sleeping_machines/folded_polynomial_readout.py) | FoldedPolynomialReadout | Frozen polynomial decoder with fit-only normalization folded into weights. |
| [sleeping_machines/fused_context_reader.py](../sleeping_machines/fused_context_reader.py) | FusedContextReader | Inference-only compilation of a late-projected addressed reader. |
| [sleeping_machines/head_conditioned_depth.py](../sleeping_machines/head_conditioned_depth.py) | grow_head_conditioned | Identity depth growth in the metric of the existing class observer. |
| [sleeping_machines/historical_write_credit.py](../sleeping_machines/historical_write_credit.py) | PackedFeatureBank, HistoricalKeyMatch, HistoricalValueCredit | Fixed-feature delayed write adjoints; these are declared local surrogates. |
| [sleeping_machines/historical_write_race_language.py](../sleeping_machines/historical_write_race_language.py) | HistoricalWriteState, HistoricalWriteRaceLanguageModel | Parallel sparse race heads with compact historical write eligibility. |
| [sleeping_machines/indexed_episodic_race_language.py](../sleeping_machines/indexed_episodic_race_language.py) | IndexedEpisodicState, IndexedEpisodicRaceLanguageModel | Per-position KV race retrieval inside the sparse temporal backbone. |
| [sleeping_machines/joint_outcome_race_query.py](../sleeping_machines/joint_outcome_race_query.py) | OutcomeRaceState, JointOutcomeRaceQuery | Protected causal outcomes, two learned key races and a bilinear query. |
| [sleeping_machines/joint_race_credit.py](../sleeping_machines/joint_race_credit.py) | _inputs, enumerated_credit, sampled_credit | Detached LOCAL joint winner/time credit reference; no model installs this. |
| [sleeping_machines/language_memory.py](../sleeping_machines/language_memory.py) | initialize_language_memory | Token-unit initialization ablations for the existing event-language blocks. |
| [sleeping_machines/late_projected_context_memory.py](../sleeping_machines/late_projected_context_memory.py) | LateProjectedMemoryState, LateProjectedContextModel | Defer linear value projection until addressed-memory retrieval. |
| [sleeping_machines/native_bounded_score_diagnostic.py](../sleeping_machines/native_bounded_score_diagnostic.py) | NativeBoundedScoreDiagnostic | Native bounded emitter prototype with exact zero bridge and paid raw taps. |
| [sleeping_machines/native_stream_language.py](../sleeping_machines/native_stream_language.py) | NativeLanguageState, NativeStreamLanguageModel | Token content adapter for the native addressed temporal core; no KV bank. |
| [sleeping_machines/native_window_diagnostic.py](../sleeping_machines/native_window_diagnostic.py) | NativeWindowDiagnostic | Frozen one-site native reception intervention; no training credit installed. |
| [sleeping_machines/nuisance_readout.py](../sleeping_machines/nuisance_readout.py) | paired_view_metric, paired_logit_risk | Fitting-only paired-view geometry for an affine event-query classifier. |
| [sleeping_machines/numpy_addressed_inference.py](../sleeping_machines/numpy_addressed_inference.py) | NumpyAddressedInference | Inference-only numeric port of the one-source addressed native temporal core. |
| [sleeping_machines/objectives.py](../sleeping_machines/objectives.py) | categorical_nll, hazard_nll | Query-time objectives on common natural scores; no hidden time grid. |
| [sleeping_machines/observer_conditioned_depth.py](../sleeping_machines/observer_conditioned_depth.py) | ObserverUnits, ObserverStateBlock | Identity growth with normalized features and directional output units. |
| [sleeping_machines/operation_audit.py](../sleeping_machines/operation_audit.py) | OperationAudit | Executed ATen arithmetic by stage, including backward and Adam. |
| [sleeping_machines/packed_episodic_race_language.py](../sleeping_machines/packed_episodic_race_language.py) | PackedKVBank, PackedBucketMap, PackedEpisodicState, PackedEpisodicRaceLanguageModel | Lossless storage for the content-indexed sparse temporal language model. |
| [sleeping_machines/paired_route_credit.py](../sleeping_machines/paired_route_credit.py) | alternative_probabilities, paired_choice_credit, sample_alternative | Finite paired choice credit; actual outcome loss differences, bounded proposal. |
| [sleeping_machines/parallel_head_race_language.py](../sleeping_machines/parallel_head_race_language.py) | ParallelHeadState, ParallelHeadRaceLanguageModel | Parallel content-bearing race heads with timestamped separate channels. |
| [sleeping_machines/parallel_stream_language.py](../sleeping_machines/parallel_stream_language.py) | ParallelEventLanguageModel | Causal parallel fitting and precise clocks for the persistent language model. |
| [sleeping_machines/phase_memory.py](../sleeping_machines/phase_memory.py) | PhaseMemory | A learned periodic event state and a hard race of class clocks. |
| [sleeping_machines/polynomial_query_readout.py](../sleeping_machines/polynomial_query_readout.py) | PolynomialQueryReadout | Small query-local affine/quadratic residual over a frozen event encoder. |
| [sleeping_machines/prefix_tokenizer.py](../sleeping_machines/prefix_tokenizer.py) | CompletePrefixTokenizer, ByteTimedTokenModel, PrefixLanguageState | Train-only, complete prefix dictionaries for causal language experiments. |
| [sleeping_machines/prepacked_sparse_inference.py](../sleeping_machines/prepacked_sparse_inference.py) | _PreparedView, PrepackedSparseWorker | A fixed-weight CPU worker for the existing winner-only evaluator. |
| [sleeping_machines/quadratic_native_readout.py](../sleeping_machines/quadratic_native_readout.py) | QuadraticNativeReadout | Zero-nested standard degree-2 readout of the current native vector. |
| [sleeping_machines/query_feature_inference.py](../sleeping_machines/query_feature_inference.py) | QueryFeatureInference | Demand-only final-query features from the frozen native numeric port. |
| [sleeping_machines/query_only_native_readout.py](../sleeping_machines/query_only_native_readout.py) | QueryOnlyLinear | Inference-only query admission for the unchanged native affine classifier. |
| [sleeping_machines/race_language.py](../sleeping_machines/race_language.py) | WinnerRace, RaceStreamState, RaceLanguageModel | Causal, indexed race retrieval alongside the unchanged language carrier. |
| [sleeping_machines/race_transformer.py](../sleeping_machines/race_transformer.py) | RaceTransformerState, RaceTransformer | Run a trained causal Transformer as an asynchronous race-attention event stream (THEORY §397). |
| [sleeping_machines/race_window.py](../sleeping_machines/race_window.py) | bounded_delay, residual_cutoff, residual_probability, conditional_residual | Exact conditional arrival law for a fixed deadline after an exponential race. |
| [sleeping_machines/readout_absorption.py](../sleeping_machines/readout_absorption.py) | project_component, optimize_head, absorption_certificate | Fold a parallel logit component into an existing affine event readout. |
| [sleeping_machines/readout_calibration.py](../sleeping_machines/readout_calibration.py) | condition_readout | Fit-only readout conditioning, including support of static metadata. |
| [sleeping_machines/relative_race_window.py](../sleeping_machines/relative_race_window.py) | relative_deadline, expected_relative_receivers, maximum_added_physical_delay | Race-scaled reception support; physical transport remains a separate operator. |
| [sleeping_machines/repeated_arrival_race_language.py](../sleeping_machines/repeated_arrival_race_language.py) | RepeatedArrivalState, RepeatedArrivalRaceLanguageModel | Independent sparse heads with shared-match repeated temporal arrivals. |
| [sleeping_machines/repeated_temporal_race.py](../sleeping_machines/repeated_temporal_race.py) | PoissonTemporalRoute | Shared-rate Poisson arrivals and a conserved local counterfactual teacher. |
| [sleeping_machines/replay_priority_sampling.py](../sleeping_machines/replay_priority_sampling.py) | orders, inclusion, parameter_variance, sample | Bounded exact inclusion probabilities for sequential weighted replay sampling. |
| [sleeping_machines/rotating_memory.py](../sleeping_machines/rotating_memory.py) | rotate_pairs, rotating_memory | Addressed signed temporal memory, with the real mean as its zero-phase case. |
| [sleeping_machines/rule_seed_event_heads.py](../sleeping_machines/rule_seed_event_heads.py) | RuleSeedEventHeads | Independent rule/embedding sharing diagnostic; inherited event dynamics intact. |
| [sleeping_machines/selective_stream_language.py](../sleeping_machines/selective_stream_language.py) | SelectiveEventLanguageModel | Input-known write/forget gates on the precise event-language state. |
| [sleeping_machines/serial_residual.py](../sleeping_machines/serial_residual.py) | SerialResidualEncoder | Serial event-depth extension with an explicitly owned correction query. |
| [sleeping_machines/shadow_lanes.py](../sleeping_machines/shadow_lanes.py) | shadow_losses | Shadow lanes: every counterfactual replay of an episode in one lane-batched, gradient-free pass (THEORY §404.2). |
| [sleeping_machines/shared_event.py](../sleeping_machines/shared_event.py) | RaceLayer, SharedEventModel | Shared event backbone: addressed temporal state, hard races and local loser credit. |
| [sleeping_machines/silence_burst.py](../sleeping_machines/silence_burst.py) | silence_bursts, finite_timeout_risk | Causal event-only silence-timeout accumulation, a scoped reference primitive. |
| [sleeping_machines/sim.py](../sleeping_machines/sim.py) | Event, Engine | Discrete-event engine for the Sleeping Machines experiments. |
| [sleeping_machines/sparse_inference.py](../sleeping_machines/sparse_inference.py) | SparseStepper | Winner-only native inference with cached key reads (THEORY §414). |
| [sleeping_machines/sparse_race_language.py](../sleeping_machines/sparse_race_language.py) | TemporalRoute, TemporalUnit, SparseLanguageState, SparseRaceLanguageModel | Integrated sparse temporal language network, without a dense carrier. |
| [sleeping_machines/sparse_training.py](../sleeping_machines/sparse_training.py) | _rotate, _transport, _unit, sparse_layer_step, compiled_sparse_step, sparse_train_logits | Winner-plus-sampled-alternative training for the batched native core (THEORY §416). |
| [sleeping_machines/split_event_heads.py](../sleeping_machines/split_event_heads.py) | SplitTemporalUnit, SharedSourcePools, CommonSourceSeed, SplitEventHeads | Experimental protected/temporal state with private or shared addressed rules. |
| [sleeping_machines/state_write_credit.py](../sleeping_machines/state_write_credit.py) | StateWriteRoute, StateCreditState, StateCreditEventHeads | Hard temporal races with addressed memory/arrival boundary credit. |
| [sleeping_machines/statistic_race_credit.py](../sleeping_machines/statistic_race_credit.py) | predictive, delivery_credit, expected_write_gain | Statistic-valued LOCAL delivery and one-next-event write credit contracts. |
| [sleeping_machines/statistic_race_memory.py](../sleeping_machines/statistic_race_memory.py) | StatisticRaceState, StatisticRaceNativeModel | Statistic-valued race memory over learned keys (THEORY §§382-383, §392). |
| [sleeping_machines/statistic_race_sampled.py](../sleeping_machines/statistic_race_sampled.py) | SampledStatisticRaceNativeModel | Sampled race writes for statistic-valued race memory (THEORY §392 collapse repair). |
| [sleeping_machines/statistic_race_top.py](../sleeping_machines/statistic_race_top.py) | TopStatisticRaceNativeModel | Top-placed statistic-valued race memory (THEORY §392 placement correction). |
| [sleeping_machines/stream_language.py](../sleeping_machines/stream_language.py) | LanguageStreamState, StreamingEventLanguageModel | Persistent language stream using the shared signed event-state primitive. |
| [sleeping_machines/temporal_modes.py](../sleeping_machines/temporal_modes.py) | WinningTemporalModes | Stable signed temporal modes for addressed winning event updates. |
| [sleeping_machines/temporal_window.py](../sleeping_machines/temporal_window.py) | temporal_window | Compact C1 temporal integration using event-driven exponential moments. |
| [sleeping_machines/threshold_spike_train.py](../sleeping_machines/threshold_spike_train.py) | threshold_spike_train | Exact event-driven leaky threshold trains for piecewise constant currents. |
| [experiments/e30_minsky.py](../experiments/e30_minsky.py) | Net, Node, Delay, Or, And, Veto, Probe | E30: a two-counter (Minsky) machine wired from the temporal operator basis (THEORY §59). |
| [experiments/e53_depth3.py](../experiments/e53_depth3.py) | Net | E53: order among three parts needs depth 3; the expanded candidate basis is paid for by activity, not size (THEORY §84). |
| [experiments/e54_chains.py](../experiments/e54_chains.py) | Syn, Net | E54: deeper order by chain composites with synapses grown on activity (THEORY §85), verified against dense weights. |
| [experiments/e61_race_attention.py](../experiments/e61_race_attention.py) | RaceAttention | E61: is attention trainable by local credit? Associative recall with a learned query-key map (THEORY §96, §97). |
| [experiments/e120_shared_tasks.py](../experiments/e120_shared_tasks.py) | Example, Task | Bounded data adapters for the shared event model, with explicit prefix queries. |
| [experiments/e173_causal_language.py](../experiments/e173_causal_language.py) | CausalWordOrder | Corrected, bounded-memory native language rerun with explicit expert arms. |
| [experiments/e174_aligned_language_baselines.py](../experiments/e174_aligned_language_baselines.py) | score, main | Rescore saved E64 controls on exactly E173's target positions, including tail. |
