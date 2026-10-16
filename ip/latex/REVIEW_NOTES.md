# Private review notes — outside the application bodies

16 October 2026. These are substantive inventor-editing drafts,not findings of novelty or filing clearance. No application number,filing date,priority claim,assignment,inventor address or signature was fabricated. Tero Keski-Valkama is recorded as inventor on his express statement. Review each eventual claim against the human contribution record.

## What changed

PF-01–03 were rewritten from the existing prose drafts into structured specifications,claims,abstracts and actual editable vector schematics. PF-04–12 are new draft specifications based on the existing portfolio,IDFs and named technical sources. They were not pre-existing complete patent texts. Their drafting date is16October;no earlier priority support is assumed. PF-10 additionally describes the newly implemented versioned/identity-role hierarchy interface;the hardware embodiments are prospective descriptions,not fabricated experimental results.

Each independent claim must have a passage-level support map and a disclosure-date comparison before filing. An implementation source establishes what was recorded,not legal novelty or a human conception date. Functional breadth beyond enabled variants needs expansion or narrowing by the inventor/counsel. Known formulas and primitives are used to explain embodiments,not represented as individually new. No numerical benchmark claim or unmeasured chip result is used to promise an effect across the full claim scope.

## Family-specific scope and sources

### PF-01: Temporal race processing with sparse addressed state and alternative-route learning

Earlier-public US candidate; feature-specific disclosure dates must be checked. Do not treat the same public foundations as novel EP matter.

Support starting points: [IDF-06_us_grace_period_foundations.md](../../ip/inventions/IDF-06_us_grace_period_foundations.md), [PF-01_temporal_race_substrate.md](../../ip/drafts/PF-01_temporal_race_substrate.md).

### PF-02: Technical-event prediction using heterogeneous defective clocks and clock-specific mark distributions

Conditional later-feature EP/US review; competing-risk and cure-model prior art remain relevant.

Support starting points: [IDF-01_race_of_delayed_clocks.md](../../ip/inventions/IDF-01_race_of_delayed_clocks.md), [IDF-02_resolution_safe_continuous_time_models.md](../../ip/inventions/IDF-02_resolution_safe_continuous_time_models.md), [PF-02_heterogeneous_clock_prediction.md](../../ip/drafts/PF-02_heterogeneous_clock_prediction.md).

### PF-03: Recording-resolution-constrained forecasting of quantized event streams

Conditional later-feature EP/US review; all-cell extension must receive its own support/priority audit.

Support starting points: [IDF-02_resolution_safe_continuous_time_models.md](../../ip/inventions/IDF-02_resolution_safe_continuous_time_models.md), [PF-03_resolution_safe_event_models.md](../../ip/drafts/PF-03_resolution_safe_event_models.md).

### PF-04: Addressed predecessor binding with persistent event memory and read-time learning

Mixed: generic KV/cache/race ideas are background; later predecessor/write/read combinations require feature audit.

Support starting points: [IDF-07_predecessor_message_binding.md](../../ip/inventions/IDF-07_predecessor_message_binding.md), [CREDIT_HORIZON_AND_ADDRESSED_MEMORY.md](../../experiments/theory/CREDIT_HORIZON_AND_ADDRESSED_MEMORY.md), [race_tpp_pointer.py](../../experiments/tpp/race_tpp_pointer.py).

### PF-05: Typed measurement-triggered temporal memory for technical monitoring

Generic statistic-valued race memory was earlier public. Review narrower typed/measurement-triggered combination.

Support starting points: [IDF-03_event_native_irregular_series_classifier.md](../../ip/inventions/IDF-03_event_native_irregular_series_classifier.md), [IDF-04_shared_temporal_core_and_cross_cohort_transfer.md](../../ip/inventions/IDF-04_shared_temporal_core_and_cross_cohort_transfer.md).

### PF-06: Local learning through event-transported eligibility and independently updated feedback

Later-feature audit; RTRL, eligibility traces, adjoints and weight mirroring are known primitives.

Support starting points: [IDF-08_dual_learning_on_temporal_substrate.md](../../ip/inventions/IDF-08_dual_learning_on_temporal_substrate.md), [EVENT_LOCAL_CREDIT_AND_VERSIONED_UPDATE_BANKS.md](../../experiments/theory/EVENT_LOCAL_CREDIT_AND_VERSIONED_UPDATE_BANKS.md).

### PF-07: Forward-taught posterior credit for event-routed processors

Later-feature audit; wake-sleep, reweighted wake-sleep, importance sampling and MH are known.

Support starting points: [IDF-08_dual_learning_on_temporal_substrate.md](../../ip/inventions/IDF-08_dual_learning_on_temporal_substrate.md).

### PF-08: Learner-aware counterfactual credit from issued optimizer-state worlds

Later-feature review; optimizer differentiation, control variates and multilevel identities are background.

Support starting points: [OPTIMIZER_AWARE_ROUTE_CREDIT.md](../../experiments/theory/OPTIMIZER_AWARE_ROUTE_CREDIT.md), [MULTISTEP_UPDATE_FUNCTION_CREDIT.md](../../experiments/theory/MULTISTEP_UPDATE_FUNCTION_CREDIT.md), [MULTIHORIZON_FORWARD_TAUGHT_CREDIT.md](../../experiments/theory/MULTIHORIZON_FORWARD_TAUGHT_CREDIT.md).

### PF-09: Dependency-validated delivery of local learning updates

Later-feature audit; compare MVCC, transactional memory, parameter servers and async learning.

Support starting points: [EVENT_LOCAL_CREDIT_AND_VERSIONED_UPDATE_BANKS.md](../../experiments/theory/EVENT_LOCAL_CREDIT_AND_VERSIONED_UPDATE_BANKS.md), [LOCAL_CREDIT_DEPENDENCY_VALIDITY.md](../../experiments/theory/LOCAL_CREDIT_DEPENDENCY_VALIDITY.md).

### PF-10: Protected temporal hierarchy growth with versioned compositional memory

Mixed disclosure and multiple potential inventions/unity risk. New versioned hierarchy branch first included here on16Oct; do not backdate support. Split standalone claims if appropriate.

Support starting points: [PROTECTED_NATIVE_LANGUAGE_GROWTH.md](../../experiments/theory/PROTECTED_NATIVE_LANGUAGE_GROWTH.md), [VERSIONED_PART_BINDINGS.md](../../experiments/theory/VERSIONED_PART_BINDINGS.md), [IDENTITY_ROLE_INTERFACE.md](../../experiments/theory/IDENTITY_ROLE_INTERFACE.md), [identity_role_composition.py](../../sleeping_machines/identity_role_composition.py).

### PF-11: Delay-mapped execution of selective temporal state with local dispatch

Mixed: public note08 and delay/race foundations remain prior disclosure. Hardware translation alone is not a novelty finding.

Support starting points: [IDF-05_selective_state_space_as_delayed_events.md](../../ip/inventions/IDF-05_selective_state_space_as_delayed_events.md), [PORTFOLIO_STRATEGY.md](../../ip/PORTFOLIO_STRATEGY.md), [split_event_heads.py](../../sleeping_machines/split_event_heads.py).

### PF-12: Path-conditioned race-tree output with normalized causal copy selection

Later-feature review; hierarchical softmax, query refinement, races and pointer mixtures are known. Claim their concrete supported combination only.

Support starting points: [r1_race_tree_lm.py](../../experiments/r1_race_tree_lm.py), [PORTFOLIO_STRATEGY.md](../../ip/PORTFOLIO_STRATEGY.md).

## Claim and drawing review

The technical descriptions map FIG.1 reference signs100–150 and FIG.2 signs210–230 to the actual attached schematics. The drawings describe functional arrangements,not transistor-level layouts. Replace or expand them when an embodiment needs state-machine,clock,arbitration or memory-layout detail to be understood. Numeric references are deliberately independent of HDL identifiers. The current claim sets include method,apparatus and storage-medium forms;office-specific eligibility,unity and claim-form requirements still apply. PF-04's industrial duration-binding branch and PF-10's protected-growth versus versioned-memory branches are especially likely to require separate independent claim sets or divisionals. Never add a priority assertion merely because applications share a theme.

For PF-01,check the actual first public appearance of every claimed combination against the existing register and external disclosures. The recorded US anniversaries are review dates,not safe deadlines to wait for. Public foundations must not be relabelled as new EP subject matter. For all later families,verify talks,investor circulation,preprints,repository visibility and earlier-public support;private status today does not undo a public disclosure.

## Hardware leg

The specifications include concrete digital memory,temporal arithmetic,arbitration,dispatch,version and learning-bank embodiments. The companion [hardware drafts](../../hardware/README.md) implement a subset of shared primitives. No FPGA configuration,complete model implementation,physical chip,energy figure or ASIC performance was invented. FPGA results can strengthen evidence of a technical effect but do not establish novelty or substitute for an enabling disclosure. HDL need not be attached in place of the description and drawings. New implementation matter introduced after filing does not automatically receive the earlier priority date ([USPTO](https://www.uspto.gov/patents/basics/apply/provisional-application));technical implementation must be described to support the claimed effect ([EPO G1/19](https://www.epo.org/en/legal/official-journal/2021/09/a77)).

Commercial grouping continues to follow the portfolio strategy:platform/routing(PF-01),prediction(PF-02),recording-resolution protection(PF-03),binding memory(PF-04),typed sensors(PF-05),local traces(PF-06),posterior credit(PF-07),learner-world credit(PF-08),update runtime(PF-09),growth/hierarchy(PF-10),execution(PF-11),and output engine(PF-12). Packaging a common disclosure does not guarantee unity or independently enforceable claims.
