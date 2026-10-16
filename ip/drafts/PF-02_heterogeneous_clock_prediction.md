# PF-02 — Technical-event prediction using heterogeneous defective clocks and clock-specific mark distributions

CONFIDENTIAL — inventor working draft, 15 October 2026. Inventor: **Tero Keski-Valkama**, according to his express statement that he made the inventions with AI-agent support. Applicant/assignee: to be determined from executed ownership documents. No filing, priority claim or declaration is made here. Example claims require prior-art, disclosure, support and jurisdiction review. Figure descriptions are instructions for drawings, not finished drawings. Implementation status is stated separately from proposed embodiments.

## Disclosure route

Recorded later subject matter: IDF-01a–e and IDF-02a. Conditional EP/US review; generic competing-risk mathematics and earlier public race disclosures remain background.

## Abstract

A forecasting apparatus encodes technical events in persistent temporal state and parameterizes competing clocks with respective mark distributions. At least one delayed clock has a learned non-firing probability. An optional soft-window clock has analytically integrated survival. A continuously evolving state can parameterize another clock during silent intervals. The apparatus evaluates a joint next-event time and mark law from the competing survivals and hazard-weighted mark distributions and uses the result for monitoring or scheduling.

## Technical field, background and detailed description

The apparatus processes event records from a machine, communication network or sensor installation. The problem is to represent immediate events, delayed events, broad firing windows and time-dependent event types without forcing every delay component to fire.

After an observed event, an encoder updates h. Clock m has survival S_m(tau|h), hazard lambda_m(tau|h), and mark probabilities p_m(k|h,tau). Conditional independence of the modeled clock timers gives S(tau)=product_m S_m(tau), and marked density f_k(tau)=S(tau)*sum_m lambda_m(tau)*p_m(k|h,tau). The event likelihood trains both timing and marks. The mark need not be independent of elapsed time because the likely winning clock changes with time.

A defective delayed clock has S_m(tau)=1-pi_m+pi_m*S_delay(tau), where pi_m lies in [0,1]. Its hazard is pi_m*f_delay(tau)/S_m(tau). At least one proper background clock can ensure eventual firing; alternatively the output explicitly permits no further event. A log-normal conditional delay is one embodiment.

For a window clock, let a<b, s>0 and d(u)=sigmoid((u-a)/s)-sigmoid((u-b)/s) for u>=0. Define F(u)=s*(softplus((u-a)/s)-softplus((u-b)/s)). Its normalization is Z=b-a-F(0). Conditional survival is ((b-a)-F(tau))/Z. Stable log-domain evaluation avoids subtracting nearly equal ordinary floating-point quantities. Defective survival is again 1-pi+pi*S_window. Window width and edge scale are constrained positive. Initial edges may be derived from training-gap components and bounded around their anchors. A data rule may enable windows only where the training distribution has distinct components; this is not per-test tuning.

A state-clock variant transports z(tau)=exp((-r+i*omega)*tau)*z_0 and reads a positive hazard and normalized marks from z(tau). The state-clock compensator can require deterministic quadrature. Analytic clock components do not make every state-clock likelihood closed form. The device outputs a no-event probability, time/type forecast, alarm or resource schedule.

## Example claims

1. A method of operating a technical-event monitoring apparatus, comprising updating persistent state from observed events; parameterizing multiple competing clocks and a respective event-type distribution for each clock from the state; including a delayed clock having survival equal to a non-firing mass plus a firing probability multiplied by a conditional delay survival; evaluating a joint next-event time and type density from a product of clock survivals and a sum of hazard-weighted type probabilities; and producing a monitoring output from the evaluated density.

2. The method of claim 1, wherein a proper background clock provides eventual firing while the delayed clock retains positive non-firing mass.

3. The method of claim 1, wherein a window clock has a density proportional to a difference of logistic functions and survival evaluated through integrated softplus terms.

4. The method of claim 3, wherein window position and width are anchored to components of a training-gap distribution and constrained to bounded ranges around the anchors.

5. The method of claim 1, wherein a clock hazard is read from memory transported continuously during an interval without events, and its cumulative hazard is evaluated by deterministic quadrature.

6. The method of claim 1, wherein event-type probabilities vary with elapsed time through changing contributions of competing clocks.

7. A monitoring system comprising an event input, temporal-state storage and a processor configured to perform the method of claim 1.

8. A non-transitory computer-readable storage medium storing instructions implementing the method of claim 1.

## Drawings, support and review notes

Figures: (1) state encoder and heterogeneous clock bank; (2) defective clock survival with nonzero asymptote; (3) elapsed-time-dependent mark law. Sources: IDF-01, IDF-02; experiments/theory/RACE_OF_DELAYED_CLOCKS_20261006.md; experiments/tpp/race_tpp_v*.py. Numerical embodiments have completed EasyTPP evidence, but benchmark scores are not claim limitations. Search competing-risk/cure models, neural point processes and intensity-free mixtures. Do not characterize all intensity-free prior art as time/type independent.
