# PF-03 — Recording-resolution-constrained forecasting of quantized event streams

CONFIDENTIAL — inventor working draft, 15 October 2026. Inventor: **Tero Keski-Valkama**, according to his express statement that he made the inventions with AI-agent support. Applicant/assignee: to be determined from executed ownership documents. No filing, priority claim or declaration is made here. Example claims require prior-art, disclosure, support and jurisdiction review. Figure descriptions are instructions for drawings, not finished drawings. Implementation status is stated separately from proposed embodiments.

## Disclosure route

IDF-02b–d recorded after the public boundary; verify earlier and external disclosures. Separate from PF-02 because the construction can apply to other event models.

## Abstract

An event-forecasting processor receives a recording-cell width and modifies a continuous-time event law consistently with that width. Hazard and mark functions are held at a boundary value over a sub-cell interval and survival is modified to integrate the held hazard. Training perturbs a target time within its recording cell while keeping the conditioning event history in recorded form. A paired recorded/dequantized score can gate model selection. This addresses timestamp-grid exploitation without requiring a changed deployment history representation.

## Technical field, background and detailed description

Logging equipment records time at finite resolution. An unconstrained density can sharpen at repeated grid locations and obtain a favorable numerical score unrelated to useful timing prediction. The invention concerns an explicit predictive law and training arrangement, not the general idea that measurements have finite precision.

Let the recording resolution be c>0, the unmodified hazard h_0(tau), survival S_0(tau) and mark law p_0(k|tau). In a first-cell embodiment, define h_c(tau)=h_0(c) for 0<=tau<c and h_c(tau)=h_0(tau) thereafter. Hold the mark law at p_0(k|c) on the same first interval. Compute log S_c(tau)=-h_0(c)*min(tau,c)+log S_0(max(tau,c))-log S_0(c). The pieces agree at c, give S_c(0)=1, and correspond to the actual modified hazard. Apply this per clock if a race engine is used. Holding a hazard without changing survival consistently is not this embodiment.

The construction protects the first recording interval; it does not by itself guarantee that no later grid cell can contain a narrow peak. Complementary scale floors, bounded window edges, bounded modal rates or an explicit all-cell integrated law address broader grid behavior. A generalized all-cell version freezes hazard/marks at a representative point within each cell and accumulates the integrated hazard of completed cells; this is a described extension requiring its own disclosure/support review.

In target-only training, a perturbation distribution is defined by the timestamp quantizer, including its rounding convention and special handling of tied events. The scored target delay is sampled within that cell. Encoder histories remain exactly the recorded histories seen at deployment. Noise must not reverse an already-declared event order or introduce future observations. A separate embodiment uses actual cell probability mass rather than a point density and labels the scoring protocol accordingly.

A validation gate scores the same selected model on recorded targets and within-cell perturbations under a fixed conditioning policy. A score change above a declared tolerance triggers review or rejection. The tolerance is not a universal theorem. The technical output is a forecast or alert from a model whose time resolution is tied to the acquisition device.

## Example claims

1. A computer-implemented method for forecasting events recorded by a device with temporal resolution c, comprising receiving a recording-resolution parameter; modifying a continuous-time hazard to equal its value at c for elapsed times below c; modifying a mark distribution on the same interval; computing survival by integrating the modified hazard; and producing a forecast or monitoring signal using the modified event law.

2. The method of claim 1, wherein log survival equals -h_0(c)*min(tau,c)+log S_0(max(tau,c))-log S_0(c).

3. The method of claim 1, further comprising training with a target delay perturbed within the corresponding recording cell while keeping conditioning event histories in recorded form.

4. The method of claim 3, wherein the perturbation respects a specified quantizer rounding rule and a tied-event ordering convention.

5. The method of claim 1, wherein delayed-clock dispersion or window-edge scale is lower-bounded as a function of c.

6. The method of claim 1, wherein temporal memory rotation and decay are bounded in relation to c.

7. The method of claim 1, further comprising computing paired recorded and within-cell-perturbed validation scores and selecting the deployed model using their difference.

8. A monitoring apparatus comprising timestamped event storage and a processor configured to perform the method of claim 1.

9. A non-transitory computer-readable storage medium storing instructions implementing the method of claim 1.

## Drawings, support and review notes

Figures: (1) recorded cells and held first-interval hazard; (2) continuous survival across c; (3) training with recorded histories and perturbed targets. Sources: IDF-02; experiments/tpp/check_cell_held.py and grid_audit.py. Dequantization, interval censoring and quantized-observation likelihoods are strong prior-art targets. A standalone audit claim is lower priority than claims to the modified deployed law. Do not promise general statistical robustness from the first-cell formula alone.
