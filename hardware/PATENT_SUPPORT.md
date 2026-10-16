# Hardware implementation and patent-support map

16 October2026. Inventor identified from his express statement:Tero Keski-Valkama. This is an implementation/support map,not a novelty finding or a statement that these generic circuits are individually patentable.

| Recorded construction | Concrete draft / prospective completion | Relevant family |
|---|---|---|
| Separate selection/payload paths;clock race;selected addressed updates | Supplied due-time/address/tag race circuit,backpressure and state-bank commit interface. Learned key scoring,clock transforms and native value proposal remain upstream. | PF-01 / PF-11 |
| Lazy decaying/rotating persistent memory | Fixed-point complex multiply datapath with explicit floor/saturation and supplied elapsed-time coefficients. On-chip coefficient generation remains missing. | PF-06 / PF-11 |
| Issued versions and local state ownership | Expected-version state commits and immutable reference directory. Complete dependency-set validation,producer joins and local optimizer are separate missing blocks. | PF-09 |
| Child identity separate from role;parent binds exact child versions | Immutable parent-version/child-identity directory with pose as data and timestamp-availability checks. Retained value allocator,GC and complete learned composition remain missing. | PF-10 |
| Heterogeneous defective clock/mark engine and causal pointer | Described in patent specifications;not implemented by the supplied-clock argmin alone. Needs stable special functions,clock-specific marks,compensator/quadrature and pointer normalization. | PF-02 / PF-03 / PF-04 |
| Forward/alternative learner consequences | Described in specifications;RTL does not yet implement replay,eligibility,counterfactual value teaching or optimizer-state worlds. | PF-01 / PF-06 / PF-07 / PF-08 |

The next evidence package should bind the exact checkpoint,RTL,build constraints,tool version,board/part,device configuration,workload trace,numerical formats and output quality to post-route utilization and physical measurements. Record event traffic,all candidate scoring,selected writes,snapshot lifetimes,learning work and idle costs separately. State whether the measurement is block-only or end-to-end. Board measurements do not establish ASIC power;generic synthesis does not establish either.

The current LaTeX descriptions include digital FPGA/ASIC embodiments and drawings now. They deliberately do not assert invented measurements. The inventor should refine specific circuit alternatives before filing,especially timer/noise realization,cancellation,tie semantics,clock-domain crossings,coherence,SRAM organization and local learning. Add actual FPGA measurements as evidence when available. A later new technical embodiment receives earlier priority only to the extent the earlier application supports it;do not assume an umbrella hardware reference preserves every future design.
