# EIC short-proposal pitch deck: 10 slides (content outline, 7 October 2026)

There is no EIC template. The guide recommends 2–4 slides on business model, market and growth. Reuse the visual
style of `investment/PITCH_DECK.json`; keep each slide to one message.

1. **EVENTCORE: AI that computes through time.**
   - Event-native models for operational event streams, on ordinary CPUs.
   - [Company] S.L., Spain.
2. **The problem.**
   - Industrial, IT and clinical data are irregular, interleaved, partly missing events.
   - Sequence AI recomputes history per event and needs GPUs.
   - Early faults hide in the interleaving.
3. **The innovation.**
   - Memories that decay and rotate with elapsed time.
   - Temporal races route events and *are* the event likelihood, including silence.
   - Sparse addressed state; constant per-event cost.
   - Contains attention and Mamba as special cases.
4. **Proof on public leaderboards.**
   - Table: all five EasyTPP datasets (Taobao, Taxi, StackOverflow at matched compute, Retweet at 1/15, Amazon at 0.29×); P19 sepsis; PAM wearables; TGB trade graphs (0.868 vs 0.863).
   - Taxi reproduced on separate hardware; third-party kit ready.
   - Each row with the best published result and our compute ratio.
   - Next (in development, 9 Oct): Temporal Graph Benchmark: node affinity won (0.868 vs 0.863, 3 sealed seeds); link prediction ahead on validation (0.852 vs 0.842), sealed test running; a second TPP benchmark running.
5. **Proof of efficiency.**
   - Per-event compute vs the leading models: 1/12 (Taxi single model), 1/15 (Retweet), 0.29× (Amazon), 0.92× (Taobao), matched (StackOverflow); PAM at 1/19 of MTM's parameters.
   - Constant cost vs a Transformer's growing context cost.
6. **Product and beachhead.**
   - Early fault and anomaly detection on operational logs.
   - Deliverables: model plus CPU runtime, calibrated alarms, detection lead time.
   - [Pilot partner logos or status.]
7. **Market.**
   - Beachhead size from the customer denominator (sites × streams × price).
   - Expansion: IT and security operations, clinical early warning, OEM platforms.
8. **Business model and growth.**
   - Licence per stream or site, pilots → contracts → OEM.
   - Revenue forecast for 3–5 years [to be built].
9. **Team and plan.**
   - Founder record: lead ML roles; 35 published US/EP patent documents.
   - Hires; advisers; 24-month TRL 6 → 8 work plan; IP filings.
10. **The ask.**
    - Grant only €2.5M (70% of €3.57M) plus the €3M private round.
    - Milestones: pilots validated (month 9), runtime v1 (month 12), first paid contracts (month 18), TRL 8 (month 24).
