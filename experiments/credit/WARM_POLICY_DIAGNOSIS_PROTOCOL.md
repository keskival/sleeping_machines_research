# R1 paired learner-state policy diagnosis

11 October 2026, before evaluation. Decision: does the completed32-window
controller align better with future route utility when lexical/Adam state is warm?
No fitting or selection. Reuse eight private development prefixes and final
zero/static/transport controllers from route48 fit, explicitly nonindependent.
Cold exact costs are already paid. Warm lexical learner sees192 preceding
observations plus64-token scored context, with64 initial native context tokens
and191 supervised updates; no future scored target enters warm-up or issuance.
Reset at observed EOS. Keep future native context identical to the cold evaluation
so this diagnoses lexical/optimizer state only, not full persistent-native state.
Controller sees its existing prefix features and warm lexical vector, but does
not explicitly see Adam moments. The exact worlds do carry independent moments.

Score48 future targets for every issued route using independent carried Adam
learners. Report controller gain over initial, uniform/oracle headroom and
transport-over-static paired differences in cold and warm regimes. This is a
post-fit diagnostic, not a confirmation or new win. Check candidate/actions,
cold native issued features and warm reference losses against independent
actual replay. Smoke one prefix; measure all8 including disclosed smoke reuse.
Source-bind parent checkpoint, completed exact-carried contract and scripts.
OneCPUthread/run_safe/RSS1.5GB/MemAvailable9GiB,240s smoke/600s measure.
