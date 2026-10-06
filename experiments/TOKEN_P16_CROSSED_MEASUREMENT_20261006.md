# Complete the smaller-width data axis before reserved extrapolation

The original P24/credit16 member remains prioritized: both horizon extensions lose on the completed 64K seed6 comparison. Its 1M results and useful memory replicate across two seeds. The next economical model-sizing measurement fills P16 at 256K and 1M, preserving the main member and the reserved 4M point.

At 64K, P16 selects 8.099440 versus P24 8.033311 and P32 8.090419 NLL. P24 leads P32 at 256K and 1M too. Width therefore has a nonmonotonic observed effect under this recipe. Completing the cheaper P16 data axis answers whether more exposure makes that member competitive; it is not a prediction that width must help. The joint monotone capacity loss surface must be judged against these measurements and repeated-seed variation. If it misfits, report discrete-width data trends and residuals rather than forcing a positive capacity exponent.

Reuse the existing, never-executed one-job 256K P16 queue. New uniquely tagged 1M queue: curie_data_growth_tokens_1m_b64_c16_p16_s6_20261006_v1. Same source/data/seed6/learning rate .003/credit16/P16D2H2U4/eight lanes/batch64/two passes/four development checkpoints/initial-inclusive selection as the completed fixed family. Public validation remains untouched. The 256K fit has 1,024 updates/524,272 targets; 1M has 4,096 updates/2,097,136 targets. Initialization counts use only the declared FIT population.

Admission waits for both current exact 64K off-diagonal work receipts and unchanged source/data/queue hashes. Run 256K, streamed selected utility, then admit 1M only with trained selected weights, passed RNG/partition checks and positive context utility/practical learning gate. Finish streamed 1M utility. Every job uses one unique queue through run_safe; no overlap with live model work.

Caps: 256K 2,100s, 1M 9,000s, utility 600/900s; 1.2GB RSS, 5GB VMS, at least 8GiB MemAvailable, one thread. Completed P16/64K ordinary wall326.167s/448368KiB and P24/1M4375.510s/508712KiB motivate these limits with margin. RSS watchdogs remain enabled. No new dense control is fitted.

These cells complete a quality measurement grid, not a complete fitting-FLOP grid. Complete larger work remains a separate requirement; no 64K cost is copied into larger cells. Update the collector with the actual new 1M tag only after completion, then publish a new immutable snapshot and raw-point visualization. Freeze the reserved 4M quality/cost prediction and conventions before running it; no 4M result or fitted scaling coefficient is admitted by this note.
