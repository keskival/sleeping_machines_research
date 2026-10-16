**B4 Stack Overflow developed result,16October:** developed attempt below composite target;recorded-data five-split totalNLL11.995598±0.647576SE versus composite11.90. Variant selected by all-five-split DEV mean;TEST once. Source:`experiments/results/tpp_b4/aws_b4x_so_test_20261016_v2.json`. No further fitting automatically admitted.

**B4 Github developed result,16October:** WIN: five-split recorded-data totalNLL-284.465765±54.288843SE versus composite-272.90;best single published model-269.70. Selectedptr by all-five-split validation mean;TEST once. Native pointer histories retained;completed old splits reused and only memory-failed fits replaced. Source:`experiments/results/tpp_b4/aws_b4x_github_test_20261016_v3.json`. No further fitting automatically admitted.

# Current developed B4 evidence and continuation

**Wikipedia WIN:** recorded-data total NLL-258.760894±37.733361SE versus the
predeclared composite target-122.62 and best single published model-2.67.
Five official splits,118TEST sequences per split;all five native totals beat
the composite mean target. The pointer variant was selected by the aggregate
five-split validation mean among pointer,local-clipped pointer and mark memory.
TEST scored once under the previously admitted developed protocol. The old
scorer failed source preflight before execution;the frozen replacement changes
no model/data/selection rule. Native source and checkpoint hashes match.
Source:`experiments/results/tpp_b4/aws_b4x_wiki_test_20261016_v2.json`.

This is the developed model's recorded-density quality result,not a claim of
recording-cell/dequantization robustness or matched compute. The earlier frozen
one-configuration transfer failure is a separate completed experiment. Clocks
and conditional marks both improve;no hardware energy or latency claim follows.

**Github memory repair:** the actual longest32-sequence TRAIN batch completes
at1072656KiB RSS with full4699-event histories,versus the original3MKiB cap
failure. Row microbatching trims padded work;no time/context truncation. Six
replacement fits (pointer/global and pointer/localclip,splits1–3) run on slots5/6.
Completed splits0/4 and all five mark-memory fits stay reusable. Exact pass
recovery includes Adam,RNG,order,caps and history;validation is bounded to one
full sequence at a time with unchanged per-sequence statistic. Three numerical
contracts validate update/recovery/evaluation;pointer law/causality contracts
also pass. `b4_github_memory_repair_v1` waits all results,freezes selection across
five DEV splits,admits TEST once,publishes,and stops.

**Stack Overflow:** the original split4 pointer/mark fits finish unchanged.
`b4_stack_scoring_v1` waits all ten development results and either retains the
original score or restores only an unexecuted source-preflight-failed scorer.
No extra variant,tuning arm,TEST selection or duplicate TEST is admitted.

Remaining mark/timing projects follow `experiments/BENCHMARK_CONTINUATION.md`.
