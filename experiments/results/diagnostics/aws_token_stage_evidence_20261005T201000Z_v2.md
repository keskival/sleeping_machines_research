# Integrated token quality and complete work

Completed GPT-2 FineWeb native development stages on one scored population. Frequency priors/cadences can differ; absolute NLL and within-fit learning are separate. Arithmetic/special functions separate; sampling work unquantified and initialization/evaluation excluded from fit FLOPs. Missing whole-fit ledgers remain null. No public benchmark, scaling exponent or matched-reference win inferred.

| TRAIN tokens | P | Seed | Credit | Fit targets | Passes | State scalars/lane | Keys/writes per target | Initial NLL | Selected NLL | Learning gain | Whole-fit GFLOPs | Fit MFLOPs/target | Eval MFLOPs/target |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 16 | 6 | 16 | 16368 | 2.00 | 256 | 16/4 | 8.340313 | 8.297491 | 0.042822 | 13.534669 | 0.826898 | 0.272633 |
| 8192 | 16 | 7 | 16 | 16368 | 2.00 | 256 | 16/4 | 8.340313 | 8.286427 | 0.053886 | Pending | Pending | Pending |
| 8192 | 16 | 6 | 64 | 16368 | 2.00 | 256 | 16/4 | 8.340313 | 8.300131 | 0.040182 | Pending | Pending | Pending |
| 8192 | 16 | 7 | 64 | 16368 | 2.00 | 256 | 16/4 | 8.340313 | 8.290236 | 0.050078 | Pending | Pending | Pending |

Pending stage names: curie_data_growth_tokens_64k_b64_c16_p16_s6_20261005_v1, curie_data_growth_tokens_64k_b64_c16_p24_s6_20261005_v1, curie_data_growth_tokens_64k_b64_c16_p32_s6_20261005_v1.

Available memory, key scoring, selected writes and learning work are distinct columns. Decoder and optimizer work are included in executed ledgers; state size is not a FLOP saving. Evaluation here includes exact likelihood/scorer reductions.
