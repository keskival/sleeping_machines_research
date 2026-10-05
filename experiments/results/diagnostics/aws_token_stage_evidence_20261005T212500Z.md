# Integrated token quality and complete work

Completed GPT-2 FineWeb native development stages on one scored population. Frequency priors/cadences can differ; absolute NLL and within-fit learning are separate. Arithmetic/special functions separate; sampling work unquantified and initialization/evaluation excluded from fit FLOPs. Missing whole-fit ledgers remain null. No public benchmark, scaling exponent or matched-reference win inferred.

| TRAIN tokens | P | Seed | Credit | Fit targets | Passes | State scalars/lane | Keys/writes per target | Initial NLL | Selected NLL | Learning gain | Whole-fit GFLOPs | Fit MFLOPs/target | Eval MFLOPs/target |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 16 | 6 | 16 | 16368 | 2.00 | 256 | 16/4 | 8.340313 | 8.297491 | 0.042822 | 13.534669 | 0.826898 | 0.272633 |
| 8192 | 16 | 7 | 16 | 16368 | 2.00 | 256 | 16/4 | 8.340313 | 8.286427 | 0.053886 | Pending | Pending | Pending |
| 8192 | 16 | 6 | 64 | 16368 | 2.00 | 256 | 16/4 | 8.340313 | 8.300131 | 0.040182 | Pending | Pending | Pending |
| 8192 | 16 | 7 | 64 | 16368 | 2.00 | 256 | 16/4 | 8.340313 | 8.290236 | 0.050078 | Pending | Pending | Pending |
| 65536 | 16 | 6 | 16 | 131056 | 2.00 | 256 | 16/4 | 8.162280 | 8.099440 | 0.062840 | Pending | Pending | Pending |
| 65536 | 24 | 6 | 16 | 131056 | 2.00 | 384 | 16/4 | 8.162280 | 8.033311 | 0.128969 | 192.211765 | 1.466638 | 0.403428 |
| 65536 | 32 | 6 | 16 | 131056 | 2.00 | 512 | 16/4 | 8.162280 | 8.090419 | 0.071861 | Pending | Pending | Pending |
| 65536 | 24 | 7 | 16 | 131056 | 2.00 | 384 | 16/4 | 8.162280 | 8.078991 | 0.083288 | Pending | Pending | Pending |

| TRAIN tokens | P | Seed | Credit | Context gain NLL | Memory erasure cost NLL | Message erasure cost NLL |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 16 | 6 | 16 | 0.017759 | 0.000381 | 0.017843 |
| 8192 | 16 | 7 | 16 | 0.027515 | 0.000980 | 0.021955 |
| 8192 | 16 | 6 | 64 | 0.001898 | -0.000268 | 0.020348 |
| 8192 | 16 | 7 | 64 | 0.017711 | 0.000449 | 0.021285 |
| 65536 | 16 | 6 | 16 | 0.208993 | 0.006892 | -0.024694 |
| 65536 | 24 | 6 | 16 | 0.252343 | 0.007313 | 0.022623 |
| 65536 | 32 | 6 | 16 | 0.166731 | 0.002895 | -0.009512 |
| 65536 | 24 | 7 | 16 | 0.234159 | 0.005551 | -0.032675 |

Frozen utility uses matched route RNG and selected weights; negative erasure cost means the intervention improves loss. Context gain uses a constant causal TRAIN-mean feature through the same readout. These are not retrained ablations or additive attribution.


Pending stage names: .

Available memory, key scoring, selected writes and learning work are distinct columns. Decoder and optimizer work are included in executed ledgers; state size is not a FLOP saving. Evaluation here includes exact likelihood/scorer reductions.
