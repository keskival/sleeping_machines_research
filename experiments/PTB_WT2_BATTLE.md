# R1 published word-level benchmarks: Penn Treebank and WikiText-2

Opened 16 October 2026 (Curie owner; user: train our own models against published benchmarks first). Battle R1.

## Frozen protocol

- **Data.** PTB: Mikolov preprocessing, `data/ptb/ptb.{train,valid,test}.txt`, 10,000-word vocabulary including `<unk>`;
  929,589 / 73,760 / 82,430 tokens. WikiText-2 (v1): `data/wikitext2/wiki.{train,valid,test}.tokens` rebuilt from the
  Salesforce parquet (`scripts/convert_wikitext2_parquet.py`), 33,278 words; 2,088,628 / 217,646 / 245,569 tokens.
  Tokenisation `line.split() + ['<eos>']` (AWD-LSTM convention); each split is one continuous stream.
- **Score.** Perplexity = exp(mean NLL) over every token of the split; the first conditions on a start `<eos>`. History
  crosses sentence and article boundaries (as in the published recurrent and Transformer-XL evaluations).
- **Two columns.** Static: frozen weights. Dynamic evaluation: weights adapt causally on already-scored text of the
  split being scored (Krause et al. 2018), the published dynamic-evaluation column. Our online arm makes each segment's
  predictions before its tokens change any weight or state (contract).
- **Sealed test.** Development uses validation only. Test is scored once per selected model
  (`r1_ptb_native.py --mode score --split test --select-from <validation score>`; online step size chosen on validation).
- **Training data.** The benchmark's training split only; no pretrained models in the main table.

## Published references (Melis et al., ICLR 2020, Table 1; local PDF `results/public_reference/melis2020_mogrifier_iclr.pdf`)

| Corpus | Model | Params | Val static | Test static | Val dyneval | Test dyneval |
| --- | --- | --- | --- | --- | --- | --- |
| PTB | Mogrifier (d2, MC) | 24M | 51.4 | 50.1 | 44.9 | 44.8 |
| PTB | FRAGE (d3, MoS15) | 22M | 54.1 | 52.4 | 47.4 | 46.5 |
| PTB | AWD-LSTM (d3, MoS15) | 22M | 56.5 | 54.4 | 48.3 | 47.7 |
| PTB | Transformer-XL | 24M | 56.7 | 54.5 | — | — |
| PTB | LSTM (d2) | 24M | 55.8 | 54.6 | 48.9 | 48.4 |
| WT2 | Mogrifier (d2, MoS2, MC) | 35M | 57.3 | 55.1 | 40.2 | 38.6 |
| WT2 | FRAGE (d3, MoS15) | 35M | 60.3 | 58.0 | 40.8 | 39.1 |
| WT2 | AWD-LSTM (d3, MoS15) | 35M | 63.9 | 61.2 | 42.4 | 40.7 |

Pass criterion (developed attempt): test perplexity below the best published number in a column on a corpus, three seeds
for the selected configuration.

## Design rationale: why the native family should win here

1. **Cache and adaptation are where these leaderboards move.** On WikiText-2, dynamic evaluation takes the best models from
   55–61 to 39–41 (15–20 points, table above); a continuous cache pointer also gives large gains to static models
   (Grave et al. 2017; Merity et al. 2018). Text repeats names and
   topics within an article. Our output keyed token-slot bonus is a learned cache: each past event writes a key to its own
   word's slot and the query adds decayed slot scores to those words' logits. It is trained end to end with the rest of
   the model. Causal online weight adaptation is a primary capability of the family (AGENTS.md).
2. **Persistent temporal memory across sentences.** Complex-diagonal decaying and rotating state carries article context
   at no attention cost; per-layer keyed race reads give addressed access to every earlier event in the window.
3. **Depth pays in this composition.** One pass over 4M FineWeb tokens: L8 beats L2 (flagship 5.8414 vs 5.8648) with the
   gap growing through the pass; the gain sits on word-initial targets and long history (word-level prediction).
4. **Softmax bottleneck.** d = 256 against 10k words is rank-limited (the reason MoS helps the AWD models); the keyed slot
   bonus adds a high-rank, context-specific term. Mixture-of-softmaxes stays available if error analysis shows the
   bottleneck.
5. **Small-corpus regularisation** (standard primitives, all in the driver): word-type embedding dropout, input, residual
   and output dropout, decoupled weight decay.

## Development log

| Round | Model | Train | Val static | Val online | Note |
| --- | --- | --- | --- | --- | --- |
| 1 (queued, Curie v18) | d256 L8, 9M params, lanes 512 × 4, 16 epochs | — | — | — | first fit; learning curve decides round 2 |

Round 2 options, chosen by round 1's curve: overfitting → stronger regularisation (AWS arm 2); underfitting → width d384
(AWS arm 1) and longer schedules; online gain below dynamic evaluation's 6–7 points → adaptation segment length, decay and
exact carried-state scoring (no window recompute; unlimited temporal-memory history).

## Timebox and stop rule

About two weeks of development (to 30 October 2026). Stop rule: no configuration within 10% of the best published
validation perplexity in either column after the documented plan; then a developed-attempt loss with a written diagnosis.
