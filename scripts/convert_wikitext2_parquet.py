"""Rebuild the original WikiText-2 (v1, word-level) token files from the Salesforce/wikitext parquet splits.

The parquet rows are the original file's lines; blank rows are the original ' \\n' lines. Output: data/wikitext2/wiki.{train,valid,test}.tokens,
which tokenised as line.split() + ['<eos>'] give the standard 2,088,628 / 217,646 / 245,569 tokens and a 33,278-word vocabulary.
Needs pyarrow on the path (not in the project venv): e.g. unpack the pyarrow wheel to a scratch directory and pass it in PYTHONPATH.
"""
import sys
import pyarrow.parquet as pq

src, dst = (sys.argv[1:3] if len(sys.argv) > 2 else ('data/wikitext2', 'data/wikitext2'))
for split, name in (('train', 'train'), ('validation', 'valid'), ('test', 'test')):
    rows = pq.read_table(f'{src}/{split}.parquet').column('text').to_pylist()
    with open(f'{dst}/wiki.{name}.tokens', 'w') as f: f.write(''.join(r if r else ' \n' for r in rows))
