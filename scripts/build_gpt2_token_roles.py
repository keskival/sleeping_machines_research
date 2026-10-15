"""Build the GPT-2 token role table used for observed part-whole boundaries (experiments/tables/gpt2_token_roles.json).

Roles: 0 word-initial (leading space + letter/digit), 1 continuation (letter/digit, no leading space), 2 other punctuation,
3 clause mark (, ; :), 4 sentence end (. ! ? ..., optionally followed by closing quotes/brackets), 5 newline, 6 other,
7 end-of-text. Source: GPT-2 encoder.json (https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json).
"""
import hashlib, json, sys
from pathlib import Path

PUNCT = set('.,;:!?"\'()[]{}-')


def role(s):
    core = s.lstrip('Ġ'); tail = core.rstrip('"\')]')
    if s == '<|endoftext|>': return 7
    if 'Ċ' in s: return 5
    if core and all(c in PUNCT for c in core):
        return 4 if tail in ('.', '!', '?', '...', '?!', '!?') else 3 if tail in (',', ';', ':') else 2
    if s.startswith('Ġ') and core[:1].isalnum(): return 0
    if core[:1].isalnum(): return 1
    return 6


def main():
    src = Path(sys.argv[1]); enc = json.loads(src.read_text()); roles = [6] * len(enc)
    for s, i in enc.items(): roles[i] = role(s)
    out = dict(source='https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json', source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
               codes=dict(word_initial=0, continuation=1, punctuation_other=2, clause_mark=3, sentence_end=4, newline=5, other=6, end_of_text=7),
               roles=''.join(map(str, roles)))
    Path(sys.argv[2]).write_text(json.dumps(out) + '\n'); print({c: roles.count(c) for c in range(8)})


if __name__ == '__main__':
    main()
