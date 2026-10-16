#!/usr/bin/env python3
"""Repair missing spaces in investor/report prose written by publication scripts (founder, 16 Oct: "weird garbage").

Inserts a space after ',' or ';' between letters/closing brackets and a following letter, between common function words and
a following number ("and1.05M" -> "and 1.05M"), and between a number and a following lowercase word ("64times" ->
"64 times"). Code spans, \\path{}, \\texttt{}, URLs, Markdown link targets and LaTeX diagram/command lines are left untouched.
Usage: fix_prose_spacing.py FILE [FILE ...]   (prints the number of edits per file)
"""
import re, sys
PROTECT = re.compile(r'(`[^`]*`|\\path\{[^}]*\}|\\texttt\{[^}]*\}|\\url\{[^}]*\}|https?://\S+|\]\([^)]*\)|\$[^$]*\$)')
FUNC = r'(?:and|or|in|to|of|at|by|vs|versus|from|with|the|The|for|over|than|scores?|is|are|was|were|reaches|improves|on|after|under|about|only|while|against|becomes|stays|gives|uses|has|have|from|into|per|all|each|its|their)'
RULES = [(re.compile(r'([A-Za-z0-9\)%\]])([;,])([A-Za-z\(])'), r'\1\2 \3'),
         (re.compile(r'\b(' + FUNC + r')(\d)'), r'\1 \2'),
         (re.compile(r'(\d)(times|versus|tokens?|targets?|layers?|levels?|seeds?|updates?|units?|parameters?|nats|words?|epochs?|splits?|datasets?|wins?|bits?|steps?|events?|slots?|concepts?|scenes?|pairs?|arrangements?)\b'), r'\1 \2'),
         (re.compile(r'([a-z])\.([A-Z][a-z])'), r'\1. \2'),
         (re.compile(r'\b([A-Za-z][a-z]{2,})(\d)'), r'\1 \2'),                                    # "isolates13" -> "isolates 13"
         (re.compile(r'(\d)([a-z]{3,})\b'), r'\1 \2'),                                    # "4096forecast" -> "4096 forecast"
         (re.compile(r'([a-z]):([0-9A-Za-z])'), r'\1: \2'),                                # "Scope:1024" -> "Scope: 1024"
         (re.compile(r'([a-z\)]),(\d)'), r'\1, \2'),
         (re.compile(r'(\d)([A-Z]{2,}[a-z]*)\b'), r'\1 \2'),                               # "128TRAIN" -> "128 TRAIN" (single-letter units like 4M kept)
         (re.compile(r'(\d)%([a-z])'), r'\1% \2'),
         (re.compile(r'(\d)(of|to|in|on|at|vs|by|or)(\d)'), r'\1 \2 \3'),
         (re.compile(r'(\d)(January|February|March|April|May|June|July|August|September|October|November|December)'), r'\1 \2'),
         (re.compile(r'(\*\*),([a-z])'), r'\1, \2'),
         (re.compile(r'(\d)M([A-Z]{2,})'), r'\1M \2'),                                       # "8MDEV" -> "8M DEV"
         (re.compile(r'\b([A-Z]{3,})(\d)'), r'\1 \2'),
         (re.compile(r'\);\['), '); ['), (re.compile(r'\},\\(texttt|path)'), r'}, \\\1')]                                       # "DEV6.16" -> "DEV 6.16" (dataset names kept)                                         # "actor,4096" -> "actor, 4096"
KEEP = re.compile(r'\b(word2vec|seq2seq|gpt2|top1|top5|top8|fp32|bf16|log1p|exp2|ndcg|text8|enwik8|enwik9|mamba2|llama2|llama3|Aug25|MIMIC2|MIMIC3|GPT2|BTCUSDT|S2P2|LSTM2)\b', re.I)
SKIP_TEX = re.compile(r'\\(node|draw|begin|end|includegraphics|tikz|path\[|fill|coordinate|foreach|usepackage|definecolor|newcommand|def\b)')


def fix_text(seg):
    keep = {}
    def hide(m): keep[f'\x00{len(keep)}\x00'] = m.group(0); return f'\x00{len(keep) - 1}\x00'
    seg = KEEP.sub(hide, seg)
    for rx, rep in RULES: seg = rx.sub(rep, seg)
    for k, v in keep.items(): seg = seg.replace(k, v)
    return seg


def fix_line(line, tex):
    if tex and (SKIP_TEX.search(line) or line.lstrip().startswith('%')): return line
    if line.lstrip().startswith(('|---', '```')): return line
    parts = PROTECT.split(line)
    return ''.join(p if i % 2 else fix_text(p) for i, p in enumerate(parts))


for f in sys.argv[1:]:
    src = open(f).read(); tex = f.endswith('.tex'); out = []; in_code = False; edits = 0
    for line in src.split('\n'):
        if line.lstrip().startswith('```'): in_code = not in_code
        new = line if in_code else fix_line(line, tex); edits += new != line; out.append(new)
    open(f, 'w').write('\n'.join(out)); print(f, edits)
