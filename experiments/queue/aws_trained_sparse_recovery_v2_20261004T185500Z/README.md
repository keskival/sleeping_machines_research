# Secondary pool4 sparse validation — prepared, unrun

This restores the original saved90M p32/D4/H2/pool4 validation at exact producer
and validator source bytes. It is a secondary richer-capacity deployment
control. Prioritize the leading completed p64 checkpoint through
`aws_best90m_sparse_recovery_20261004T190000Z` after existing owner priorities.

Use only these wrapper queues through `run_safe.sh` on the actual AWS host,
with manifest guards and inherited reservations. Contracts precede both DEV
prefixes; published, hash-matching contracts and the matching prefix precede
full scoring. Nothing here has run numerically. The original192500Z queues
remain preserved and source-drifted on the live tree.

See [the recovery theory](../../theory/153_source_exact_sparse_deployment_recovery.md)
for scope, byte provenance, numerical gates and the remaining serving proof.
