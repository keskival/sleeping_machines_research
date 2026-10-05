# Preserve temporal computation while reducing backward bookkeeping

The completed 64K/P24 timing replay records 153.375s backward out of 263.314s
inside fitting (58.25%), versus 2.018s for both readouts (0.77%). This identifies
backward/core execution as the next CPU engineering target. It does not identify
rotation as the dominant operator; the bounded probe tests that hypothesis.

For each pair, y=(c a-s b,s a+c b), c=cos(phi),s=sin(phi). The input adjoint is
(c g0+s g1,c g1-s g0); the angle adjoint is
-s*(g0 a+g1 b)+c*(g1 a-g0 b). The implementation preserves float64 angles and
trigonometry, casts c/s to payload precision for forward and input credit, and
reduces each broadcast multiply adjoint before summation and conversion back
to angle precision. Floating addition ordering is checked numerically rather
than assumed identical.

This is an isolated execution experiment, not an architectural substitution.
Temporal transport, decay/rotation, races and first-time credit, all key
scoring, separate keys/values, sparse memory writes, messages, persistent state,
and actual alternative-write learning are retained. Inference still computes
exactly the same stated forward expression. Learning retains first-order
adjoints; higher derivatives are explicitly refused by once_differentiable.
The default kernels and frozen FAS/language source files remain unchanged.

Required evidence: mixed-precision/broadcast primitive forward/gradient parity;
finite-difference first derivatives; whole integrated forward/state/all-gradient
and optimizer-step parity; unchanged forced-write continuation; and bounded
interleaved CPU timings including factual backward and alternative replay.
No speedup, arithmetic reduction or language-quality gain is claimed before
those measurements. A full fitting replay follows only if the contract passes
and measured execution improves. Existing quality/work records remain controls.
