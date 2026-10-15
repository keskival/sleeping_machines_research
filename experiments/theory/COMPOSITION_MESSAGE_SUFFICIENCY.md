# Task-sufficient compositional messages

## What compression must earn

Let H_c be a child's complete revealed local history and M_c its emitted state.
A parent predicting Y from its own state P needs a message satisfying
p(Y | H_1,...,H_k,P) = p(Y | M_1,...,M_k,P)
for the task distribution. This is a task-specific sufficiency criterion, not
an implemented guarantee. It explains why slower cadence alone cannot create
abstraction: an event budget is useful only when the emitted state retains the
relations future parents need. Preserve relative pose outside a canonical part
value so the same part can be reused in different parent arrangements.

For the controlled task, write A,B for the children's three-bit arrangement
codes and R for the root orientation. Y=parity(A) XOR parity(B) XOR R. In each
split the allowed A XOR B differences contain equal numbers of even/odd codes.
Consequently H(Y | A,R)=H(Y | B,R)=log(2), while H(Y | A,B,R)=0.
A root must combine both sides; repeating or attending harder to one side is
insufficient. Every split's label prior is 1/2. This formalizes a compositional
information requirement but does not prove that a solution must use our chosen
number of neural stages; a sufficiently expressive flat model could solve it.

The next learning gate should use TRAIN-only head fitting/calibration and
report raw and calibrated loss separately on unseen DEV combinations. Fitting
is development until all three requirements hold: transfer, paired dependence
on emitted content, and response to the known arrangement-changing intervention.
A failed short fit diagnoses that construction/training budget, not the family.
