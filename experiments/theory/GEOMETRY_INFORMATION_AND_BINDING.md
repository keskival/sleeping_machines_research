# Geometry is learned;nonadditive binding is the next capability

16 October 2026. Completed three-arm acquisition and calibrated decoder audit.

Joint local grounding reduces root geometry DEV MSE from the frozen encoder's
0.0401215 to0.0262533 with the same32-output affine decoder class and fixed
TRAIN-only ridge0.01. That is34.5656% less geometry error. Relative to a constant
TRAIN target,the skills are22.91% and49.56%. Frozen encoder features are exactly
unchanged;the joint encoder actually changes. This is representational learning
on held-out combinations/nuisances,not merely a decoder learning its mean.

Whole-label DEV NLL is0.732158(joint),0.783903(CE continuation),0.686609(frozen
acquired encoder). Joint beats continued label-only learning but not the acquired
model or balanced prior0.693147. No arm has qualified class transfer. The current
joint actor is not promoted as a new flagship. Geometry is therefore useful
evidence about representation preservation,but not proof of the desired semantic
hierarchy or a reason to scale the unchanged reconstruction objective.

The fixed geometry sketch composes by an additive role transformation. Such a
target can reward preserving layout information without rewarding the nonlinear
relations required by the parity task. More faithful child content is necessary
but may remain linearly unreadable for the whole label. The next design should
separate two questions:which parts are present,and how those identities bind to
roles. A useful identity may remain stable when its own parts rearrange,while a
relational state must retain the difference. Collapsing both into one continually
changing layout vector makes higher-level identity comparisons harder to reuse.

Keep native persistent memories,keys/values,role-sensitive races,generated views
and existing geometric evidence. Investigate intrinsic part frames and distinct
identity/binding channels;do not replace the native actor with a fixed symbolic
tree or feed latent task identities. Evaluate candidate invariance to observed
translations/role swaps and sensitivity to real part changes before a fit.
Whole-label and relation-specific predictive targets must require combining
children;single-child class supervision is information-free on this task.

Centering/scaling geometry innovations fixes the measured auxiliary conditioning
problem and preserves live local credit. It is supporting learning machinery,
not by itself the selected solution to nonadditive binding. Its separate native
class and sparse exact teacher pass numerical contracts;neither replaces active
or acquired models automatically. A next integrated fit must distinguish
preserved local information,learned relation interaction and genuine transfer.
