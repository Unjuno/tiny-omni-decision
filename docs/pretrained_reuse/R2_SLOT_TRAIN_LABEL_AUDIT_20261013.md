# Train corpus scope and label audit

Read-only audit of the fixed99-video/198-question training corpus. Two unique
question strings and three exact option sets occur. There are53 changed-target
and46 unchanged-target video pairs. Initial targets occur33/43/23 times for
indices0/1/2; final targets26/46/27. All nine initial-final target combinations
occur. Exact counts and source SHA256 are in the adjacent JSON.

The condition-only training majority-index baseline is89/198=.449495. This is a
training-distribution statistic, not an evaluated model or generalization result.
It also operates on option indices, not grounded spatial semantics. Therefore,
uniform33.3% chance alone is an insufficient account of the training distribution.

The trained readout's .757576 training Accuracy exceeds that simple statistic,
but its .361868 validation Accuracy and large NLL gap still reject useful transfer.
Two question templates cannot establish broad language-condition generalization.
All these videos address one occlusion-game task; they cannot establish the full
Omni product's multimodal quality target.

## Implication for the next bounded experiment

Do not respond with an unrestricted head-capacity sweep on the observed validation.
A future relational-head candidate needs a fixed, grouped development split drawn
from training only, exact source/participant metadata checked before defining it,
and a same-split baseline. Preserve all questions from a video together. Freeze
the head and selection policy before consulting the old development validation.
The expected issue is whether joint slot-time interaction improves transfer;
the current evidence does not justify assuming that it will. Additional training
has not been launched as part of this audit.
