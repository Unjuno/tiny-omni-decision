# Conditional development curriculum v1

Purpose: test whether bounded training repairs conditional candidate selection
without assuming that a stronger frozen checkpoint will supply that skill.
User authorized autonomous strategy and local experiments on2026-10-12.
This is programmatic development data created in this project, not an upstream
licensed corpus, a blind audit, or evidence of general multimodal quality.
It is explicitly motivated by observed failures; do not present it as unbiased
model evaluation. Existing Teacher/data/model artifacts remain unchanged.

Generator source commit424ac48, exact source SHA and file hashes in JSON.
Subsequent line wrapping changes no generated text. No downloaded content is
incorporated. States contain four named items with distinct cost/risk/time
integers1..199. Eight questions per state: four cost/risk extrema, two earliest/
latest scalar-time extrema, one threshold count, one filtered minimum-cost
selection with a none option. "temporal_order" means numeric timestamp order,
not understanding raw video. No arithmetic/rule engine is added to inference.

Train: seed17,1,024 states/8,192 questions. Validation: seed29,256 states/
2,048questions. Group identity hashes sorted numeric tuples ignoring labels and
row ordering; groups are disjoint. All questions from one state stay together.
Option order is independently shuffled;4or5options remain intact. Target
positions and task counts are in JSON. Oracle verification independently sorts
items and checks filtering/counting. Tests reject deliberately wrong labels.

Limits: templates, numeric range and labels are shared across splits, so this
is in-distribution state generalization only. These are not human-authored
questions, natural scenes or audio/video tasks.95% here would not meet the
user's product contract. The earlier256-question diagnostic is retained as
already observed development evidence and is not a final target to tune against.

## Predeclared next bounded training stage

Use the existing22.7M adapted Cross-Encoder as a preserved starting reference,
updating only the same final block/classifier(1.775M parameters). Mix this new
curriculum with original Typed Decisions Synth training questions at equal
question counts using deterministic seed17 order, with original validation
retained to detect forgetting. No data/labels from validation enter training.
Separate smoke process first; then a512-update pilot at batch8questions,
fixed2e-5 LR and existing optimizer/clip/dropout settings. Record initial and
128-step interval metrics on both full development validations, task breakdowns,
unique-state/question counts, gradient/frozen-weight checks and resource use.
Select earliest minimum of the unweighted mean NLL across the two validation
corpora; retain best and final separately and report every scheduled result. Freeze this
pilot before observing outcomes; no automatic budget or capacity extension.
No new global product benchmark or deployment promotion is implied.

The pilot has **not** run yet. The corpus generation and oracle/split checks
have executed successfully. Large data remain outside Git.

## Execution follow-up

The declared512-update pilot has now completed. See
[R2_CONDITIONAL_REPAIR_PILOT_20261012.md](R2_CONDITIONAL_REPAIR_PILOT_20261012.md).
Execution and freeze/reload checks passed; useful reasoning improvement was
not established. No product gate is closed by this development experiment.
