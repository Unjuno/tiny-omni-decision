# Fresh-head internal baseline

Executed on the predeclared79/20-video internal development split from the
original training corpus. Both questions of each video remain together. Participant
identity is unavailable; this is not participant-disjoint or blind evaluation.
All99 videos had previously been inspected in train diagnostics, but this new
head receives gradients from only79. The original257-video validation was not used.

Fresh seed17 head,115,008 parameters, existing frozen224x129 features and text
features; AdamW .001, weight decay .01,8 video pairs/batch,50 fixed epochs.
500 updates,158 unique training questions,7,900 question presentations. No best
selection; the final epoch is the predeclared baseline endpoint. CPU fit4.690s.

| Epoch | Train Accuracy | Train NLL |
|---|---:|---:|
|0|.240506|1.099727|
|1|.474684|1.051553|
|10|.677215|.789934|
|20|.734177|.538108|
|30|.753165|.470745|
|40|.772152|.456477|
|50|.778481|.468665|

Internal endpoint evaluation (40 questions): Accuracy .425, NLL2.914194,
Brier .977441, ECE .465143. Initial Accuracy .50, final .35; both-correct4/20,
changed-target both-correct1/12. This confirms poor endpoint generalization on this
small internal split. There is no validation learning curve or claim of a selected
optimal epoch. The sample size is too small to support a precise quality ceiling.

## Integrity and provenance

Saved example IDs independently checked:79 train videos,20 development videos,
zero overlap. All artifact hashes rechecked;51 history entries and500 updates
confirmed. The model reloaded exactly. Existing runs and features are untouched.
Source parent93d74c1; orchestration subsequently committed as3bd8ae9. The recorded
fit helper hash belongs to train_slot_readout.py, not the new orchestration script.
Orchestration was wrapped in main and formatted after execution; no second training
run was launched. Command: `python -m scripts.run_slot_internal_baseline`.

Local directories under C:/CodexArtifacts/pretrained-reuse-r2/:
`perception-occlusion-slot-internal-baseline-01` and
`perception-occlusion-slot-internal-baseline-eval-01`.
The adjacent JSON preserves evaluation metrics, predictions' hashes, and input hashes.
This is a comparison baseline, not a product candidate. No additional head has yet
been trained on this split.
