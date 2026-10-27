# Open Images target-scope shortcut audit

## Finding

The pilot generator chose positive targets only from 11 classes frozen using
training metadata, while negative choices could be any explicitly verified
negative sibling, including names outside those 11 classes. A model or observer
that knows this task-design whitelist can use option text alone to eliminate
all out-of-scope candidates. This is a task-construction shortcut, not evidence
of train/test image duplication and not a claim that a model was explicitly
given the list. The existing frozen model may learn a similar prior from its
training examples.

This post-hoc diagnostic uses the published train-only class whitelist and the
already saved 168 pilot and 168 fresh-task-ID development predictions. It reads
no pixels, model weights, official test or sealed audit. For each question, the
rule selects the first supplied option in the 11-class whitelist. A second
quantity is the *expected* accuracy from choosing uniformly among whitelist
options; this is calculated analytically rather than selecting a random seed.

| Previously exposed cohort | Whitelist has exactly one option | Expected uniform-within-list accuracy | First whitelist option | Saved model accuracy |
|---|---:|---:|---:|---:|
| Pilot (VideoPrism + MiniLM) | 65/168 | 65.33% | 66.07% | 79.17% |
| Fresh task IDs (TinyCLIP) | 66/168 | 64.98% | 68.45% | 64.29% |

For the pilot, saved-model minus deterministic prior was +13.10 points with
paired image bootstrap interval +4.76 to +21.43; however, previous zero-image
controls leave all 168 model choices unchanged, so the excess is not evidence
of image grounding. For fresh task IDs, TinyCLIP minus this prior was -4.17
points, interval -14.29 to +5.95. Bootstrap uses 10,000 images, seed 17; rows
are small nonblind samples and may share source-level dependencies. The fresh
TinyCLIP image-control intervention still showed input sensitivity, but accuracy
is not interpretable as a clean measure of general image understanding when
this option-only shortcut exists. The whitelist baseline does not prove what
features either model used, or that all visual information is useless.

Do not remove examples or retune the existing validation sets after this audit.
For a new task, freeze a training-only option-generation rule that makes target
and distractor class roles symmetric, preserves explicit negative verification,
keeps every predeclared class, and reports task-only and blank-image controls.
Establish source/image independence and rights before fit or independent quality
claims. No replacement task has yet been built, and the current 95–98% general
multimodal goal remains unmet.

Result:
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-target-scope-shortcut-20261027-attempt-01/result.json`,
SHA-256 `2c56b433f867f26b976c9522cd310787e2f8a145e34dcc536d5500619d64304c`.
The receipt pins the train scope and both saved prediction files; no new model
inference, training, prompt change or validation-set selection was performed.
