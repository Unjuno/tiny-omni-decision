# Saved text predictions: domain-condition failure

Read-only post-hoc analysis of the completed coverage candidate, without a new
model call, fitting, selection or calibration. Joined family0/family4 predictions
to the frozen vocabulary-transfer inputs; asserted matching base question IDs,
state, options/order and targets in every paired row. Original input hash and
all four prediction hashes are retained in the adjacent JSON.

Focus on six extremum conditions across128 states (768questions per arm);
count and conditional-selection questions remain in the original full report.
This subset is a mechanism diagnostic, not a replacement benchmark.

| Condition | Familiar correct /128 | Domain correct /128 |
|---|---:|---:|
| Minimum cost |124|72|
| Maximum cost |108|14|
| Minimum risk |121|58|
| Maximum risk |115|32|
| Minimum time |120|43|
| Maximum time |120|36|
| Total |708/768|255/768|

On the affine-expanded numeric arm, corresponding totals are630/768 and236/768.
The report also preserves the selected rank1–4 in the requested field and counts
familiar-correct questions lost to the opposite extreme. Mistakes include middle
ranks as well as opposite extremes; simply reversing predictions cannot fix this.

The family4 question changes vocabulary, adds an explicit field-alias preamble,
and changes the question structure. Same numbers/options within paired arms rule
out numeric input changes as the explanation for this contrast, but do not isolate
one lexical cause or prove arithmetic is reliable. Repeated states across arms
are not additional independent samples. These are observed development results.

Next admissible text hypothesis is condition interpretation under changed wording,
not automatic model expansion or more epochs on the same data. Any training
intervention must use distinct grouped states, retain a prospectively held-out
wording family, and preserve the current transfer results as observed diagnostics.
Do not add these evaluation rows to training or tune wording until this score rises.
The weaker domain performance remains part of the full product quality assessment.

Local result: C:/CodexArtifacts/pretrained-reuse-r2/text-domain-condition-diagnosis-01.
Reproduce by taking each saved prediction's argmax option, sorting the unchanged
state by oracle field, and accumulating its1-based rank by field/direction.
Use the oracle only for this diagnostic, never as model input.
