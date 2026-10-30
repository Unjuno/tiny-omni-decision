# R2 timed spatial relation diagnostic — fixed 2026-10-14

## Question and scope

After an exposed small-item mask is observed, does same-time nearest-anchor
geometry retain its relation to the source-defined covering object better than
fixed initial positions or reversed anchor motion? This is a limited diagnostic
of a geometric baseline, not a semantic selector, learned memory module or QA.

One already known public Perception Test **training** regression scene10903;
all1,070 timed model masks/full558-frame tracks and the unchanged32 original
sample IDs. Prior tracking result pin:
`1ed0d7e4b3114be53fd6df7359d895f04f2e361579d052231f7c95b4e96d1c05`.
Corrected query plan pin:
`a683e5a6f9772ca1cde37eec10419364707b9f2f295bb282ca62cfab4ce2fee0`.
No inference, training, new encoder, dev/sealed scoring, media download, cloud
compute or settings changes. Do not extend the failed99-scene readout from this
one-scene result. No unknown/ambiguous case or difficult example is removed.

## Input-only predictions before any source labels

All40 first-frame candidate tracks are anchors; no container classifier, GT
group IDs, mask-area rule, score gate or TopK selects them. Keep all1,070 query
tracks and34,240 query/time cases in each arm. Query xy and anchor xy are
normalized image-space coordinates from the fixed model tracks, without clipping.
FP64 Euclidean/hypot distance; nearest anchor uses deterministic first input index
on exact ties. Save indices, distances, second-distance margins and tie counts.
Distances/margins are not calibrated Decision probabilities.

Three predetermined arms:

- actual: same-time anchor positions;
- static: each anchor's estimated frame0 position broadcast to all32 times;
- reversed: its same32 track positions in reversed time order, with identities
  unchanged, a temporal misalignment control rather than a plausible scene.

No source questions, role IDs, boxes or visibility flags enter these predictions.
Save every prediction and query/anchor/time record before scoring. A fresh CPU
process must reproduce every saved array shape/dtype/byte. Scorer admits complete
pinned inputs and recomputes all three prediction banks before opening annotations.
Resources: fresh output path,≥8GiB disk reserve; CPU only, no GPU/model work.

## Separate posthoc source-box diagnosis

Open only the pinned training ZIP after prediction admission. Grounded hidden,
cover and group answers refer to track IDs, never option IDs or `is_occluder`
fallbacks. Only range-limited sparse bbox interpolation; unavailable ranges and
source `is_masked` brackets stay explicit. Masked flags do not prove visibility.

Classify **every** seed as a geometric role proxy using own foreground point
inside source box plus model-mask bbox IoU≥.5 at its original observation time.
Multiple available role matches=-2 ambiguous, none=-1 unknown. Store all1,070
labels and source-range availability. A unique available bbox match does not
prove visual recognition or rule out unannotated/occluded objects.

For all seeds matching the hidden bbox, expected anchor role is the grounded
source cover ID. For all seeds matching a group bbox, expected role is that same
group ID. Report both chosen anchor's initial seed-role proxy and its actual
same-time track's source group-box proxy. All arms evaluate the original same-time
anchor tracks for these labels; reversing or freezing selection inputs does not
rewrite the ground truth or anchor identity. Require every source group's range
before assigning a unique current group-box role; otherwise unknown.

Future denominators use sample_time **strictly later** than the seed time;
exclude self/equal-time trivial matches and pre-observation backwards inference.
Unknown/ambiguous/wrong cases remain in every future denominator. No-future seeds
stay in query coverage with rateNone for empty future cohorts. Report separate
hidden-track point-in-hidden-bbox retention under the same range/unknown policy.
Also split the group cohort into frame0 anchors and later-observed candidates:
frame0 queries can match their own identical tracked path in the actual arm,
so their result is a construction sanity check, not independent association
evidence. Keep both strata and the total; do not discard that trivial cohort or
let it justify a claim for later candidates.

This diagnostic has no automatically recognized hidden-item selector. The
posthoc hidden cohort is label-conditioned, potentially one correlated cue/scene;
do not call its point-time count independent QA examples, compute misleading
significance, claim semantic container understanding, or optimize thresholds.
TAPIR sees the full video; on/after-observation is not causal/online memory.
Geometry differs from verified visibility and actual object identity.

## Decision rule and records

Record all arms/hits/denominators/unknowns/ties, seed-role coverage, source box
availability/brackets/flags, input/output/script/config hashes, original commit,
CPU seconds, save/reload and independent audit. If actual geometry fails to
retain the cover relation or controls are comparable/better, close this simple
distance policy without declaring tracker/base-model capacity limits. Favor
explicit semantic/temporal relation supervision or representations only after
its needed source/input semantics and generalization protocol are established.
If actual improves, it permits a wider fixed source diagnostic, not adoption.
Do not repeatedly tune this single observed training scene.
