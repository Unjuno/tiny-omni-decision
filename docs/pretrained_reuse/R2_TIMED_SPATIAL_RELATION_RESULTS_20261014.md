# R2 timed spatial relation — executed comparison, 2026-10-14

## Result and decision

**Execution/reproducibility PASS; nearest-distance policy not adopted.**
On one already observed Perception Test train regression scene, all1,070 timed
model queries were routed to all40 first-frame candidate anchors at all32 times.
All34,240 cases per arm were saved before source labels; a fresh process reproduced
all18 prediction arrays shape/dtype/byte exactly. No model/GPU inference,
training, validation, sealed audit, download or cloud work was performed.

Actual anchor motion helps later same-group bbox association relative
to fixed/reversed positions. It does **not** establish the critical hidden-item
to covering-object relation: the one posthoc hidden-cue seed has0/18 future
correspondences to the source cover role under actual nearest routing. Keep that
failure and all controls; do not tune distance/thresholds on this scene, adopt
this policy, infer a general tracker/base-model ceiling, or extend the failed
99-scene readout fit. Cue-to-cover binding and identity memory require evidence.

## Fixed inputs and safeguards

[Pre-execution protocol](R2_TIMED_SPATIAL_RELATION_PROTOCOL_20261014.md).
Code commit `3fd009189deda462ff31b8c9a491698d327c49e5`; script
`scripts/diagnose_timed_relations.py`; pure helper
`src/tiny_omni_decision/spatial_relations.py`. Pinned full558-frame TAPIR outputs,
the unchanged32 sample IDs and all timed masks from10903; no re-extraction or
change to those artifacts. All40 original-time0 queries remain anchors, including
parts/background/unclassified masks and duplicated coordinates.

Three fixed selection inputs: actual same-time anchor xy, frame0 xy broadcast,
and reversed-time anchor xy. All use normalized image-space FP64 Euclidean/hypot
distance with first-input-index exact ties; no GT role, classifier, filtering,
TopK, mask-area heuristic or confidence gate selects predictions. The scorer
admits pinned inputs and independently recomputes all prediction arrays before
opening the original training ZIP. A read-only reviewer found no Critical or
Important issue before execution.

Posthoc source roles are hidden7, cover3, group1/2/3. Every seed receives a
range-limited geometric label using its own point in the source box and mask-box
IoU≥.5; source masked flags are retained but not visibility labels.93 seeds have
one available-role bbox match:92 group seeds and1 hidden seed;974 unmatched
and3 ambiguous seeds remain in the full inventory. The unmatched cases include
valid non-role scene objects/parts; they are not974 detection failures or negative
training examples. Hidden-box availability is absent at13 of32 retained times;
all source brackets/availability/flags are retained in the publication JSON.

The hidden cohort is determined **after predictions**, not an input-only small-item
recognizer: query501, original observation234, proposal592. None of those values
was used to pick inference candidates. It is one correlated cue/scene, not18
independent examples. Only times strictly later than234 enter its18-case metric.
Both initial-anchor seed role and that same selected anchor's actual current
group-box role are reported. The controls retain actual anchor identities/tracks
for labels; their frozen/reversed selection positions do not rewrite truth.

## All fixed comparisons

### Posthoc hidden-cue to source-cover proxy

| Selection policy | Initial anchor role3 | Current anchor in group3 bbox | Current unknown / ambiguous / known wrong |
|---|---:|---:|---:|
| Actual motion |0/18|0/18|12 /1 /5|
| Static initial positions |2/18|5/18|7 /3 /3|
| Reversed anchor motion |4/18|3/18|11 /3 /1|

The hidden query's own tracked position lies in the source hidden bbox at2/18
future samples;16/18 remain outside/unknown. Its source bbox is available for
every one of these18 samples, so missing future annotations do not explain that
particular unknown count. Sparse-box membership remains a geometric proxy,
not verified visible point correspondence or semantic object identity.

At observation234 the annotated cover3 box is at image y0.074–0.196 while the
hidden box is y0.560–0.619; the source role need not be the closest point at the
reveal. The actual nearest anchor belongs to source group2 at234/252/270. This
shows why current point proximity alone is not a sufficient cue-to-cover semantic
rule. It does not prove annotation visibility, exact causal history or universal
failure of pretrained temporal representations.

### Later-observed group seeds (frame>0)

89 query seeds,1,270 strict-future point-times; duplicates and unknowns retained:

| Selection policy | Initial anchor same-group role | Current anchor same-group bbox | Current unknown / ambiguous / known wrong |
|---|---:|---:|---:|
| Actual motion |1,091/1,270 =85.91%|878/1,270 =69.13%|131 /165 /96|
| Static initial positions |211/1,270 =16.61%|473/1,270 =37.24%|466 /33 /298|
| Reversed anchor motion |337/1,270 =26.54%|438/1,270 =34.49%|394 /80 /358|

This positive geometric control is separate from the negative hidden-cover result.
These correlated mask/point-times are neither QA Accuracy nor a held-out estimate;
no confidence interval or statistical independence is claimed.

All92 group seeds, including the3 original anchors, have1,363 future cases:
actual initial/current hits1,184/950, static235/516, reversed363/471. The3 frame0
anchors have93 strict-future cases and can route to their own identical tracks:
actual initial-role93/93 is a **construction sanity check**, not independent
association evidence. Actual current group-box72/93, static24/43 and reversed26/33
initial/current hits are retained separately. A future-time restriction alone
does not remove that self-trajectory tautology.

Exact-tie cases across all34,240 cases: actual6,963, static4,162, reversed6,002.
All were kept, with deterministic first-input-index routing and margins/tie counts;
those margins are not calibrated Decision confidence. NLL/Brier/ECE are not
reported for this nonprobabilistic geometry diagnostic. Omni probabilistic QA
and the95–98% task/OOD/joint goals remain unqualified.

## Reproducibility, cost and checks

| Separate process | Measured body seconds |
|---|---:|
| Input-only prediction |2.759339|
| Fresh full prediction replay |2.801273|
| Admitted separate posthoc scorer |6.942033|
| Independent all-case CPU audit |4.788475|

Sum17.291119s. Terminal exit0 observed for every execution. Python3.11.9,
NumPy2.4.6, CPU only; no new parameters, weights or feature extraction. This cached
diagnostic timing does not replace the preceding native SAM discovery and TAPIR
runtime or establish end-to-end Decision latency. Local electricity/peak RAM
were not measured; cloud cost0. Existing GPU/training/unrelated processes and
legacy artifacts were preserved.

Independent audit uses separate square/sum/sqrt distance arithmetic and compares
all102,720 predicted indices/ties; nearest-distance difference0. It recomputes
all1,070 seed labels from original source boxes/banks, current anchor/hidden
memberships, all24 cohort/proxy summaries, the hidden-retention count, and all18
fresh prediction arrays. Source ZIP was opened only after full prediction checks.
No production geometry helper is imported by this audit.

Local Ruff and all model/dataset manifest, four catalog and policy-audit commands
passed. New25 geometry/denominator tests and7 source-role parsing tests
had missing-API failures before implementation, then passed. An initial collection
failure for the absent geometry module was corrected by creating the empty module
before the RED run. The source cover question suffix was checked against the
previous pinned source-audit code before execution. Full CPU suite **1,171 passed,
1 skipped,16.06s**. Model/GPU tests remain outside CI. All attempts/controls were
preserved; no experimental seed/threshold/policy retry was made after results.

## Artifacts and next gate

Small [JSON](R2_TIMED_SPATIAL_RELATION_RESULTS_20261014.json) contains every report,
source-time proxy and all summary tables. Its SHA-256:
`93981f37571fb504cc9e93ee63a336467d31850d7b42ad51a8e8903550822ee7`.
[Per-time hidden-cohort CSV](R2_TIMED_SPATIAL_RELATION_RESULTS_20261014.csv) retains
all19 on/after-observation rows (including the excluded equal-time row), selected
anchor IDs, both role proxies, unknowns and source masked brackets. CSV SHA-256:
`1f2cb608cded22feab3863b171314631b9397f9cfa2bfa4ddd2b0af5601bb750`.

Local root `C:/CodexArtifacts/pretrained-reuse-r2/`:

- `timed-spatial-relations-prediction-01/`;
- `timed-spatial-relations-fresh-replay-01/`;
- `timed-spatial-relations-posthoc-01/`;
- `timed-spatial-relations-independent-audit-01.json`;
- preserved `audit-timed-spatial-relations-01.py` and publisher script.

Primary prediction NPZ SHA-256 (same fresh bytes):
`a74c71e5fc88094dced91842079eb596b97a352feaf55c0203d5b56391f9bad9`.
Posthoc result SHA-256:
`97a039cbbbf188873e300593b9f4e4b6dce968ef97506dc43f3a9e90f1a7dc7c`.
Independent audit script SHA-256:
`e0d6bfc7e4eb917ad77c287dd49f2491a7bd58e1b24a45ad78b08b2e3f4029d6`.
Weights/media/NPZ remain outside Git.

Next require a supervised/validated semantic cue-to-cover association and
persistent cover-identity representation, instead of following the small-item
point by proximity. A wider source/input protocol is needed before
learning or claiming generalization; source role IDs/boxes must never become
inference seeds. Native multimodal benefit, total resident Omni package and
95–98% quality remain open. No current component is promoted by this diagnostic.
