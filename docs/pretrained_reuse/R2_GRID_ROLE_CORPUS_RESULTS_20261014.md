# All99 training scenes: native grid correspondence support measured

## Result and next action

The fixed [protocol](R2_GRID_ROLE_CORPUS_PROTOCOL_20261014.md) was actually executed
on **every99 existing train clip**, with a fresh complete repeat and an independent
scalar audit. Input/source/config and array consistency pass. This is a weak
bbox-supervision corpus, not a new blind benchmark, QA evaluation or product.

**Do not start another hidden-memory fit from the aggregate pair count.** Although
there are2,111,587 eligible grid pairs, only19 same-hidden pairs come from5 scenes.
Most support is container/group correspondence. This does not repair the sparse
small-object supervision identified after the rejected eight-scene appearance MLP.
All zero-support scenes remain. The label rule, grid pooling and sparse annotation
limits are confounded; these counts are not a pretrained model capability ceiling.

## What was executed

Source/protocol commit`8804056c63af80ba7213e4281e71a3c7e57a49a0`.
Effective configSHA`ea2938eda94f164d82eb5eb22a6b5cdcdea08ead4fe3e7c439e67f0fdf087c28`.
Existing pinned32×16×16×384 TAPIR inputs, unchanged source cuts/native frame order,
all train media/arrays checked before loading training annotations. No new encoder
load/inference, GPU, model training, downloads, validation arrays/annotations,
sealed audit, source-ID input, GT-directed crop or package/Teacher promotion.

```powershell
python -m scripts.prepare_grid_role_corpus --output C:/CodexArtifacts/pretrained-reuse-r2/grid-role-corpus-all-train-01
python -m scripts.prepare_grid_role_corpus --output C:/CodexArtifacts/pretrained-reuse-r2/grid-role-corpus-all-train-fresh-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/grid-role-corpus-all-train-01
python C:/CodexArtifacts/pretrained-reuse-r2/audit-grid-role-corpus-01.py
```

All three separate CPU processes ended successfully. Primary body15.097018s,
fresh15.105586s, independent audit19.403883s; total49.606488s excluding imports.
Peak producer/fresh Windows working set601,866,240/603,549,696bytes (~574/576MiB).
Python3.11.9, NumPy2.4.6, existing Windows environment; zero GPU/cloud cost.
This is preparation timing, not product inference latency.

Primary began2026-10-14T09:27:51.100268UTC, ended09:28:06.197315UTC. No early stop,
metric selection, retry or budget adjustment. Saved99 compressed supervision/index
archives total4,624,922bytes per copy; no cached feature duplication in this corpus.
Weights/media/arrays remain outsideGit. [JSON](R2_GRID_ROLE_CORPUS_RESULTS_20261014.json)
binds local paths and hashes; [CSV](R2_GRID_ROLE_CORPUS_RESULTS_20261014.csv) retains
all99 scene counts.

## Native grid and source support

| Quantity | Count |
|---|---:|
| Distinct known train media | 99 |
| Native sampled times | 3,168 |
| Native cells retained | 811,008 |
| Uniquely role-labelled cells | 37,410 (4.613%) |
| Unknown cells | 764,018 |
| Ambiguous cells | 9,580 |
| Grounded hidden/group instances (video-local IDs) | 381 |
| Instances with any uniquely labelled cell | 258 |
| Eligible-pair unique native endpoints | 37,373 |
| Source track/time cases | 12,192 |
| Available source boxes | 12,052 |
| Exact source annotation times | 779 |
| Interpolated source times | 11,273 |
| Missing source boxes | 140 |

Labels require **whole-cell containment plus no positive-area overlap with another
available task-role box**. Unknown=-1 and ambiguous=-2 are retained, never treated
as negatives. Partial cells are unknown even when their centers lie in a small
object. Task roles include group and hidden; cover is a group subset. Other scene
objects are not part of that namespace. Available-box uniqueness cannot prove
complete object membership or foreground purity, and missing competing annotations
remain a limitation. Source masked flags are stored without declaring visibility.
Missing boxes/annotation starts are not object absence/birth.

## Correspondence support — all opportunities retained in the ledger

| Quantity | Count |
|---|---:|
| All native Cartesian opportunities (56 windows per clip) | 363,331,584 |
| Eligible pairs | 2,111,587 (0.581%) |
| Unknown/ambiguous endpoint pairs | 361,219,997 |
| Same hidden instance | 19 |
| Same group instance | 773,325 |
| Different hidden/group | 2,114 |
| Different group/group | 1,336,129 |
| Same cover subset of same-group | 183,457 |
| Scenes with any eligible pair | 87/99 |
| Scenes with any hidden labelled cell | 8/99 |
| Scenes with a same-hidden pair | 5/99 |

The19 same-hidden positives are0.000900% of eligible pairs. They are correlated
grid pairs, not19 distinct physical objects or independent videos. Twelve scenes
have zero pairs;94 have zero same-hidden pairs. No scene was removed to improve
coverage or a score.

| Scene | Same-hidden pairs | Hidden-supported sampled times |
|---|---:|---:|
| video_1087 | 2 | 2 |
| video_10886 | 8 | 2 |
| video_2775 | 2 | 4 |
| video_68 | 3 | 3 |
| video_8638 | 4 | 3 |

| Time stratum | All opportunities | Eligible |
|---|---:|---:|
| Adjacent sample | 201,129,984 | 1,207,268 |
| Gap8 samples | 155,713,536 | 863,073 |
| First→last | 6,488,064 | 41,246 |

Unknown/ambiguous/intersection counts for each clip/stratum are retained in JSON
and the local reports; they overlap and must not simply be summed. Pair index order
is full eligible Cartesian order in the fixed native window order. No score/target
sampling, duplicate generation, best-mask choice, balancing or new split occurred.
Future fitting requires scene grouping and source-free feature access; corpus
indices/source-role labels are auxiliary supervision, not runtime state.

## Integrity and scientific limits

Fresh repeat: all1,188 arrays match shape/dtype/bytes exactly. Independent reference
reconstructs **all811,008 cell labels** with scalar containment/intersection and
checks all2,111,587 eligible pairs, targets, ordering, complete exclusion ledgers,
strata, role counts, source availability and recorded array inventory. It does not
call the producer geometry/pair helpers. All comparisons pass.

**Source interpolation is not independently reconstructed** by that audit; it
starts from the saved source-box proxies. It therefore verifies the rule/accounting
given those proxies, not physical identity, visibility, annotation correctness or
representation semantics. The prior report includes already-observed development
metadata; no development arrays/annotation archive is opened in this unit. No
evaluation metrics are computed and no selector or final audit is consulted.

Primary resultSHA`a897195ccba8214bfc0d3f91c30d7f33d70910e8ebf4ee211fd463a967b56fd2`;
freshSHA`0c6759083b140fd8e641edf4b5fad9d3ddfde3267732d750878d0e9d522818fb`.
AuditorSHA`ef9be795bce5e9ee1a28dfa7412498fb09792ca9a5155c97e70916c85c36a621`.
The reviewer caught a missing own-config exit hash check; a mutation regression was
RED→GREEN and the guard fixed **before** corpus execution. No post-result rule
relaxation. All protected source/config/media/feature hashes and NumPy RNG checks
pass. Legacy artifacts and rejected experiments unchanged.

25new tests cover geometry/role ledgers/input isolation and config/reference/media
binding. FullCPU1,370passed/1skipped17.52s, Ruff/manifests/fourcatalogs/policy pass;
read-only review resolved its only Important finding before execution. Exact
publication-head CI is checked after push and retained locally.

## Next gate, without shrinking the product goal

The larger known train inventory does not automatically provide hidden-object
correspondence supervision at this fixed pooling/rule. Before a new fit, require
an independently defined way to retain actual small-object/time evidence and
diverse cue→cover supervision. Finer tokens must come from real feature extraction,
not upsampled coarse tokens relabelled as new information. Relaxing eligibility
would change supervision and its noise, not prove additional visible evidence.

This corpus can support auxiliary group-object investigations but is **not adopted
as sufficient hidden-memory training data**. No new learning run is justified here.
Original QA, conditional state reuse, useful native Audio/Image/Text/Video fusion,
OOD/calibration95–98%, full <=200M runtime/package and release gates remain open.
