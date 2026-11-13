# Track-appearance association diagnosis — executed

## What was measured

The fixed train-only protocol completed on eight reused training scenes, all256
model-produced tracks and all32 source times:65,536 query/time predictions per arm.
No learning, new model, encoder forward, checkpoint selection, development or sealed
audit access. This is a **geometric proxy diagnostic**, not Decision Accuracy.

All input-only predictions were saved before opening the training annotation ZIP
or prior replay's posthoc metadata. All256 first-time descriptors query all256
candidate descriptors at each time with FP64 cosine/first-index ties. Fixed arms:
actual-track appearance, first-position appearance over time, actual features
rolled64 indices with positions unchanged, and original fixed track-index routing.
Grounded group IDs and sparse boxes enter only the subsequent report.

The source has24 game objects across2/3/4-object groups. Boxes are interpolated
inside their annotated ranges, never extrapolated; missing any group range makes
the complete time unknown. All24 ranges were available at these32 sampled times.
Overlapping group boxes and out-of-raster/no-group points stay unknown/ambiguous.
Initial labels:413 unique-group queries,1,632 unknown,3 ambiguous out of2,048.
Time0 self-match is excluded from the future denominator413×31=12,803; future
unknowns/ambiguities remain in this denominator. The other initial queries still
receive all input-only predictions and remain in the saved accounting.

## Results and controls

|Arm|Future same-group bbox hits|Pooled proxy rate|Macro clip proxy rate|Future unknown|Future ambiguous|
|---|---:|---:|---:|---:|---:|
|actual|5872/12803|45.8643%|49.2429%|3259|369|
|static|5256/12803|41.0529%|42.6419%|3489|354|
|misaligned|3073/12803|24.0022%|16.4297%|7562|190|
|fixed_index|6622/12803|51.7223%|54.2593%|3334|476|

|Training clip|Game objects|Initial unique-group queries|Actual|Static|Fixed index|Misaligned|
|---|---:|---:|---:|---:|---:|---:|
|video_10103|2|3|73.1183%|61.2903%|75.2688%|0.0000%|
|video_10186|3|32|39.4153%|30.2419%|44.0524%|11.8952%|
|video_10212|3|55|46.1584%|41.4076%|53.0205%|16.5982%|
|video_10387|4|6|27.9570%|22.5806%|33.8710%|3.7634%|
|video_1087|3|84|56.5668%|49.0399%|64.9386%|36.6743%|
|video_10886|3|103|41.7476%|36.8306%|47.1970%|28.3119%|
|video_10894|3|112|39.8041%|39.1705%|44.7581%|21.1118%|
|video_10903|3|18|69.1756%|60.5735%|70.9677%|13.0824%|

Actual minus static:616 additional bbox matches, +4.8114 percentage points pooled.
However, actual remains750 matches below fixed track routing (−5.8580 points),
with that ordering in all eight clips. Feature-only misalignment degrades the
result, supporting a correspondence signal in the features/positions. It does
not establish correct object semantics or target-hidden-item association.

The actual arm's conditional resolved-only hit rate is64.0%, versus its primary
45.8643%. The difference demonstrates why3,259 unknown and369 ambiguous future
predictions must stay in the denominator. Do not present64.0% as the primary result.
No zero-norm initial query or exact maximum-similarity tie occurred in these clips;
the tested zero/tie handling nevertheless retains such queries deterministically.

One object has no initially unique model-grid point:video_10103 group5. Report
23/24 initial group coverage separately; that unsupported object is still present
in source/group records and no QA or group is deleted. Bbox coverage at time0
does not establish discovery of a hidden target or its later container.

## Interpretation and next action

Do not adopt nearest-descriptor reassignment as a tracking/Decision improvement:
it trails existing model trajectory routing on every clip. Motion-aligned sampling
beats static-position sampling here, but does not reverse the earlier full-readout
generalization failure. Do not launch a new fit or repeat/extend that99-scene
training solely from these feature differences.

Next establish input-only target/prelude and covering-object association, preserving
variable object groups and actor/viewpoint semantics. The missing initial grid
object motivates using the already tested model-produced proposal/track references
for that coverage question; no GT-match initialization or target-based seed search.
Existing mask-query24/24 initial opportunity uses a different query cohort and is
not a causal quality comparison against this413-point subset. Any new intervention
needs a frozen same-task control and genuine scene-generalization evidence.

Sparse interpolated boxes may be wrong between annotated frames and are not visible
pixel/semantic point-tracking truth. These scenes are reused training data; points
and times within a scene are correlated. No new blind quality result or population
confidence interval is claimed. Accuracy/NLL/Brier/ECE for supplied-option Decisions
were not measured here; cosine scores are not calibrated option probabilities.
The full <=200M resident pre-quantization Omni model and95–98% modality/joint
quality target remain unchanged and unmet. No quantization, Recovery, paid compute,
main merge or product promotion.

## Reproduction, resources and safeguards

- Source commit `17fb965341bf46d7ccabd9117ef93985022c3bf4`; protocol/code committed before execution.
- Started `2026-10-14T03:22:56.369210+00:00`, ended `2026-10-14T03:23:08.037507+00:00`, exit0.
- Primary body11.670031s (input-only stage6.900465s);
  fresh body11.652105s; independent audit0.568854s. These include
  input hashing/annotation parsing, not raw-video inference or isolated throughput.
- CPU only, Windows/i7-12700H, Python3.11.9, NumPy2.4.6; existing
  environment. BLAS/OpenMP environment requests4threads; actual BLAS thread count
  was not independently measured. No dependency upgrade/install.
- Windows peak working set 978,051,072bytes primary,
  977,956,864bytes fresh.8GiB disk reserve and300s
  between-clip/final admission checks passed. No hard watchdog is claimed.
- Added trainable/resident model parameters0; optimizer updates0; cloud cost0.
- Fresh process matched192 saved arrays by shape/dtype/bytes, including every
  predicted index/score/margin/tie/zero flag and every posthoc label/availability.
  Repeats/readback and NumPy RNG/nonmutation/source-file hash gates passed.
- Independent audit recomputed all32 clip/arm metric groups and per-object
  counts, aggregate denominators, unknown accounting and fresh bytes.
- Before execution, review caught partial-group-coverage falsely resolving a label.
  Its staggered-range regression failed first, then passed after the conservative
  unknown fix. Protocol/config/replay guard were aligned; no result-based changes.
- CPU tests1092passed/1skipped15.82s; Ruff, all manifests, four catalogs and policy
  audit pass. Exact publication-head GitHub Actions checked after push.
- Config SHA `28d2d8f1d86a07eba2251c42b629d3f5f2432c9f3789915353e3ecf7141016d5`.
- Primary result SHA `a09a5c6d1ac7aac21699d65cc08a0b9eaafb3619b99e9eed6c1077ad5cd6ee28`.
- Fresh result SHA `cb760fb435ff98426166adc6e6b402ae4207e810e85bb51e2ac8139510b8c4c1`.
- Independent audit SHA `66fba80bdba9685aa8423ef78b27391f6d22f9b518d4effa06a01e25eef680d8`; script SHA `30d7a72ba29b93ce651108a4e62846bd76c2f686edf8e5135b70c2a6723f942c`.
- Artifacts `C:/CodexArtifacts/pretrained-reuse-r2/track-association-diagnosis-01/`, `C:/CodexArtifacts/pretrained-reuse-r2/track-association-fresh-replay-01/`,
  `C:/CodexArtifacts/pretrained-reuse-r2/track-association-independent-audit-01.json` and `C:/CodexArtifacts/pretrained-reuse-r2/audit-track-association-01.py`.
  All raw predictions/posthoc labels and unsuccessful/previous experiments are
  retained locally. Only code/tests/protocol/small result metadata enter Git.
- Protected input file hashes before/after agree; original feature arrays and
  training ZIP remain intact. Legacy weights/configs/checkpoints were not edited.

```powershell
$env:OPENBLAS_NUM_THREADS='4'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
python -m scripts.diagnose_track_association --output C:/CodexArtifacts/pretrained-reuse-r2/track-association-diagnosis-01
python -m scripts.diagnose_track_association --output C:/CodexArtifacts/pretrained-reuse-r2/track-association-fresh-replay-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/track-association-diagnosis-01
python C:/CodexArtifacts/pretrained-reuse-r2/audit-track-association-01.py
```

Existing output directories are refused; use fresh paths for a rerun.
