# Native visual extraction: launched, not yet complete

**Completion update:** The5,367-source native job finished with all integrity
checks; the fixed readout study also completed. See the
[final results](R2_LOMBARD_GRID_RESULTS_20261014.md). This original launch snapshot
and its JSON remain preserved below.

The original frozen native JAX/Flax VideoPrism-B is actually extracting all
5,367 common-valid AV assets. This report's adjacent JSON is a **start/progress
snapshot**, not a finished result. Current state is the local progress file and
live PID74692; PID alone must not later be treated as evidence of running.

Output`C:/CodexArtifacts/pretrained-reuse-r2/lombard-grid-videoprism-native-features-01/`.
The smoke and full jobs have separate outputs. Full configSHA
`c246c99d8155afe45a4b0e249e25f7c7b003c08306a050b590661702c70f94a4`;
exact copied runnerSHA
`965099b6b9bef40dd033aaa862f66725bcb2694fddab889aff970edf88ad558d`.
Repo HEAD at launch`bbd194e454bf5191cf2ea189ea61e91a5cbd1788`.
The runner uses the pinned native source and existing pure progress-IO helper;
it does not import the experimental Torch port. This is not a claim that all
unrelated worktree files were immutable or clean at launch.

## Actual preflight and fixed job

The new runner's first four-training-source smoke failed before its first forward:
the prior media audit stores(width,height), while decoded arrays have(height,width).
The failed artifact/runner are retained. Correcting that assertion orientation
passed smoke02, not weakening an image-shape or parity check. All four outputs
match the original smoke bitwise; repeated full tokens are bitwise; parameter
hashes unchanged; saved framewise/global features reload exactly.23.23s, peak
working set4,323,475,456bytes. The adjacent JSON preserves the full smoke result.

Full job fixes native16-frame actual-PTS sampling/288 crop/FP32, all archive,
media, layout and processor/model source checks. CPU-only, no encoder training.
Keep chronological16x768 spatial-mean features and768-dimensional global means.
These are pooled research caches, not full4096-token stores or proof of retained
fine lip motion. Retain source indices/timestamps and preprocessing hashes.

Wall cap6h, disk reserve3GiB. Every source hash/decode/PTS/finite check is mandatory;
repeat first/every128th/reference source outputs, report the exact replay count.
After finishing, require all5,367 IDs, feature hashes, unchanged parameters and
complete saved-feature reload checks. A partial progress file or allocated NPY
shape cannot substitute for completion. Current snapshot192/5,367 at444.84s;
later throughput and completion must be measured. Local compute only; cloud cost0.

Run with the approved existing system Python and existing dependency directory:

```powershell
python -c "import sys,runpy;sys.path[:0]=['C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Lib/site-packages','C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source'];sys.argv=['lombard_grid_videoprism_features.py'];runpy.run_path('C:/CodexArtifacts/pretrained-reuse-r2/lombard_grid_videoprism_features.py',run_name='__main__')"
```

Do not reissue this command while the process is live or reuse its output path.
The application-policy-blocked virtualenv launcher remains untouched. No upgrades.
In this experiment, training/validation/sealed data of old Teachers remain
unchanged and unopened.

## Frozen next readout comparison, no quality yet

Before any readout quality metric, freeze
`configs/pretrained_reuse/lombard-grid-pooled-slot-v1.json` and
`manifests/pretrained_reuse/lombard-grid-source-20261014.json`.
The corpus publisher's CC-BY4.0 declaration applies to its released corpus;
all local source/archive/layout SHA256 pins are explicit. No immutable upstream
Git revision is invented to fit the legacy40-character manifest schema. This is
a separate Path A research source, not a change to old Teacher catalogs.

First compare the same77,235-parameter pooled six-slot head for audio-only,
video-only and joint; seed17,24epochs, identical unique4080 train398calibration889
validation and full51-option grammar. All25 letters remain despite the legacy
DecisionExample20-option cap; use native scoped JSON, not a reduced alphabet or
a false legacy-adapter compatibility claim. Input is frozen raw-media features
and presence masks; filenames/transcripts/alignments are never features.
Train-only normalization, earliest raw validation macro-NLL best, separate final,
per-slot calibration on calibration only. No additional seed or epoch search.

Report6 per-slot Accuracy/NLL/Brier/ECE, macro/minimum slot Accuracy, all-six-correct
utterance Accuracy, p/l conditions/speakers, full curves and train/validation gap.
Keep best/final consumption counts distinct. Selected joint matched/mismatched
and absent-stream controls plus speaker-group paired bootstrap(10,000,seed17,
percentile95%) diagnose actual contribution without manufactured noise. Selection
uses development validation; uncertainty is not blind final-audit evidence.

This first baseline deliberately pools temporal features. It cannot itself
establish temporal order reasoning; its results will guide any separately
specified chronological-token follow-up. No readout training or new quality
result is claimed yet. Native command slots do not close the generic text/image/
environmental audio/video/sensor task battery or the95–98% final goal.
