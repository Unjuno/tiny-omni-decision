# Greatest Hits development tasks and physical input results

## Result and scope

The previously running complete decode finished successfully. All409 acquired
development parents decode, all17,932 supplied event windows are in bounds, and
all32,582 questions with released labels are now materialized. A separate
process rebuilds both task files byte-for-byte. The fixed three-TRAIN input probe
and a fresh-process replay produce18 identical payloads, including raw RGB,
96kHz PCM, independent16/32kHz conversions, actual frame PTS and contact previews.

This is completed data/input work for the proposed conditional AV comparison.
It loads no neural encoder, performs zero optimizer updates and establishes no
gate-quality gain. Physical sound/contact offset, participant/site identity across
dates, near-duplicate scenes and valid complementary intervention labels remain
unresolved. Training admission remains **NOT_ADMITTED**; source review remains
**REVIEW**. Official TEST media/annotations, sealed audit and calibration payloads
remain closed. The product and R2 are incomplete.

The [earlier inventory publication](R2_GREATEST_HITS_DEVELOPMENT_INVENTORY_20261017.md)
correctly described decode as pending at that time. This report records its later
completion without rewriting that history.

## Actual execution

Local artifact root below: `C:/CodexArtifacts/pretrained-reuse-r2/`.

| Operation | Source commit | Output directory | Actual outcome |
|---|---|---|---|
| Complete409 media decode | `7de98c5c2117c984d522d932aa63a91adc8aa628` | `greatest-hits-development-decode-01` | PASS,1095.672s CPU wall time |
| Native task construction | `09f2f6b6785a31973dde9619d5a8ed5600b73d5f` | `greatest-hits-development-tasks-01` | PASS, both complete splits |
| Independent task rebuild | same | `greatest-hits-development-tasks-replay-01` | Exact two-file equality |
| Fixed resampler contract | same | `greatest-hits-input-resampler-01` | PASS,0.078s |
| Three-parent physical input preparation | same | `greatest-hits-input-physical-01` | PASS,0.937s |
| Fresh-process physical replay | same | `greatest-hits-input-physical-replay-01` | PASS,0.938s;18 identical payloads |

Complete decode ran2026-10-16T17:04:34.210192Z through17:22:49.865650Z
(2026-10-17 02:04:34–02:22:49 JST), with four CPU workers. No process interruption
or source-code replacement occurred during it. Earlier acquisition used568.687s;
that is acquisition time, not training time. Task construction did not instrument
its wall time; do not infer a duration from filesystem timestamps.

Python3.11.16/PyAV18.1.0/NumPy1.26.4 were used in the existing VideoPrism venv for
decode and physical input work. Physical receipts also pin libavcodec62.28.102,
libavformat62.12.102 and libswresample6.3.102. CPU tests/task construction use the
existing system Python3.11.9 environment. No package upgrade, GPU fit, paid cloud
usage or model purchase occurred. RTX3080 Laptop16GB was idle at publication;
C: free space measured427,428,048,896 bytes. Those are post-run snapshots, not
peak-resource measurements. Peak RAM was not instrumented for these operations.

### Reproduction commands

Run from `C:/CodexWorktrees/pretrained-reuse-omni` with the pinned clean source.
All output paths must be new; the scripts refuse to overwrite existing outputs.
The exact actual argument arrays are preserved in each `result.json`.

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:PYTHONPATH='C:/CodexWorktrees/pretrained-reuse-omni/src;C:/CodexWorktrees/pretrained-reuse-omni'
$avPython='C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Scripts/python.exe'
$taskRoot='C:/CodexArtifacts/pretrained-reuse-r2'
& $avPython -m scripts.audit_greatest_hits_development_media --inventory "$taskRoot/greatest-hits-train-inventory-01" --media "$taskRoot/greatest-hits-development-media-01" --output '<NEW_DECODE_OUTPUT>'
& $avPython -m scripts.probe_greatest_hits_inputs --phase resampler --output '<NEW_RESAMPLER_OUTPUT>'
# Physical probe intentionally binds the original completed decode/partition paths.
& $avPython -m scripts.probe_greatest_hits_inputs --phase physical --output '<NEW_PHYSICAL_OUTPUT>'
$env:PYTHONPATH='C:/CodexArtifacts/pretrained-reuse-r1/python-deps;C:/CodexArtifacts/pretrained-reuse-r2/native-speech-deps-01;C:/CodexWorktrees/pretrained-reuse-omni/src;C:/CodexWorktrees/pretrained-reuse-omni'
$taskPython='C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe'
& $taskPython -m scripts.build_greatest_hits_development_tasks --inventory "$taskRoot/greatest-hits-train-inventory-01" --media "$taskRoot/greatest-hits-development-media-01" --audit "$taskRoot/greatest-hits-development-decode-01" --output '<NEW_TASK_OUTPUT>'
```

## Complete decoded scope and task accounting

The prospectively fixed date partition remains unchanged.324 official TRAIN
parents sharing timestamp prefixes with official TEST remain quarantined; their
labels were inventoried previously, but their media was not acquired here.

| Quantity | TRAIN | Development validation |
|---|---:|---:|
| Parent recordings |308|101|
| Date groups |21|5|
| Decoded RGB frames |350,462|111,705|
| Decoded video seconds |11,693.748733|3,727.223500|
| Source event rows |13,327|4,605|
| Events with at least one known field |8,102|3,049|
| Material questions,17 options |7,928|2,978|
| Action questions,2 options |7,946|2,987|
| Reaction questions,5 options |7,833|2,910|
| Total questions |23,707|8,875|

Every video is456×256 RGB at decoding; every audio stream is mono96kHz float32.
Maximum decoded absolute audio value is2.166194 TRAIN /2.216857 validation.
These finite source values were retained without clipping or peak normalization.
The eventual frontend must declare any amplitude transform rather than quietly
discarding this information.

Missing fields produce no fabricated target. Missing material/action/reaction
counts are5,399/5,381/5,494 TRAIN and1,627/1,618/1,695 validation. The entire fixed
vocabulary is retained, including carpet/tile with zero positive validation
support. Every question records the underlying parent/event, original source
line, asset SHA, event time,0.5s window, fixed question and full options.

The independent publication audit checks every known source field against its
question/target and every question against its asset/window. TRAIN/validation
overlap is zero for parent IDs, date groups, question/event IDs, raw file hashes,
complete decoded audio hashes and complete decoded video hashes. Hash equality
tests exact content only. They do not establish absence of visually similar clips,
same actors/sites across dates or semantically equivalent events. Repeated
question templates are intentional and are not independent observations.

### Input-free TRAIN-frequency control

One constant distribution per task is calculated solely from TRAIN labels and
evaluated on all corresponding validation questions. This requires no learned
model or modality input. It exposes class imbalance; it is not a gate result.

| Task | Validation n | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Material |2,978|10.31%|3.204009|0.976209|0.115629|
| Action |2,987|61.60%|0.674588|0.480824|0.062200|
| Reaction |2,910|42.23%|1.992648|0.882993|0.314929|

Brier sums squared class errors per example. With constant confidence, ECE is the
absolute confidence/Accuracy gap in the single occupied bin. All four metrics
were independently recomputed from expanded labels. Future learned query-only,
audio-only and video-only controls are still required. Water/dirt/splash/scatter
are much more prevalent in validation; no partition or class was changed to
improve these metrics.

## Actual preprocessing evidence

The existing `native_audio.speech_waveform` only accepts16/24kHz. Routing this
96kHz source through it without an explicit new input path would fail; relabeling
the sample rate would be wrong. The new input helper independently converts the
original96kHz window to16kHz for speech and32kHz for event audio. It does not route
the32kHz path through16kHz. Existing frontend code and all prior features remain
unchanged. Actual encoder integration is not yet executed.

Fixed tone results at amplitude0.25:

| Input tone | Direct16kHz RMS | Direct32kHz RMS |
|---|---:|---:|
|4kHz|0.176776966|0.176776944|
|12kHz|0.001325573|0.176769961|

All outputs have exact8,000/16,000 sample lengths, finite float32 values and exact
repeat/save-reload equality. These are resampling checks, not audio quality scores.

The [prospective input protocol](R2_GREATEST_HITS_INPUT_PROBE_20261017.md) chose
the first3 TRAIN parents by SHA256(`17:`+ID), event row0 regardless of labels.

| Parent | Supplied onset(s) | Audio center error(s) | Source window peak |
|---|---:|---:|---:|
|2015-03-30-00-26-27|1.778125|0|1.143058|
|2015-03-12-20-15-29|1.820208|+0.000000333333|1.112080|
|2015-03-31-02-08-23|1.504146|−0.000000166667|1.343546|

Each produces eight distinct raw RGB frames within the same half-open0.5s interval
as the48,000 source audio samples. Actual selected frame PTS are retained; these
are not synthetic uniform timestamps. Three contact sheets were visually checked:
they show the implement/surface sequence, with motion blur and brief contact.
They cannot determine the precise audio/contact offset or certify the released
material/reaction labels. No manual relabeling or example replacement occurred.

Existing readout positions express relative sampled order, not a guaranteed common
physical AV clock. Future feature binding must preserve the measured time basis
explicitly before interpreting time-offset tests as temporal reasoning.

## Preservation, verification and hashes

Ruff passes; the complete CPU suite is **2,230 passed,1 skipped in33.78s**.
All25 model/dataset/catalog validation commands pass. New input tests cover crop
rounding/boundaries, distinct frame selection and malformed input refusal.
Normal CI does not execute the optional full-media/PyAV experiments; actual
runtime receipts above supply that separate evidence.

An initial contact-preview line exceeded Ruff's line limit in commit0ad352e.
Commit09f2f6b corrected it before the actual probes. No test failure is hidden or
treated as a scientific result. The earlier plot missing-import attempt is
preserved in `greatest-hits-development-verification-01/plot-preflight-failure.json`.

`greatest-hits-development-finish-verification-01/audit.py` independently verifies
all question bindings, media pins, exact split intersections, control metrics,
task rebuild and18-payload physical replay. It rehashes994 preserved older
artifacts with no change and records7,310 current source/result file hashes in
`artifact-hashes.json`. Its own SHA is
`e38aa59c236fb57e04390dd58cca7437716c8ab0422d212f7b1fce42c8099a29`.
This is an explicit preservation scope, not a claim to have hashed every file
on the machine. Existing user worktree changes remain untouched.

| Artifact | SHA-256 |
|---|---|
| `train.jsonl` | `aa1a1e19d555be19455f422396b9f1fc0ec240208354359148b15659eff0cada` |
| `validation.jsonl` | `dc442e9845c986e5aca1cff619d61a91f1ae372448843701a2ed4d7eecc85814` |
| Complete decode receipt | `fa18ca0958846c6143007c252960239a25e0735267f1a35f48d563ed4cfc7e76` |
| Task build receipt | `7ef21536a84904df946f96cd774df9694b99027b1f36d6c8221b8b164fe7a831` |
| Physical probe receipt | `97c91689effda1e7a080b81ddd4be7738718168493137a565f821371d493c0f6` |
| Physical replay receipt | `de370f85e5df2ef0eca564aad8c8100ce9346936d335e2434afcb7876e147163` |
| Independent audit receipt | `5abfd4010d0dcf38b73132ec08baf0e1c621947caee6e181211f686e1ea49652` |
| Full current artifact hash ledger | `73a02406b30f6e4413f1ca33fcdb8c18911108aee29587fe00e8c3a9456d2e79` |

Small public evidence:
[`greatest-hits-development-20261017.json`](../../manifests/pretrained_reuse/greatest-hits-development-20261017.json).
The exact publication-head CI receipt belongs in the verification directory after
push; do not substitute earlier-head CI success for it. Media/arrays/checkpoints
are not added to Git. No merge or product promotion.

## Next measured unit

1. Freeze a bounded encoder-input contract for these same three TRAIN events:
   direct audio branches, explicit frame geometry, real event-relative time basis,
   frozen encoder revisions and no label-derived feature inputs. Run actual
   extraction and independent reload/replay, measuring latency/memory and checking
   that brief event windows are accepted by both audio paths.
2. Establish the limits of source synchronization and grouping, and define valid
   audio-sufficient, video-sufficient and both-required tasks/interventions. The
   released material/action/reaction labels alone do not establish that both
   modalities are necessary. Do not train a fake joint task from arbitrary pairs.
3. Only after the data gate passes, compare original Attention and the fixed
   exchange/gate2×2 under the same data, initialization, long-budget schedule,
   validation rule and four-modality retention checks. Gate magnitudes and
   decision changes on invalid swaps are not quality evidence. Preserve failed
   arms and best/final checkpoints; no premature capacity/long-training conclusion.

The implemented interaction candidate remains197,097,949 total parameters before
quantization. No training result in this report supports preferring it to the
196,254,874-parameter baseline or reaching the95–98% product target.
