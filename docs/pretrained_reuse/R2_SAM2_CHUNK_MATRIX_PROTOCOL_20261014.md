# R2 SAM2 fixed chunk matrix — 2026-10-14

## Question and scope

The preceding64-object native-head batch was approximately5.04× faster in
cached forward windows, but score maxerror1.335144e-5 exceeded the unchanged
1e-5 guard. Preserve that failure. Test fixed chunk sizes **16,32,64** on the
same64-point known train10903/frame0 prefix, before any broader input corpus.
This is a new numerical/engineering experiment, not semantic quality selection.
The prefix is chosen for its archived floating reference, not GT or detection.

## Unchanged inputs and guards

Same frozen38,962,498-parameter native SAM2 checkpoint/revision, processor,
single positive-point prompts, cached native image embeddings with initial
no-memory addition, FP32/sdpa/TF32off/deterministic seed17/CPU4. No encoder,
projector or decoder learning, source annotations, dev/sealed data, download,
environment upgrade, quantization, cloud or intervention in live processes.
All prior code/model/config/corpus/artifacts remain read-only.

For each arm, every64-object mask/score is retained in original order. Native
mask best-IoU gather and score<=0→mask=-1024 semantics are unchanged. Contiguous
chunks16/32/64 mean4/2/1 decoder calls; native wrapper uses64 single-object calls.
Instrument actual head call object counts and image encoder calls. No omitted
tail/object, changed mask threshold, score rule, NMS0.7 or foreground seed.

Same guards: raw-mask max absolute difference≤1e-3; score difference≤1e-5;
relative tolerance0. All64 threshold0 masks, boxes, own-foreground seeds,
nonempty flags and prefix NMS identities must be exact. Floating byte equality
is reported separately. Every arm retains its own result, including failures;
no tolerance relaxation, runtime batch retries, quality/seed-based selection.
The64 arm must reproduce the preceding failed batch arrays byte-for-byte.

## Actual execution and verification

New `sam2_chunked_masks.py` helper and separate `probe_sam2_chunked_decoder.py`
runner; protected helpers/runners remain unchanged. CPU tests use the actual
installed tiny native heads without pretrained/GPU loads: ordering, tail,
complete-input admission before partial forward, call counts, freeze, state,
input and RNG preservation. Optional ML packages skip in ordinary CI.

Warm every path. Two measured64-object windows per arm with fixed order:

1. native,chunk16,chunk32,chunk64;
2. chunk64,chunk32,chunk16,native.

Same work/one cached image, zero measured encoder calls. CUDA synchronization
around each measured call; setup/preprocessing/output D2H excluded, body time
separate. Save16 raw/score arrays; in-process warm/repeated calls exact; native
calls exact versus original archived64. Fresh separate process must reproduce
all16 arrays byte-for-byte under identical hash-bound config. Independent CPU
audit rechecks arrays, errors, mask/presence decisions, fixed count/order,
source/config hashes and unchanged guard outcome. No task labels are opened.

Fresh output only; GPU free≥12GiB, disk free≥8GiB+256MiB. Body≤180s and Torch
peak allocation≤8GiB checked after calls, not watchdogs. Faults preserve
incomplete attempts; no inference retries with altered settings. Record global
peak allocation/reserve, parameter/model hashes, RNG, call counts, environment,
source commit, all windows, output hashes and unsuccessful arms.

## Decision

Passing a chunk's64-prefix guards permits a separately fixed full1024-grid and
multiple-time admission comparison only. Never replace a protected corpus from
this prefix. Prefer the fastest admitted arm for that next engineering gate
only; every matrix result remains reported. If no arm passes, stop this drop-in
batch path and use a separately specified derived-input protocol or native
reference for semantic cue/cover learning. Do not keep searching chunk widths.
This experiment cannot establish Decision Accuracy/NLL/Brier/ECE, semantic
binding, native joint benefit, end-to-end latency or95–98% Omni quality.

## Execution

```powershell
python -m scripts.probe_sam2_chunked_decoder --output C:/CodexArtifacts/pretrained-reuse-r2/sam2-chunk-matrix-prefix-01
python -m scripts.probe_sam2_chunked_decoder --reference C:/CodexArtifacts/pretrained-reuse-r2/sam2-chunk-matrix-prefix-01 --output C:/CodexArtifacts/pretrained-reuse-r2/sam2-chunk-matrix-fresh-replay-01
```

Do not reuse these paths for a later attempt. Keep weights/media/NPZ outside Git.
