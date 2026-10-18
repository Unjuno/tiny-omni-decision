# Native training preparation: persisted video tokens and measured source gaps

2026-10-16. **Video cache complete; common-model training preparation incomplete.**
The full goal remains the same <=200M native text/image/audio/video Decision
candidate, with conditional, temporal, joint and OOD quality. No model fitting,
quality evaluation, checkpoint selection or audit unsealing occurred here.

[Compact measurements](R2_NATIVE_TRAINING_PREPARATION_20261016.json) ·
[Fixed cache protocol](R2_NATIVE_VIDEO_CACHE_PROTOCOL_20261016.md) ·
[Candidate integration](R2_NATIVE_CONDITIONAL_CANDIDATE_20261016.md).

## Actual work completed

The historical eight-frame CLEVRER extraction retained full tokens only in RAM.
It left pooled features and per-scene raw-token digests, which are insufficient
to train the new conditional attention readout from a persistent full-token bank.
The existing Lombard video bank uses16 spatially pooled frames; it cannot silently
replace the candidate's8-frame full-token profile.

Re-extracted **all256 TRAIN /200 development-validation scenes** with the same
frozen VideoPrism weights and exact existing decoder/preprocessor. No scene,
question set, frame count, dtype or model weight was changed. Each of456 saved
FP32 `[2048,768]` arrays exactly matches its historical raw-token SHA-256, passes
save/reload, and was independently re-read afterward for shape, dtype,
finiteness, file digest and raw-array digest. No new quality result was computed.

| Measurement | Result |
|---|---:|
| Completed video files | 456 |
| Independent scenes, train / validation | 256 / 200 |
| Token files on disk | 2,868,962,304 bytes |
| Extraction wall time | 568.0424 seconds |
| Sampled process-tree peak RSS | 2,356,662,272 bytes |
| Worker OS peak working set | 4,153,794,560 bytes |
| Model updates / paid compute | 0 / 0 |

Sampling on token receipt or a5s timeout misses short memory peaks, as the larger
OS worker peak demonstrates. Do not call the sampled value a true peak bound.
The frozen limits were1200s,6GiB sampled process-tree RSS, and3GiB disk reserve;
about5.05GB C: free remained after publication preparation. This is CPU JAX
extraction, not a GPU throughput benchmark. Worker JAX/jaxlib0.6.2, NumPy1.26.4,
TensorFlow2.18.1 and PyAV18.1.0; parameter count114,365,184. No upgrades.

Execution started2026-10-16 13:10:26 JST and finished13:19:54 JST normally.
The dedicated parent/worker terminated normally; no unrelated process was stopped.

## Source and preprocessing audit

| Source | Existing TRAIN / development validation | Finding for the same candidate |
|---|---|---|
| Conditional text | 18,432 /10,732 questions | Full language feature/query extraction needed; long states exceed the current512-token limit |
| Clevr-4 image | 8,424 /512 images | Existing17x768 cache hashes pass; one fixed TRAIN image exactly matches native integration output |
| Lombard speech | 4,080 /889 utterances;36 /9 speaker groups | Existing speech tokens differ from native eager output; environmental-audio features are also needed for the combined audio path |
| STARSS23 environmental audio | 1,955 /486 windows;49 /21 recordings;6 /2 rooms | Existing event features are a separate path; speech features and a unified audio profile are needed |
| CLEVRER video | 256 /200 scenes | Full8-frame tokens now persisted and exactly verified |
| Genuinely joint AV tasks | Separate existing catalogs | Native8-frame paired-video features and the common task/selection contract remain open |

These sources are development material with prior observations; combining them
does not create an untouched evaluation. The table does not establish broad
speech understanding, natural-image quality, Japanese coverage or joint benefit.
Existing rights audits remain prerequisites; this audit does not grant new rights.

### Text length

Original local DeBERTa tokenizer, no truncation:

- Binding TRAIN512 unique states: maximum38tokens; numeric TRAIN1024:115tokens.
- Legacy TRAIN1090 unique text states: maximum753tokens; **28 exceed512**.
- New/old validation128/384 states: maximum38/115tokens.
- Legacy validation1244 states: maximum824tokens; **10 exceed512**.
- All **22,328 distinct question/candidate pairs**: maximum316tokens; none exceed512.

Plain-language observations and JSON observations both occur. An initial
inspection assumed every state was JSON and failed on plain text; the committed
auditor handles both without altering observations. The next input implementation
must preserve long states rather than drop rows or silently truncate them.
Whole-readout8192-token capacity is sufficient for these measured lengths;
cross-chunk contextual quality still requires evaluation after implementation.

### Identity versus attribute combinations

The text train/validation normalized-state intersection is0; the image
train/validation raw-media-hash intersection is0. The combined text/image
inventory has317 overlapping **four-attribute signatures**:301 image-to-image,
16 text-to-image. These are unique signatures, not317 duplicate images.
Different images can have the same color/count/shape/texture while depicting
different object positions. Signature equality alone does not establish media
leakage. It does show that existing validation is not wholly an unseen-attribute-
combination test. Preserve the full original development cohorts, distinguish
these scopes, and define composition/OOD claims separately before selection.

### Audio compatibility and coverage

For fixed TRAIN `s11_l_bbab2n`, the same110x384 speech token bank differs from
the native integration's eager path by maximum absolute **4.9591064453125e-5**.
Both paths used frozen FP32 weights; the audit does not identify the numerical
cause. Do not loosen a replay threshold and assert exact interchangeability.
Align/re-extract the intended audio profile, preserving existing artifacts.
The one-input image/speech comparisons are not full-corpus fresh-encoder parity.

STARSS23 has **zero bell-positive training windows**, while validation has32;
validation also has zero water-tap positives. Only two validation rooms are
independent room groups. Retain all classes and disclose unsupported/imbalanced
quality slices; negative-only accuracy is insufficient evidence of useful event
understanding. Additional approved supervision or explicit unresolved coverage
is needed, without inspecting the final audit.

The source audit hashes the existing speech bank as opaque bytes, including its
reserved calibration portion, but loads feature slices only for the one TRAIN
comparison. No calibration labels/features were used for model selection; no
sealed-audit file was accessed.

## Exact execution and provenance

Cache source commit: `4a9823ce7d6a3987a241a872ae8d0ce5d2d714fc`.
Audit source commit: `7042a4a7c6e76f179ecdc368aa1c361c4e174aea`.
The exact config pins and456 token-file hashes are in the local cache receipt;
the source auditor also records its input hashes and tokenizer environment.

```powershell
$env:PYTHONPATH = 'C:/CodexWorktrees/pretrained-reuse-omni/src;C:/CodexWorktrees/pretrained-reuse-omni'
$env:OMP_NUM_THREADS = '4'
$python = 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe'
# Historical commands; use a fresh output directory for any new invocation.
& $python -u -m scripts.capture_native_training_video --output C:/CodexArtifacts/pretrained-reuse-r2/native-training-video-cache-01
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
& $python -u -m scripts.native_training_readiness --output C:/CodexArtifacts/pretrained-reuse-r2/native-training-source-audit-01
```

| Receipt | SHA-256 |
|---|---|
| Video config | `82fc5d8b123bab7ae5e00509b5cf7dd6f4438c3316609741e5c7780d49661a35` |
| `native-training-video-cache-01/result.json` | `11358d26a6628f113caf4407c5297f6c492f8e263ed5eb2c688eab1fe6d1a089` |
| `native-training-source-audit-01/result.json` | `9f202bdbfad5660fedc95c71bcc5891bac00f1967a695051c981691877b342b8` |

All directories are under `C:/CodexArtifacts/pretrained-reuse-r2/`.
`native-training-cache-report-01/summary.json` contains the compact report;
`native-training-cache-verification-01/` contains the CPU/Ruff/manifest receipts
and final publication CI evidence. The source audit's13.24s counter precedes its
final input rehash; it is not a complete process-wall or model-latency measurement.

## Verification and next execution unit

Nineteen new CPU tests failed before implementation and pass afterward. Full
CPU pytest: **1,929 passed,1 skipped**; Ruff and25manifest commands pass.
Extraction code commit `4a9823c` has successful
[CI1801](https://github.com/Unjuno/tiny-omni-decision/actions/runs/38849785691).
Final publication commit CI is recorded separately after push.

Next implement lossless long-observation encoding and fixed language caches,
then complete both audio feature paths under the same pinned runtime profile.
Keep every development row and the new video cache unchanged. Freeze the common
native/conditional/joint sampling, training budget, retention and checkpoint
selection rules before the first shared-readout fit. Native joint coverage and
all95–98% task gates remain open; this preparation is not model training success.
