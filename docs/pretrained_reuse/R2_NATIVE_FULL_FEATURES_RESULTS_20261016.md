# Full native language and audio features: completed input preparation

2026-10-16. **All registered language and audio examples are extracted and pass
independent full recomputation. The shared Decision readout remains untrained.**
This completes the language/audio observation-feature unit, not R2, model quality,
calibration or the final <=200M/four-modality95–98% objective.

[Machine-readable measurements](R2_NATIVE_FULL_FEATURES_RESULTS_20261016.json) ·
[Language protocol](R2_NATIVE_LANGUAGE_CACHE_PROTOCOL_20261016.md) ·
[Audio protocol](R2_NATIVE_AUDIO_CACHE_PROTOCOL_20261016.md) ·
[Previous smoke and disk failure, preserved](R2_NATIVE_LANGUAGE_AUDIO_RESULTS_20261016.md).

## Executed scope

The old language attempt stopped before loading the model at a disk-reserve gate.
This continuation rechecked available disk/GPU/processes, then used new output
directories and the same frozen numerical profile. No failed output was reused.

| Input | Completed extraction | Complete fresh-process replay |
|---|---|---|
| Language observations | 4,382 unique states, retaining all38 states longer than512tokens | All4,382 token sequences exact |
| Language condition/candidate pairs | 22,328 distinct pairs, preserving29,164 original questions | All22,328 vectors exact |
| Native audio | 7,410 complete waveforms through both Whisper encoder and EfficientAT | Every speech/event value and per-example hash exact |

The maximum observed feature difference is **0** in both complete replays. Model
state hashes before/after match. All1,135 distinct input/source pins and all saved
output hashes were independently checked after completion. Persisted banks were
also reopened for shape, FP32 dtype and finite-value checks. The exact encoder
profiles and revisions are unchanged; there are no optimizer updates.

The long-text path preserves original tokens through510-token content windows
with32-token overlap. It does not establish full-context attention equivalence
or cross-window reasoning quality. The original single-pair query convention and
short native text parity pass. Audio retains physical Whisper tokens and the
separate event vector without transcripts, normalization or changed frame profiles.

## Audio corpus and implementation checks

| Source | TRAIN examples / assets / groups | Validation examples / assets / groups |
|---|---|---|
| Lombard | 4,080 /4,080 /36 speaker groups | 889 /889 /9 speaker groups |
| STARSS23 | 1,955 windows /49 recordings /6 rooms | 486 windows /21 recordings /2 rooms |

Together:7,410 examples,5,039 physical audio assets. All7,410 rate+decoded-PCM
identities are unique, with zero cross-split exact PCM overlap and zero all-zero
waveforms. This is exact-content integrity, not perceptual deduplication.
STARSS room separation does not establish global speaker separation.

The initial unfinished iterator did not itself verify STARSS physical-file bytes
and hashes. Regression tests demonstrated missed raw-content/size changes and a
change after consumption. It now checks each physical recording before decoding
and after the full iteration, and rejects conflicting identities. The iterator
closes window readers explicitly. Existing Lombard raw-member and decoded-PCM
hash checks, speaker/phrase grouping and STARSS room/recording/window gates remain.

The storage helper requires both speech/event outputs, valid development splits,
FP32 finite values, physical token counts and in-bounds offsets. A one-float replay
change is rejected. Full extraction reconstructs the audited plan from pinned
source files, verifies all waveforms against the completed CPU audit and checks
both fixed TRAIN smoke examples against the new banks.

The first physical audit ran with exact script
`C:/CodexArtifacts/pretrained-reuse-r2/native-input-execution-02/audit_audio.py`,
SHA256`7949bb57d30cc3e2d81c58b2957723e7738284fcce1e3c5e9919dabfc0f3d7a0`.
Its core imported source was committed at`6a075d0`. The repository auditor adds
an explicit fresh `--output` argument and formatting; it was executed at`bcb775b`.
Both complete `plan.json` and `decoded.jsonl` reproduce byte-for-byte. The original
audit receipt remains the immutable input to the audio cache configuration.

Only existing approved development train/validation payloads were decoded.
The source metadata and archive checksums include unused entries; reserved
calibration media/label payloads and sealed audit were not loaded. Existing
Teacher/QAT/corpora/weights and all previous outputs were preserved.

## Persisted arrays

| Bank | FP32 shape | Actual NPY bytes |
|---|---|---:|
| Language TRAIN states | 375078 x384 | 576,119,936 |
| Language validation states | 237067 x384 | 364,135,040 |
| Language query pairs | 22328 x384 | 34,295,936 |
| Audio TRAIN speech | 999234 x384 | 1,534,823,552 |
| Audio TRAIN event | 6035 x384 | 9,269,888 |
| Audio validation speech | 236130 x384 | 362,695,808 |
| Audio validation event | 1375 x384 | 2,112,128 |

Language NPY total974,550,912bytes; audio NPY total1,908,901,376bytes. Indexes,
receipts and logs are additional. The replays do not duplicate the large banks.
All arrays/media/model weights remain outside Git.

## Runtime and environment

| Process | Wall seconds | OS peak working-set bytes | Peak CUDA allocated bytes |
|---|---:|---:|---:|
| Language smoke | 12.2878 | 2,068,283,392 | 371,848,704 |
| Language complete cache | 538.4010 | 2,619,707,392 | 372,304,896 |
| Language complete replay | 529.3768 | 3,118,628,864 | 372,304,896 |
| Audio physical audit | 20.5474 | 111,792,128 | CPU only |
| Repository audit replay | 23.0080 | 110,559,232 | CPU only |
| Audio profile smoke | 5.4694 | 1,394,532,352 | 200,990,208 |
| Audio complete cache | 357.0046 | 3,481,264,128 | 200,990,208 |
| Audio complete replay | 375.3842 | 3,479,199,744 | 200,990,208 |

Sum of recorded process wall times1,861.4794seconds (31.0247minutes). CPU audits
and tests overlapped extraction, so this is neither elapsed end-to-end duration
nor GPU-kernel time. It excludes preparation, tests and final publication checks.
Cloud cost0; electricity/energy not measured. All eight processes terminated
normally; no background training was scheduled.

Local RTX3080Laptop16GiB, Python3.11, Torch/torchaudio2.6.0+cu124, CUDA12.4,
Transformers5.6.2, NumPy2.4.6, soundfile0.14.0, safetensors0.8.0. FP32, eager,
batch1, seed17, TF32off, existing packages only. Original dependency deprecation
warnings remain in logs. The tokenizer's length-inspection warning does not mean
an over512-token encoder window was forwarded.

## Reproduction and hashes

Language source:`6a075d08601fa1076d2ca9ebe51f6da74420de06`.
Audio extraction/replay source:`bcb775b8ebc0f76a3962cc3781f6de813c860908`.
Exact commands, times, versions, config hashes, original source pins and per-bank
hashes are retained in the linked JSON and corresponding local receipts.

| Completed receipt under `C:/CodexArtifacts/pretrained-reuse-r2/` | SHA256 |
|---|---|
| `native-language-cache-02/result.json` | `21d427849547541ff3824843b3c2e5e855a6a958fbc88e21ae9e037ffe405278` |
| `native-language-replay-01/result.json` | `6ff0d20d0776e53a17730509c49e3bf7b187245ec97e4361fcf19833e8b590da` |
| `native-audio-corpus-audit-01/result.json` | `5a289f858b0db764b03b93fe1a6dd6f9ac95ff1ba4df07b8715fccfc9af48784` |
| `native-audio-cache-01/result.json` | `02029f2935587f36b8a8deab1aa82e974b0e9d1582d074d2fede15f25ba87ae4` |
| `native-audio-cache-replay-01/result.json` | `dca50582dbc52f3ef32bbfd834da047c69f089297c5d8fad549048214a776b08` |

Commands use the existing isolated dependency paths documented in the preceding
input report. Language uses `scripts.capture_native_language --phase
smoke|cache|replay`; audio uses the existing `scripts.probe_native_audio_profile`
then `scripts.capture_native_audio --phase cache|replay`. Every non-smoke phase
requires its successful reference directory and exact `--reference-sha256`.
Use new output directories for any reproduction; these completed outputs are protected.

Verification:36new CPU tests pass, full local suite **2,007passed,1skipped**;
Ruff passes; all25 existing manifest validation/audit commands pass. Model/GPU
execution remains outside CPU CI. Receipts and exact verification script are in
`native-input-execution-02`; compact checked summary is in
`native-full-features-report-01`. Publication CI is checked on the pushed commit.

## What remains before the first common fit

1. Bind existing image/audio/video questions and complete option sets to this
   candidate's fixed language encoding. Language features above cover the six
   text-corpus files; they are not a complete query bank for every modality.
2. Freeze one task/index/sampler/coverage/budget/selector/retention contract,
   measure a bounded fit smoke, then train the889,345-parameter common readout
   with seed17 and all encoders frozen. Save best and final separately.
3. Evaluate the same candidate across all admitted tasks, including input-use
   controls. Report missing tasks, worst slices and calibration rather than
   transferring old scores from different heads.

Known scope gaps remain: STARSS TRAIN has no bell positives, validation has no
water-tap positives; two validation rooms provide limited independent evidence.
Current frozen CLEVRER rows contain512 TRAIN/400 validation questions, equally
split between static/temporal descriptive. They do not supply explanatory,
predictive or counterfactual labels to the next fit automatically. Image labels
here cover four attributes, not all relations. General speech, natural images,
Japanese, unseen wording, no-text task mode and genuinely complementary joint
decisions still require evidence. Joint video uses a different historical cache
profile and cannot silently substitute for the native8-frame candidate.

No Accuracy/NLL/Brier/ECE result or quality improvement was measured in this unit.
The196.25M candidate's common readout still has zero learning updates. Overall
R0–R8 and the product quality/packaging/final-audit gates remain open.
