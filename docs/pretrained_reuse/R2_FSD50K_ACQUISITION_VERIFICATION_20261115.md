# FSD50K annotation-Decision acquisition and verification — 2026-11-15

## Scope

This is a narrow Path A audio task: predict whether the frozen FSD50K dev
annotation assigns one of four supplied broad labels. A negative means “not
annotated in this release,” not verified acoustic absence. Only the frozen,
uploader-grouped dev manifest is eligible. The official eval labels and audio
remain unopened and unused.

## Frozen inputs

- Dataset revision: `10.5281/zenodo.4060432`.
- Preflight: `C:/CodexArtifacts/pretrained-reuse-r2/fsd50k-annotation-presence-v1-preflight-20261115-attempt-01/`.
- Preflight manifest SHA-256: `0bbe5db5e6167a4c041e349889920f8da0f1a0703ee00fa63ff12a59a0531b6d`.
- Split counts: train 12,845 / calibration 2,053 / validation 3,981.
- Uploader counts: train 2,089 / calibration 290 / validation 572.
- All 129 previously selected IDs and their 128 uploader groups are excluded.
- The archive central-directory SHA-256 is
  `a52c8e77acf8c01e5108a3872461a65f2a583b2ddc3e0bb6554e51dc7ae12c5f`.
- The frozen sample IDs resolve to 18,879 WAV members, 12,352,131,640 expanded
  bytes, and a planned maximum range transfer of 8,867,277,632 bytes. C: had
  over 279 GB free when acquisition began; the downloader preserves a 2 GiB
  reserve.

## Source-duration discrepancy

The FSD50K paper describes all clips as 0.3–30 seconds, uncompressed PCM
16-bit, 44.1 kHz mono. ZIP central-directory sizes and seven bounded WAV-header
reads show that the frozen cohort contains **39** files over 30 seconds by
that declared encoding: 20 train and 19 validation, from 17 uploader groups;
none are in calibration. The largest inspected header corresponds to about
550.25 seconds. These are official dev members and have allowed per-clip CC BY
3.0 or CC0 licenses. They are retained in the frozen manifest; this stage
does not truncate or silently remove them. Whole-clip feature extraction must
process them safely or fail closed before fitting. Full decoded-audio duration
validation remains part of the media verification.

## Acquisition attempts

| Attempt | Result | Evidence |
|---|---|---|
| `...media-...-attempt-01` | FAIL before saving any WAV | Windows Python lacks `os.statvfs`; the archive index also revealed a selected member larger than the original 5 MiB guard. Only the directory index was read. |
| `...media-...-attempt-02` | FAIL before saving any WAV | ZIP local filename length and extra-field length were read in reverse order. The new output contains an empty receipt and the failure record. |
| `...media-...-attempt-03` | FAIL at 2026-11-15 02:38 JST | 445 WAVs / receipt rows; terminal `result.json` records `ValueError('range length mismatch')`. Official eval access flags remain false. |
| `...media-...-attempt-04` | RUNNING at 2026-11-15 02:59 JST | Fresh output; 445 prior files copied only after SHA/CRC/manifest verification. Receipt checkpoint 450; 472 WAVs visible, so 22 files are beyond the last durable receipt. |

The old reader raised a generic `range length mismatch` after its retry limit;
it did not preserve the failing response body length or member ID. A read-only
probe of the next member after the last durable receipt (`119193.wav`, volume 2,
offset 700,301,253, compressed size 337,333 bytes) succeeded on a fresh request.
This supports a transient or inconsistent range response as the likely cause,
but the exact failed response remains UNKNOWN. The failed attempt was not
restarted in place. All 445 committed receipt rows were validated against
manifest identity, WAV bytes, SHA-256 and ZIP CRC, and the exact partial WAV set
matched.

The new reader retries short bodies from the same offset up to six total
attempts, uses `Accept-Encoding: identity`, validates `Content-Range`, and
charges each retry against the frozen total budget. The resume path accepts
only the identified attempt-03 failure, revalidates every partial row, copies
into a distinct output directory, confirms the same central-directory hash,
and resumes after the exact verified receipt prefix. Unit tests exercise
short-body retry and reject corrupted partial media.

Attempt-04 was observed alive at 02:59 JST after copying the 445 verified WAVs
to its new output and fetching the pinned archive index. Its durable progress
checkpoint reached 450 rows; 472 WAV files were visible during the next batch.
It remains incomplete until a terminal PASS result and the full verifier.

## Pinned Path A audio encoder recheck

The selected EfficientAT DyMN-0.4 checkpoint is present at
`C:/CodexArtifacts/pretrained-reuse-r1/efficientat-audit-v1/dymn04_as.pt`;
its size is 8,086,973 bytes and its locally recomputed SHA-256 is
`49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`. The
upstream source checkout is clean at commit
`a425fdce92572e602a1d5634799bd9f1f2efa806`. The existing R1 release-license
follow-up records explicit upstream commercial checkpoint permission and
admits this as an ALLOW research candidate; task quality remains open. The
reusable extractor pins the same source and checkpoint, requires the measured
PyTorch/torchaudio `2.6.0+cu124` pair, and records 1,961,239 model parameters.
The active Python runtime was rechecked as Python 3.11.9, PyTorch
`2.6.0+cu124`, torchaudio `2.6.0+cu124`, and CUDA runtime 12.4. CUDA was
reported available; no checkpoint forward was run in this audit. These checks
establish artifact identity and a licensed candidate, not an FSD runtime
forward or audio-task accuracy result.

## Frozen-feature Decision implementation

`scripts/train_fsd50k_annotation_decision.py` now implements the predeclared
384-D frozen feature route using four typed binary questions and the shared
`NativeFeatureDecisionScorer(384, 6, 64)` (25,089 readout parameters; 1,986,328
encoder-plus-readout parameters). It fits on train only for a fixed 20 epochs,
fits one scalar temperature on calibration only, selects the epoch by
calibrated validation macro NLL, records per-label Accuracy/NLL/Brier/ECE and
train-validation metrics, and saves separate best and final checkpoints plus
selected validation predictions. The loader verifies the frozen manifest,
complete acquisition report, feature hashes/order, encoder revision/weight
hash, temporal policy, and disjoint source/group identities. Five focused CPU
tests, including a 20-epoch synthetic save/reload smoke, pass. This is pipeline
plumbing only; no real-media feature extraction, FSD fit, or quality metric has
been performed yet.

Attempt-03 uses a distinct output path and a 64 MiB per-member safety cap, above
the observed 48,532,044-byte maximum. Its range-transfer budget remains bounded
by the planned 8,867,277,632 bytes. The two implementation defects were fixed
before attempt-03 and covered by tests. Focused tests currently pass: 13; Ruff
passes. The live acquisition script was not changed after launch.

## Verification gate before feature extraction

[`validate_fsd50k_annotation_acquisition.py`](../../scripts/validate_fsd50k_annotation_acquisition.py)
checks the passing terminal receipt against the exact frozen manifest, verifies
all sample IDs, split/uploader/source/license fields, file sizes, SHA-256 and
ZIP CRCs, requires the exact WAV file set, rejects cross-split exact byte
duplicates, and requires official eval access flags to remain false. Four
fixture tests cover success, corrupted media, exact cross-split duplicates,
and eval/unlisted-file rejection. This verifier must pass before any
features are generated.

The acquisition process does not use the GPU. At the latest live snapshot the
RTX 3080 showed 54% utilization and 8,270 MiB allocated by another workload;
that process was left untouched. No FSD50K features, model predictions,
training, calibration, validation scoring, or official eval access has yet
occurred.

## Next gates

1. Continue monitoring attempt-04's live PID/session; never infer completion
   from an observation timeout.
2. On terminal PASS, run the full receipt/media verifier and exact cross-split
   content checks.
3. Extract the pinned EfficientAT features over the entire clip, including all
   39 duration outliers; abort on decoder, memory, non-finite, or repeatability
   failures.
4. Fit the predeclared 25,089-parameter typed readout on train only, fit one
   scalar temperature on the uploader-held calibration split, and select by
   macro validation NLL. Preserve seed, learning history, predictions, and
   per-label Accuracy/NLL/Brier/ECE.

This remains one audio annotation-prediction component. It does not close
image, text, video, multimodal necessity, package-size, or release-quality
gates.
