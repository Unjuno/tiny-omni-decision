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
| `...media-...-attempt-03` | RUNNING at 2026-11-15 02:22 JST | 238 WAVs / 209,034,254 bytes visible, 218 receipt rows. PID 47840 remains alive. `result.json` is not present, so this is not complete. |

At 02:26 JST, the same PID was still alive; the output contained 288 WAVs and
250 receipt rows. The receipt and progress timestamps remained 02:24 JST while
process CPU time had increased. This may indicate a long single-member transfer
or decode, so it is being observed rather than treated as terminal.

At 02:31 JST, PID 47840 remained alive and the 50-row checkpoint advanced to
350 receipt rows (357 WAV files were present while the next batch was in
progress). This confirms the previous apparent pause was the bounded receipt
flush interval, not a terminal state.

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

1. Wait for attempt-03 to terminate; never infer completion from a timeout.
2. Run the full receipt/media verifier and exact cross-split content checks.
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
