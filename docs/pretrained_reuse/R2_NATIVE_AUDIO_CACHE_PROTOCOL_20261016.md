# Complete native speech/event cache: fixed execution contract

Continue the approved Path A input preparation; no optimizer, checkpoint selection,
quality scoring, calibration payload or final-audit access. Preserve all prior
corpora, native smokes, encoder weights and feature profiles. This is the same
audio frontend already exercised by `probe_native_audio_profile`, not a new
architecture or preprocessing comparison.

## Source scope and completed physical preflight

Use exactly the existing development train/validation Lombard and STARSS23 rows.
Lombard: 4,080/889 utterances, 36/9 speaker groups. STARSS23: 1,955/486 five-second
windows, 49/21 recordings, 6/2 rooms. Total7,410 examples and5,039 physical assets.
Keep every admitted row and its original order. Source files and rights receipts
are pinned by the independent `native-audio-corpus-audit-01/result.json`, SHA256
`5a289f858b0db764b03b93fe1a6dd6f9ac95ff1ba4df07b8715fccfc9af48784`.

The CPU audit actually decoded all7,410 examples in20.5474seconds, found7,410
unique rate+PCM identities, zero cross-split exact PCM overlap and zero all-zero
waveforms. It checked source/speaker/phrase/room/recording/raw-media identities,
nonoverlapping STARSS intervals, raw-file hashes and unchanged source inputs.
This exact-content check is not perceptual deduplication. STARSS room separation
does not establish global speaker separation. Metadata may include reserved IDs;
no reserved calibration waveforms/labels or final-audit payloads are loaded.

Keep known label/coverage limitations: STARSS has no bell-positive TRAIN windows
and no water-tap-positive validation windows. This extraction does not fix them.
No labels are used by the feature encoders; no unknown-license source is added.

## Frozen numerical profile

- Seed17, local CUDA, batch1, FP32, eager Whisper attention, TF32 disabled.
- Original frozen Whisper encoder8,208,384 parameters and original30s processor.
  Keep only physical `ceil(samples_at_16k/320)` token states, width384.
- Original frozen EfficientAT1,961,239 parameters and mel frontend; one384-vector.
- Lombard mono16k preserved. STARSS4-channel24k uses existing channel0 convention.
- CPU resample original24k to16k for Whisper; separately resample original input
  to32k on CUDA for EfficientAT. No24->16->32 chain, ASR, normalization or cropping.
- Use the pinned model revisions and original config in
  `configs/pretrained_reuse/native_audio_profile_v1.json` (SHA256
  `57fe149d4bccb09a4d28d3da0950efd4bb1cafdc15f8a95e78212c36ef9d39ae`).

## Storage and run gates

Reconstruct the audited row plan from pinned source files and require exact
equality before model load. Fixed storage in separate train/validation banks:

| Bank | FP32 shape |
|---|---|
| TRAIN speech | 999234 x384 |
| TRAIN event | 6035 x384 |
| Validation speech | 236130 x384 |
| Validation event | 1375 x384 |

Total tensor bytes1,908,900,864 plus file headers and small receipts. Plan offsets
map every source ID to both representations. Record source order and per-example
PCM/speech/event hashes. Hash source/model files before/after and frozen states
before/after. Preflight complete allocation plus2GiB disk reserve; maintain that
reserve throughout. No silent row removal, altered precision or partial-cache
admission. Each process:3600seconds maximum,3GiB peak CUDA allocation,6GiB host
OS peak working set. Stop incomplete on integrity, numerical or resource failure.

1. On clean committed source run fresh `native-audio-profile-smoke-02` using the
   existing two fixed TRAIN inputs. It retains historical native-array parity.
2. Run `capture_native_audio --phase cache` to new `native-audio-cache-01`, passing
   that successful profile receipt and its hash. Require identical model/source
   pins, versions and frozen states. Recheck both fixed examples within the bank.
3. Run `--phase replay` in a new process/output `native-audio-cache-replay-01`,
   passing the complete cache receipt/hash. Re-decode all7,410 audio windows,
   recompute both representations and compare every saved float exactly. Require
   identical per-example digest ledger. Do not duplicate the large output banks.

Never retry into an existing output directory or relax a failure gate. The exact
source commit, command, config hash, corpus/model pins, completed counts, duration,
resource usage and error (if any) belong in every result. No environment upgrade.

## After acceptance

This closes the full audio feature preparation unit only. It does not show
speech comprehension, environmental accuracy, multimodal benefit or calibration.
Combine accepted language/image/video/audio banks under one task/sampler/selection
contract; encode other modalities' question/candidate strings with the same fixed
language profile before the first shared-readout fit. Preserve missing temporal,
joint, natural-input and unknown-wording coverage as explicit quality gaps.
