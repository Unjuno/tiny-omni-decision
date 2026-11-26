# RoboMNIST PositionGT clock-bridge audit protocol — 2026-11-02

## Purpose and boundaries

This is a one-file metadata/timestamp audit to test whether a synchronized
PositionGT stream can bridge the previously observed relative CSI timestamps
to the video presentation timeline for the already observed TRAIN-side
mechanics group `Ar1_Vhigh_Ac0_Rp01_UNC062`. It is not a new corpus split or
quality evaluation. This group remains permanently development/training-side;
it cannot be reused as validation, audit, or blind evidence.

The original seven-input smoke protocol remains frozen and its artifact remains
unchanged. This new protocol is a narrow explicit exception for the single
PositionGT member that the earlier protocol excluded. PositionGT is
supervision-only: inspect only its schema, timestamp field names, timestamp
units, lengths, and timing values necessary for clock comparison. Do not
inspect, serialize, derive features from, or report position/label values, and
never pass PositionGT to a model. Do not access any other repetition, archive
member, split, or sealed data.

## Pinned source and member

- RoboMNIST v1, Figshare article `28179383`, declared CC BY 4.0; attribution is
  retained. No external license acceptance or third-party message is involved.
- Source archive: `00_Ar1_Vhigh_Ac0.zip`, Figshare file `51588737`, size
  889,834,555 bytes, supplied and previously computed MD5
  `99a61bc10010855189daa45d0ad3b968`.
- Sole member:
  `00_Ar1_Vhigh_Ac0/Rp01_Ac0_Vhigh_Ar1_UNC062_Rxall_pos.json`.
- Previously audited central-directory entry: uncompressed 313,307 bytes,
  compressed 31,956 bytes, DEFLATE, CRC32 `082e8e4d`, offset 888,233,944.
- Revalidate source receipt, source archive metadata, directory-index hash,
  group membership, central-directory member, size/compression/CRC/local
  header before extraction. Do not trust the old receipt alone.

## Bounded execution

- Use HTTPS/TLS and exact HTTP 206 range responses with exact
  `Content-Range`/length verification; no retries or fallback source.
- Maximum transferred bytes: 1 MiB; body timeout: 120 seconds; maximum output
  1 MiB; retain at least 1 GiB free disk; CPU only; 2 GiB process peak.
- Create one new artifact directory. Never overwrite the original seven-input
  smoke or source audit products.
- Verify extracted file byte count, ZIP CRC32, and SHA-256. Record runner,
  protocol, parent receipt, source directory-index, source metadata, and
  previously saved CSI/video timing artifact hashes.
- Never load a model, fit a readout, compute decision metrics, or use a GPU.

## Clock test and decision rule

Compare only timestamp metadata from PositionGT with the existing CSI
`time_stamp_ns` sequence and all three existing video PTS/timebase records.
Report field paths, units, counts, finite/monotonic status, start/end values,
sampling cadence, and deterministic nearest-time residuals. Do not resample,
warp, estimate offsets from semantic position values, or train a synchronizer.

A common timebase is **supported for this one group only** if source metadata or
the PositionGT schema explicitly identifies the same acquisition clock/epoch
for the streams, with consistent units and timestamp alignment within one
native frame period across the full duration. Equal zero starts or similar
durations without an explicit shared clock are **insufficient**. A pass cannot
establish audio/WAV alignment, other repetitions' clocks, or session-level
independence. If no explicit bridge exists, retain `SYNC_UNVERIFIED` and stop
this source path before fitting.

## Frozen follow-up gate

Only if the clock bridge passes, prepare (but do not run until the source/task
split is frozen) a new video+CSI experiment on previously untouched TRAIN
repetitions, with all receivers and modalities from each physical repetition
in one group. The already observed group is excluded. Include video-only,
CSI-only, question/option-only, shuffled/misaligned, and missing-stream
controls; the target/options must be defined from source annotations before
labels are inspected. No results on this one-group audit can be called
generalization or product quality. Audio is excluded from the sync claim.
