# ADL partial archive inventory protocol

Purpose: check whether the predeclared participant split has per-activity
support across the complete 25-activity source without storing 4.6 GB of
archives. This is **source feasibility only**, not a training admission or
validation score.

- Pin version-1 source-record SHA-256
  `c01bf60460981280c5063ea54c038d41c751bf3056220059c3896cbc5d2b97c6`
  and the existing participant split protocol SHA-256
  `9c301d4a3acc37711d25943e7165533410bb3297630eef19c7f6431d2daaf5f8`.
- Enumerate **all 25** numbered ZIPs, in ascending published archive size and
  then name order. Do not select or remove categories based on observed results.
- Reuse the hash-pinned aggregate receipts for the first two fully downloaded
  archives and the already CRC-checked third archive. For each remaining ZIP,
  fetch the bounded 65,536-byte tail and only its `multisensor-1` HDF5 member.
  Exact HTTPS Content-Range and length, local/central headers, uncompressed
  limits and ZIP member CRC are mandatory. Hash every inspected member.
- Do not write the downloaded member, audio or vibration bytes to disk. Read
  only participant, trial, timestamp and sample-rate columns from the HDF5.
  Save per-activity aggregate counts and hashes; do not save participant IDs.
- Process sequentially in a new output directory. Require at least 2 GiB of
  available host RAM and 1 GiB free disk before each source; compressed members
  above 64 MiB or decoded members above 128 MiB are explicitly marked
  unsupported, never truncated or omitted from the final inventory. Per-source
  successes and failures are saved; network errors stop the run and retain
  partial results. No automatic retries or restarts are implied.
- Apply the unchanged seed-17 participant hash. Keep the special `999` person
  unassigned. Report missing train/validation class support and all unassigned
  samples. Five-sensor alignment is **not established** by the one-sensor
  scan. Never infer physical timestamp synchronization from row IDs.
- The publisher provides no ETag for the tested range endpoint. Partial
  members can be checked against the ZIP CRC and locally hashed, but the whole
  publisher MD5 is not verified. Treat them as weaker exploratory source
  evidence, not sufficient source authentication for training or promotion.
- Do not fit a model, tune splits, access a sealed audit, or infer broad quality.
  Four-modality and independent-group quality gates remain open.

This protocol follows the first two full-MD5 activity checks and the third
partial-member pilot. The dataset is CC BY 4.0 but contains only 14 participants
in one environment; its results alone cannot establish generalization.
