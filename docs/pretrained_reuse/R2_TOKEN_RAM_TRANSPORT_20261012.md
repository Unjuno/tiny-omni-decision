# FP32 video token RAM transport audit

Status: **PASS_TRANSPORT_CAPACITY_ONLY**, not a quality result.

The preceding token audit showed a 2,868,903,936-byte raw bank for 456 scenes,
larger than available disk. This audit tests RAM capacity and separate-process
transport without reducing precision or duplicating features for each question.

Source commit: `1bc4eed17b3accc37111d1c5f442e5c248c732d6`.
Output: `C:/CodexArtifacts/pretrained-reuse-r2/token-ram-transport-01/`.
Exact arguments, input hashes, script hash and start time are in `config.json`;
measurements are in `result.json`; worker stderr is retained separately.

## Executed

- Allocate and touch all 456 FP32 `[8,16,16,768]` slots in the normal Python process.
- Replay the two previously audited training-video tensors from the existing
  JAX environment's Python through a binary pipe. No model is loaded by this worker.
- Bound headers to 4096 bytes, enforce expected shape/dtype, verify payload SHA-256
  and finite values. Reject truncated or corrupt tensors.
- Compare both received arrays exactly with pinned originals. Repeated cache
  lookup returns the same read-only view into the bank, with no per-question copy.
- Preserve all previous artifacts. No training, GPU computation, data download,
  validation quality evaluation or sealed audit access occurs.

## Measurements

| Measurement | Result |
| --- | ---: |
| Allocated/touched slots | 456 |
| Real saved video assets loaded | 2 |
| Synthetic zero slots | 454 |
| FP32 bank bytes | 2,868,903,936 |
| Parent peak working set bytes | 2,938,957,824 |
| Available RAM before allocation | 17,549,271,040 |
| Available RAM after worker exit | 14,651,043,840 |
| Allocation + replay + checks wall time | 1.069923 s |
| Local compute cost | $0 |

Started `2026-10-11T20:50:51.327472+00:00`, finished
`2026-10-11T20:50:52.397155+00:00` (October 12 JST).
This timing excludes imports and initial input-file hash checks.
Parent peak excludes child memory; the live encoder was not present. The earlier
encoder audit measured 4.16 GB peak separately. Concurrent encoder/readout peak
must still be measured rather than inferred by adding separate measurements.

## Interpretation and next gate

Full-size RAM allocation is practical on this machine. This does **not** establish
that 456 real scenes have been extracted, that the encoder-to-trainer live path
works, or that token-aware readout improves video decisions. RAM features are
ephemeral and must be re-extracted after process exit; no persistence/resume claim.
Next connect a pinned live encoder to this bounded transport and measure coexistence
before committing to a token-aware readout comparison. Keep the same source/split,
frames and FP32 representations. The four-modality >=95% objective remains unmet.
