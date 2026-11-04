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

## Live encoder follow-up

`token-ram-live-01` (source `e97daa8`) successfully regenerated both tensors
exactly, but its combined RAM metric counted only the Windows venv launcher,
not the descendant encoder process. Its 2.93 GB combined value is invalid for
encoder coexistence. The artifact and log are retained, not overwritten.

`token-ram-live-02` (source `3c6cc5e`) corrects recursive process accounting.
It again allocates/touches 456 slots, extracts **two** real training videos via
the pinned live CPU JAX/TF encoder, transfers their full FP32 grids and compares
them exactly to the previous saved arrays. The other 454 slots are synthetic.
Checkpoint/media/manifest/source revisions are checked before encoding; tracked
upstream source modifications are rejected and checkpoint hash is checked after.

| Live measurement | Result |
| --- | ---: |
| Parent + all descendant sampled peak RSS | 7,045,480,448 bytes |
| Sampling interval / samples | 50 ms / 193 |
| Encoder own peak working set | 4,159,995,904 bytes |
| Parent own peak working set | 2,939,097,088 bytes |
| First video decode/preprocess/compile/encode/send | 6.582459 s |
| Second video decode/preprocess/encode/send | 1.311054 s |
| Parent allocation + worker lifecycle + checks | 13.099995 s |
| Free RAM after child exit | 14,851,526,656 bytes |

These sampled simultaneous RSS measurements can miss sub-50 ms peaks and may
double-count shared pages; they are not unique physical memory or commit charge.
Separate per-process peaks need not occur simultaneously. The transport and live
encoder coexist successfully with the full-sized bank. No PyTorch readout/GPU
training is present yet. Encoder parameters remain 114,365,184, JAX backend CPU;
versions remain JAX/JAXlib 0.6.2, NumPy 1.26.4, PyAV 18.1.0, TensorFlow 2.18.1.
Both runs cost $0 and leave all original checkpoints/features untouched.

Reproduce with the previous command and `--live`, using a **new** output directory.
Results and worker environment/timing logs are retained under each output path.
The next experiment may now use live extraction into the bank; no disk-backed
full-token cache or precision downgrade is required.

## Interpretation and next gate

Full-size RAM allocation is practical on this machine. This does **not** establish
that 456 real scenes have been extracted, that the encoder-to-trainer live path
works, or that token-aware readout improves video decisions. RAM features are
ephemeral and must be re-extracted after process exit; no persistence/resume claim.
Live encoder coexistence has now passed as described above. Next predefine and
execute a token-aware readout comparison. Keep the same source/split,
frames and FP32 representations. The four-modality >=95% objective remains unmet.
