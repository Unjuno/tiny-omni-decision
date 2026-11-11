# Path A Perception TRAIN runtime parity — 2026-11-11

## Result

`TRAIN_RUNTIME_PARITY_PASS`. This checks implementation fidelity between the
raw video/text JSON runtime and the frozen saved-feature/readout path. It is not
a task-quality evaluation and makes no accuracy or calibration claim.

The existing three-request runtime canary (attempt 06) had already established
JSON dispatch, option permutation behavior, and same-video caching. The missing
check was numerical agreement with the exact frozen features used to fit the
readout. This work adds that check without changing production runtime code,
weights, caches, or the existing attempt artifacts.

## Scope and safeguards

- Source baseline commit: `a922fda5616b7460dfe945db47a33abdc589a094`; the
  verifier ran from the isolated follow-up branch
  `codex/path-a-video-text-parity-20261111` with the new verifier file
  uncommitted. Its SHA-256 below binds that executed code.
- Verifier: `scripts/verify_path_a_perception_train_runtime_parity.py`, SHA-256
  `ccf895275d679e4c467584bf869fd3ed53cd2e659de4bac09d64aa0cf6296435` at
  measurement time. A later CPU-CI fix moved optional PyTorch/runtime imports
  into `run()` and reused the equivalent local SHA helper; the resulting source
  hash is `b9ec4a069ca926c4dbb0e97978c3d589e3b32f5c6a610137212bf872d6b2c02`.
  No parity algorithm, threshold, input, or output changed; the GPU run was not
  repeated for this import-only change.
- Frozen training corpus: official TRAIN JSONL only, SHA-256
  `2f2fa8ed4e81baaadcede50cc4343f0f0c55a02dd63703d3313e8d2188cb60a7`.
- The fit/selection group split was reconstructed with seed 17 and checked
  against the frozen config: 1,661 fit groups (hash
  `9a053ff7ae6d1c70b56df610f8b310dc81426b0af83c5cd1a325ec92af08d7ec`) and
  294 excluded internal-selection groups (hash
  `357115c13cce7e837beaec5abe635942945347572d08e2889a39444d2e8cf138`).
- A deterministic 8-video / 16-question subset was taken from fit groups. The
  prior runtime-canary video `video_10001` was explicitly excluded. IDs and
  their order are in the external result artifact.
- Runtime selection projects only IDs, source video IDs, question, options,
  and question type. Targets/labels were not used. No validation or sealed
  audit input was opened. No quality metric or checkpoint selection was run.
- Fixed absolute tolerances were set before execution: video features `1e-3`,
  text features `2e-4`, and logits `2e-3`.

## Execution and measurements

The pinned VideoMAE v2 Small encoder, MiniLM text encoder, and selected frozen
readout were loaded from their existing local artifacts. The runtime re-decoded
eight TRAIN videos, encoded the 16 projected questions/options, and compared:

1. Runtime VideoMAE final-block tokens against the existing per-video `.npy`
   feature shards.
2. Runtime MiniLM masked-mean candidate embeddings against the frozen TRAIN
   feature archive.
3. Readout logits and probabilities using saved features against logits and
   probabilities using raw-input runtime features.

| Check | Maximum absolute difference | Fixed tolerance |
| --- | ---: | ---: |
| Video tokens | 0.0 | 1e-3 |
| Text candidate embeddings | 4.47035e-7 | 2e-4 |
| Readout logits | 2.86103e-6 | 2e-3 |
| Option probabilities | 5.36442e-7 | informational |

- Device: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB.
- Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, Transformers 5.6.2,
  timm 1.0.15, PyAV 18.1.0.
- Elapsed time including model load and all parity calls: 28.72 seconds.
- Peak CUDA allocation: 327,003,136 bytes (about 312 MiB).
- GPU was idle after the verifier exited. No training was performed and no
  cloud compute or paid resources were used.

Exact per-sample deltas, ID order, input hashes, command line, environment, and
the result hash are in:
`D:/CodexArtifacts/pretrained-reuse-r2/perception-test-video-text-runtime-parity-20261111-attempt-01/result.json`

`result.json` SHA-256:
`c0fdd499edc633878eb95ca059975d3d728ee5bf4e53f0313faac02050dcf1b9`.

## Interpretation and limits

The run demonstrates that this pinned raw-input implementation reproduces the
frozen saved-feature path within the predeclared tolerances for this small,
fit-side TRAIN sample. It does not demonstrate useful video reasoning, text/video
complementarity, generalization, calibration, or product quality. The Perception
Test validation remains exposed and was not replayed. No changes were made to
Teacher v1/v2, existing readout weights, frozen corpora, or prior runtime
artifacts.

The only runtime warnings were upstream timm deprecation and duplicate model
registry warnings; the verifier completed and all numeric checks passed.

## Tests

- New CPU regression tests cover deterministic video-group partitioning,
  invalid split inputs, and label-field exclusion: 3 passed.
- Full repository CPU suite: 2,910 passed, 11 skipped in 185.76 seconds. The
  first attempt used Windows' default temporary directory on C: (0 bytes free)
  and ended with 6 failures / 336 setup errors; rerunning the complete suite
  with `TEMP`, `TMP`, and pytest's `--basetemp` on D: passed without excluding
  tests. The D:-backed run is the authoritative suite result.
- PR CI run 4154 initially failed during test collection because the new test
  imported optional PyTorch at module import. The runner now imports PyTorch
  only when executing the GPU verifier; this makes the CPU-only test module
  collectable without installing the ML extra. PR CI run 4156 passed for code
  commit `aaa47968c1db8bd63306d9320dd8b3e693a49ee4`.
- Ruff: `ruff check .` passed.
- CI manifest validation commands passed for the model manifest, all candidate
  dataset manifests, the four catalogs, and the dataset-manifest audit.
- GitHub Actions status is recorded after the push.
