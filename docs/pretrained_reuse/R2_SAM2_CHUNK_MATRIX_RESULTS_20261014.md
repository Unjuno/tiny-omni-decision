# R2 SAM2 fixed chunk matrix — actual results, 2026-10-14

## Result and decision

**Execution and reproducibility PASS; all three prefix admissions FAIL.**
Fixed chunk sizes16/32/64 process the same known train10903/frame0/first64
positive-point grid prompts. All binary masks, normalized boxes, foreground
seeds/nonempty flags and prefix NMS identities match the unchanged native
reference. Every arm exceeds the **unchanged1e-5 presence-score guard**.
The raw-mask1e-3 guard passes. Do not loosen tolerances or substitute this path
into protected corpora. The predefined drop-in batch-width search ends here;
do not keep trying smaller widths until one happens to pass.

Same64-work cached native median 0.938912s,
four measured windows per arm across two processes:

| Chunk width | Cached forward median s | Native/arm ratio | Raw maxabs | Score maxabs | Prefix admission |
|---|---:|---:|---:|---:|---|
| 16 | 0.201659 | 4.656× | 0.000114440917969 | 1.52587890625e-05 | FAIL |
| 32 | 0.208713 | 4.499× | 6.86645507812e-05 | 1.43051147461e-05 | FAIL |
| 64 | 0.187322 | 5.012× | 0.000179290771484 | 1.33514404297e-05 | FAIL |

Each arm has one of64 scores over the guard and zero changed pixels across
4,194,304 mask values. Floating raw/score bytes differ from native in all arms.
Four windows and one grid prefix are not a production latency distribution,
full1024-grid/other-time admission, quality comparison or end-to-end speedup.
No claim about Decision Accuracy/NLL/Brier/ECE or95–98% Omni quality follows.

## Fixed implementation and actual work

[Pre-execution protocol](R2_SAM2_CHUNK_MATRIX_PROTOCOL_20261014.md).
Source commit `0d8a8d2acc5a56e7d06b08d8ade200a719624ea7`.
New `sam2_chunked_masks.py` and separate `probe_sam2_chunked_decoder.py`;
protected previous runners/helpers/model/config/data/artifacts were unchanged.
One frozen native SAM2,38,962,498 parameters, FP32/sdpa/TF32off/seed17,
deterministic inference/CPU4. Native preprocessing, scaled prompts, cached
features with initial no-memory addition and low-resolution mask semantics
are identical. Each chunk executes unchanged native prompt/mask heads.
This remains independent-frame proposals, not a video propagation API.

Hooks confirm actual decoder object counts: native64 calls×1, chunk16 four
calls×16, chunk32 two×32, chunk64 one×64. Every window contains exactly64
objects and zero new image encoder calls; one image encode total per process.
Both paths are warmed. Fixed forward/reversed arm order, synchronization
around the call, setup/preprocessing/output D2H outside measured windows.

| Process | Rep | Arm | Forward seconds | Decoder calls |
|---|---:|---|---:|---:|
| primary | 0 | native | 0.930937 | 64 |
| primary | 0 | chunk16 | 0.202640 | 4 |
| primary | 0 | chunk32 | 0.207638 | 2 |
| primary | 0 | chunk64 | 0.186037 | 1 |
| primary | 1 | chunk64 | 0.187887 | 1 |
| primary | 1 | chunk32 | 0.209787 | 2 |
| primary | 1 | chunk16 | 0.202926 | 4 |
| primary | 1 | native | 0.932359 | 64 |
| fresh_replay | 0 | native | 0.945464 | 64 |
| fresh_replay | 0 | chunk16 | 0.199313 | 4 |
| fresh_replay | 0 | chunk32 | 0.207129 | 2 |
| fresh_replay | 0 | chunk64 | 0.186757 | 1 |
| fresh_replay | 1 | chunk64 | 0.190881 | 1 |
| fresh_replay | 1 | chunk32 | 0.210615 | 2 |
| fresh_replay | 1 | chunk16 | 0.200678 | 4 |
| fresh_replay | 1 | native | 0.992558 | 64 |

Primary body 14.514523s, fresh body
15.665677s, independent CPU audit
7.126282s; total 37.306481s.
Bodies include setup/warmups/copies/array save, after Python/module imports.
Isolated GPU-active time/energy/temperature/end-to-end product latency were
not measured. Cloud cost0. Both GPU runs and CPU audit returned terminal exit0.

## Integrity and verification

All16 raw/score arrays agree shape/dtype/bytes in an independent fresh process.
Every candidate warm/repeated call agrees exactly; both measured native calls
agree with the original raw archive. The64 arm byte-exactly reproduces the
previous failed64 experiment, including its numerical differences.
Frozen state SHA-256 before/after/fresh:
`658e3aaff3d08b6cc8337a1f125e5ee55451ed3bfec38f9def601a14bb6912d8`.
All parameter flags remain frozen/gradientsNone; CPU/CUDA RNG, embeddings,
prompts and preprocessed RGB unchanged. Source/config hashes rechecked.

Code review found an inherited-source admission gap before GPU execution:
updating a protected helper path with its current hash could silently re-pin it.
The new runner now verifies inherited hashes first; additions reject any
conflicting existing pin, including runtime class files. Regression tests
failed for the absent guard API then passed; no protected old file was edited.
Read-only re-review found no remaining Critical or Important issues.

28 new CPU tests cover actual tiny native heads, tail/order/permutation/call
counts, complete-input rejection, freeze/input/state/RNG and protected source
pins. Optional model libraries skip native-head tests in normal CI without
ML dependencies; no pretrained weights/GPU loaded by CPU tests.
Local full suite **1,221 passed /1 skipped,17.14s**; Ruff and model/dataset
manifest, four catalog and policy validations passed.

Independent CPU audit opens no model/GPU/annotations. It compares all16
arrays, both archived native matches, preceding64 outputs, mask/presence
decisions, errors, count/order/windows/medians, source/config hashes and
guard classifications. It independently regenerates normalized boxes/NMS;
foreground seeds are checked by the runner, not independently regenerated.

Global Torch peak allocation 3,568,890,880 bytes (~3.32GiB),
reserved 6,603,931,648 bytes (~6.15GiB) in both processes.
Global peaks include warmups; no per-arm memory advantage is claimed.
GPU free≥12GiB/disk≥8GiB+256MiB admission passed; allocation≤8GiB/body≤180s
checks passed after calls, not watchdogs. Existing RTX3080 Laptop16GiB,
Torch2.6.0+cu124/Transformers5.6.2; no environment upgrades/downloads.
Checkpoint `de431c4043854a71d8101e17995dfe596bf101a5`, SHA-256
`48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7`.

## Artifacts and next action

Small [publication JSON](R2_SAM2_CHUNK_MATRIX_RESULTS_20261014.json) SHA-256
`63f29425675e8a5caadbfdadb890cf5c53c31a1d4a2b5bc8801f180dc2044451` includes all reports/source hashes/windows/counts/guards.
Primary config SHA-256 `863cb6eb3a1da7cfc9a0f191439074bf66a5eb88ae155c8cb2eb461c68bf4eb2`.
Primary/fresh result SHA-256:
`cae6796180950241ecc8c59a79986795332c7aa18813d05a36c454f3dd1c29b2` /
`0bb7030135f9278e3b0248a8eed39392b73ce9c252ac8a643e7a7c17bd0f1582`.
Both 116,044,542-byte array archives:
`27d0e1addb2085ec52d0a71b4c6711acb41f67463b1174a89b32f5722caa36f6`.
Audit script SHA-256 `f8e717bf167af91e73766e5ca1768607f8e0c2c8f798b2f4aa2683d49afb194c`.

Preserved local paths under `C:/CodexArtifacts/pretrained-reuse-r2/`:

- `sam2-chunk-matrix-prefix-01/`;
- `sam2-chunk-matrix-fresh-replay-01/`;
- `sam2-chunk-matrix-independent-audit-01.json`;
- `audit-sam2-chunk-matrix-01.py` and publisher script.

Weights/media/NPZ stay outside Git. No train/validation corpus or checkpoint
changed, labels/dev/sealed audit opened, training/quantization/product
promotion/main merge performed. Numerical nonadmission is not a task-quality
failure or proof of a model capacity ceiling. Wider semantic cue-to-cover and
identity-memory data must use native reference or a separately fixed
derived-input protocol; never silently import this nonadmitted path. No more
batch-width search or nearest-distance semantic tuning. Useful text/image/
audio/video/joint/OOD decisions and full lightweight package remain open.
