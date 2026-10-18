# R2 SAM2 cached decoder batch — actual measurement, 2026-10-14

## Result and decision

**Execution/reproducibility PASS; drop-in prefix admission FAIL.**
The fixed64-object cached forward is faster with batched native low-resolution
heads: median0.931569s→0.184805s, approximately5.04× across four matched windows
per arm from two processes. All64 binary masks, boxes, foreground seeds/nonempty
flags and prefix NMS identities match. However, presence-score maximum absolute
error **1.335144e-5 exceeds the predefined1e-5 guard**. Keep the failed guard;
do not substitute this path into the protected corpus or relax tolerance.

This is one known train image's first64-point engineering comparison, not full
1024-grid, multiple-time or task-quality admission. It does not establish5×
end-to-end acceleration, better semantic binding, Decision Accuracy, NLL/Brier/ECE
or95–98% Omni quality. Numerical nonadmission is not evidence of task-quality
failure or a pretrained-model capacity ceiling.

## Fixed implementation and actual comparison

[Pre-execution protocol](R2_SAM2_BATCH_DECODER_PROTOCOL_20261014.md).
Code commit `2efbd752718685b024994d71e94fc8e66d00934e`;
`scripts/probe_sam2_batched_decoder.py` and
`src/tiny_omni_decision/sam2_batch_masks.py`.

One unchanged pinned native `Sam2VideoModel`,38,962,498 registered parameters;
FP32/sdpa/TF32off/seed17/deterministic/CPU4. Known train10903, native frame0 RGB,
same1024-grid's first64 points and the native processor's exact scaled coordinates.
The64 points are the grid prefix with an existing saved floating reference,
not a random representative or an annotation-selected subset.

The new path reuses the same native cached image features and initial no-memory
embedding. It runs native prompt/mask heads as one image with64 independent
positive-point prompts, gathers each object's best-IoU mask and keeps native
score<=0 → mask=-1024 behavior. No candidate, score threshold or NMS rule is
changed. Only unused1024-resolution interpolation, object-pointer projection and
temporal-memory output are bypassed. This is not a propagation/session API.

Native first and both measured native calls match the archived64 raw masks/scores
byte-for-byte. Both paths are warmed; call order alternates native→batch then
batch→native. Measured windows have the same64 objects and zero additional image
encoder calls. Setup/preprocessing and output-to-CPU conversion are excluded from
these windows; body time is retained separately.

| Process | Rep | Cached native64 seconds | Batched64 seconds |
|---|---:|---:|---:|
| Primary |0|0.932019|0.184214|
| Primary |1|0.931120|0.185395|
| Fresh replay |0|0.925464|0.183734|
| Fresh replay |1|0.936458|0.185668|

Four windows per arm, not a production latency distribution. Body seconds:
primary9.466856, fresh9.868461, independent CPU audit3.315309; total22.650626.
GPU-active time, energy, temperature and isolated end-to-end latency were not
measured. Cloud cost0. Both GPU processes and the CPU audit returned terminal
exit0. GPU was idle afterward; unrelated processes were preserved.

## Numerical/discrete result — all guards retained

| Check | Actual | Predetermined requirement |
|---|---:|---:|
| Raw-mask max absolute difference |0.000179290771484375|≤0.001 — pass|
| Presence-score max absolute difference |0.0000133514404296875|≤0.00001 — **fail**|
| Relative tolerance |0|0|
| Changed threshold0 mask pixels |0/4,194,304|0 — pass|
| Changed proposal masks |0/64|0 — pass|
| Boxes / foreground seeds / flags / prefix NMS |Exact|Exact — pass|
| Raw/score floating byte agreement with native |Both false|Reported separately|

One of64 presence scores exceeds the guard. Native minimum score11.794122 and
the score>0 flags are unchanged on this prefix. Those facts do not permit
loosening the guard or extrapolating to other images/prompt sets, where scores
or mask boundaries could be closer to decisions. The candidate's warm/two
measured calls and all8 saved arrays also match the independent fresh process
shape/dtype/bytes exactly. Numerical differences are reproducible, not a missing
output or random-seed improvement.

All model parameters remain frozen with gradientsNone; persistent model-state
SHA-256 before/after and fresh:
`658e3aaff3d08b6cc8337a1f125e5ee55451ed3bfec38f9def601a14bb6912d8`.
CPU/CUDA RNG, source embeddings/prompts and native preprocessed pixels are
unchanged. One image encoder call total per process, none in measured windows.
Global peak Torch allocation3,568,890,880 bytes (approximately3.32GiB), reserved
5,064,622,080 bytes (approximately4.72GiB). This includes warmups/cached paths;
no per-arm peak advantage is claimed.12GiB free-GPU and8GiB+128MiB disk admission
passed;≤8GiB allocation/≤180s body checks passed after calls, not via watchdogs.

Existing RTX3080 Laptop16GiB, Torch2.6.0+cu124 and Transformers5.6.2; unchanged
local pinned checkpoint revision `de431c4043854a71d8101e17995dfe596bf101a5`,
weight SHA-256 `48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7`.
No new weights, training, source annotations, dev/sealed data, downloads,
environment upgrades, quantization, product promotion or main merge.

## Verification and artifacts

Independent CPU audit rechecks all8 fresh arrays, both native archived matches,
every64 mask/presence flag, maximum errors, unchanged source/config hashes,
gate outcome and matched-window count/order/medians. It loads no model or GPU.
Native32-time discovery and all legacy artifacts remain unchanged.

22 CPU tests first failed for absent helper APIs, then passed: hand-checked
per-object gather/presence/ties/order/schema cases plus actual installed tiny
native heads versus serial low-resolution output. The tiny fixture forces
positive presence so an all-absent comparison cannot pass vacuously. It has no
pretrained weights/backbone/GPU. Optional Torch/Transformers tests skip when
those libraries are absent from ordinary CI. Read-only review found no Critical
or Important implementation issue before GPU execution. Local Ruff passed;
full CPU suite **1,193 passed /1 skipped,16.00s**.

Small [publication JSON](R2_SAM2_BATCH_DECODER_RESULTS_20261014.json) contains all
three reports, source/result hashes, windows, sizes and admission outcome.
Publication SHA-256:
`8700ff83f39ed370931210044bc70b143d174670b1e7497549be361ac353a247`.
Primary config SHA-256:
`e214a6ce2aa91a48c4951a77466ebdfa3c790236d46a2d61f2436a771dc23660`.
Primary result SHA-256:
`5122a01547fc8b1b2733b8582485ea2893b25b86511a75360cee0a461962856e`.
Fresh result SHA-256:
`08f73c65d014f6e4d00851ae82f814b21a0da2b2bdede870acac47f619d2488d`.
Both58,023,638-byte output archives have SHA-256
`d8357e6ee0fd21a27e05daed05eeb64817c5eb1736a4de3dbb344aa58139d828`.

Local directories under `C:/CodexArtifacts/pretrained-reuse-r2/`:

- `sam2-batched-decoder-prefix-01/`;
- `sam2-batched-decoder-fresh-replay-01/`;
- `sam2-batched-decoder-independent-audit-01.json`;
- preserved `audit-sam2-batch-prefix-01.py` / publisher scripts.

Audit script SHA-256:
`1fe19f5e24654a050faa15779a5ca036d4c310123c51a1327ae4d20125b12343`.
Weights/media/NPZ stay outside Git. Fresh paths are mandatory for reproduction:

```powershell
python -m scripts.probe_sam2_batched_decoder --output C:/CodexArtifacts/pretrained-reuse-r2/sam2-batched-decoder-prefix-01
python -m scripts.probe_sam2_batched_decoder --reference C:/CodexArtifacts/pretrained-reuse-r2/sam2-batched-decoder-prefix-01 --output C:/CodexArtifacts/pretrained-reuse-r2/sam2-batched-decoder-fresh-replay-01
python C:/CodexArtifacts/pretrained-reuse-r2/audit-sam2-batch-prefix-01.py
```

## Next gate

The wider semantic cue-to-cover/identity-memory corpus still needs admission.
An explicit fixed chunk-size comparison or separately derived-corpus protocol
can investigate this numerical limitation while keeping the same guards and
preserving every attempted result. Do not silently change batch size/tolerance
and call it the same experiment, proceed to full corpus from this nonadmitted
prefix, or repeat nearest-distance semantic tuning. Full native Omni quality,
total resident package and95–98% task/OOD/joint performance remain open.
