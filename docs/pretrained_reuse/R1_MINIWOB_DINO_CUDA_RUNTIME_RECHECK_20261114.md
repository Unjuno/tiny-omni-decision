# MiniWoB DINO CUDA runtime recheck (2026-11-14)

## Result

`PASS_CUDA_RUNTIME_PARITY` for one previously trained TRAIN example. This is
an inference/runtime smoke only, not a new quality estimate. The 512-row
validation set, any sealed audit, and all training paths were left untouched.

The pinned DINOv2-small encoder and selected MiniWoB readout were loaded
offline on the local NVIDIA GeForce RTX 3080 Laptop GPU. For
`train-seed-940001`, the GPU runtime selected the same opaque option ID as the
saved CPU replay. Maximum absolute probability difference was `2.756715e-7`;
probabilities summed to `1.000000102`. The runtime's JSON schema was valid and
the encoder remained frozen.

| Measurement | Result |
|---|---:|
| Device | NVIDIA GeForce RTX 3080 Laptop GPU |
| PyTorch / CUDA runtime | 2.6.0+cu124 / 12.4 |
| Local load time | 7.254 s |
| Single-example inference time | 0.262 s |
| Peak allocated CUDA memory | 104,218,112 bytes |
| Selected option | `action-2` (same as saved CPU replay) |
| Maximum probability difference | `2.756715e-7` |

The timings include a cold load and one example only; they are not a latency
distribution or a product-device SLA. GPU use was inference-only. No training,
checkpoint write, model download, validation scoring, cloud compute, or
intervention in the other GPU process occurred.

## Reproduction and pins

- Repository source commit: `f8fd0e50da92ff79a00a10c7cd7c343194ef99bf`
- Encoder: `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`
- Encoder weight SHA-256: `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`
- Readout SHA-256: `983366413dc2afd9f21a5411dd8793525f68342c44936d3bb2ccffe123f5c3fb`
- TRAIN screenshot SHA-256:
  `593f4c64b61996c5668f062a94bf1471d4c7313df638a61963c3171a61fb3df3`
- Effective experiment config and corpus hashes are unchanged from
  `R2_MINIWOB_DINO_PATH_A_RESULTS_20261112.md`.
- Output directory:
  `D:\CodexArtifacts\pretrained-reuse-r2\miniwob-dino-cuda-runtime-smoke-20261114-attempt-03`
- Result JSON SHA-256: `D0DE44E8C632C6594E4C49E11946A87DF8B5CCECB91700998C61D1AA19F2FB61`
- Runner SHA-256: `1F17C28FA40616EC4577519FAC6AABDA8930328576AF46A34E98BA9C32D73F82`

The command was `python run_cuda_smoke.py` from the output directory, with
`PYTHONPATH=C:\CodexArtifacts\pretrained-reuse-r1\python-deps`. The runner
uses the checked-in `MiniWoBDinoDecisionRuntime`, local-only artifact loading,
and the already saved TRAIN replay as the parity reference.

## Attempt note

The first harness attempt loaded the model and completed inference but then
raised `KeyError: 'option_ids'` while parsing the runtime's `options` JSON
array. This was an error in the temporary comparison harness, after model
inference; it did not indicate a model or runtime failure. A corrected
attempt passed. A subsequent clean attempt used a separate output directory
and is the result pinned above. No repository source code or experiment
artifacts were modified by those attempts.

## Scope limits and next gate

This reconfirms only the already measured synthetic screen-only task's
single-example JSON handoff on CUDA. It does not establish natural image,
video, audio, text, sensor, or multimodal quality. The train/validation gap
and the narrow task limits documented in the original results remain. R2's
rights-cleared, source-compatible task requiring multiple input streams is
still open; this smoke does not justify shared-backbone work or product
promotion.
