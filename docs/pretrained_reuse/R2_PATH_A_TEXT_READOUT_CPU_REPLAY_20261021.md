# Path A text readout CPU replay (2026-10-21)

## Scope

The selected, serialized Typed Decisions Synth readout was reloaded on CPU and
scored against all 2,540 rows of its existing frozen validation-feature cache.
No encoder, optimizer, model selection, calibration fit, media, or sealed audit
was used. This verifies the cached-feature Decision head and its saved weights;
it does not verify end-to-end text encoding or general text quality.

## Pinned inputs and command

- Run report SHA-256: `d9a8aa50d66ff13ea81071416789c9c34b989c78de13c16a30437f0e735642bf`.
- Feature cache SHA-256: `6dbe639194d172d4223abeb768baadb6c1a89e0fa631907afcc49c962692d344`.
- Best readout SHA-256: `12d8278cb0c13c384216ba706c4edd39a30ce6888f6a1900ea7ce26304bf02d0`.
- Validation source SHA-256: `7be1d1624752938553e13e1eecf5363ca2c56dd264da2915b97ddae8d0b60a8`.
- Selected epoch: 3; readout parameters: 196,865.
- Runtime report: `C:\CodexArtifacts\pretrained-reuse-r2\text-probe-v0\cpu-reload-replay-01\report.json`, SHA-256
  `05649d39ee63738e962441b362ff44b320bf26ca5f9fd3b19195834ebe5e8d9a`.
- Python 3.11.9, PyTorch 2.6.0+cu124, CPU device. The GPU was not initialized.
- Replay command:

  ```powershell
  python scripts/replay_cached_decision_readout.py `
    --features C:\CodexArtifacts\pretrained-reuse-r2\text-probe-v0\frozen-features.pt `
    --readout C:\CodexArtifacts\pretrained-reuse-r2\text-probe-v0\best-readout.pt `
    --run-report C:\CodexArtifacts\pretrained-reuse-r2\text-probe-v0\run-report.json `
    --output C:\CodexArtifacts\pretrained-reuse-r2\text-probe-v0\cpu-reload-replay-01\report.json
  ```

## Result

| Metric | Original selected validation | CPU replay | Absolute difference |
|---|---:|---:|---:|
| Accuracy | 49.3701% | 49.3701% | 0.000002 pp |
| NLL | 1.030276895 | 1.030277003 | 0.000000109 |
| Brier | 0.594610393 | 0.594610330 | 0.000000063 |
| ECE-15 | 0.026798876 | 0.026798884 | 0.000000007 |
| Mean confidence | not in prior report | 0.510805 | — |

Replay status: `PASS_CPU_FROZEN_FEATURE_READOUT_REPLAY`. Cached readout
outputs and aggregate validation metrics are reproducible within floating-point
roundoff after serialization/reload. The head remains weak on this synthetic
benchmark and its validation was already used for selection; this is not blind
or product-quality evidence. Full end-to-end CPU encoder and the image/audio/
video runtime profiles remain separate work.
