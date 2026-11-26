# Native common readout CPU replay (2026-10-29)

## Purpose and scope

This replay checked that the trained Path A common readout can be exported,
reloaded, and run on CPU over the same cached-feature validation rows used by
the fit. It is a runtime/parity check on cached frozen features, not end-to-end
inference from raw text, images, audio, or video. The 24,346 validation rows
are exposed development data; this is not an independent quality estimate.
No training, checkpoint selection, calibration, sealed-audit access, or CUDA
work was performed. The trained checkpoint and all pre-existing fit artifacts
were read-only.

## Pinned inputs and outputs

- Fit source commit: `72c1eb47dde0c81e503d3a4d783aa5528660186e`
- Replay source commit: `69f320fb26b19de33befbbe2fd85637c8db41cfe`
- Fit config SHA-256: `45954a7240ee8da32e4b5b5dc3283994c40c4849451f27ff4fce13d3cc118f75`
- Fit receipt SHA-256: `aaa1fe96a4be85a71ce146981c6b2e8b58e1308d91fc110e1ece6804b4d8048c`
- Selected checkpoint: step 1,792; SHA-256
  `3eed6558731de60da82e75e38e599699f29663dd601f942505fec26a48b8dc1d`
- Validation IDs/order/options/targets SHA-256:
  `85cec27e5b653c44694094d2268671847caf54877124a3e933ca3581ef9f4815`
- External output directory:
  `C:\CodexArtifacts\pretrained-reuse-r2\native-common-cpu-full-replay-02`
- `report.json` SHA-256:
  `25795fa06045b9bc42d3b2c0c9d5aef40c783fad32ed03071ee24249f966110c`
- `validation-scores.npy` SHA-256:
  `d0cf4a2413437d419664243794d4ad66064e1f3613ec3720dcb4a913db8a071f`
- `validation-metrics.json` SHA-256:
  `573c205f9673b9c64497a683b93b827f161e235649421e5b4477b403356f429a`
- Exported readout state SHA-256:
  `cade80fb406549f10ccb0541c2d682ae79dc7550543004c7be0d94a476a09c1f`

## Result

All 24,346 rows were scored in their pinned order. CPU and saved GPU predictions
had **0/24,346 top-option disagreements**. Maximum absolute option-logit
difference was `3.52859497e-5`; accuracy was unchanged and macro metric
differences were below `1.3e-8` in absolute value.

| Validation slice | Count | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text | 10,732 | 20.89% | 1.92106621 | 0.82368459 | 0.07423809 |
| Image | 2,048 | 20.12% | 2.16512196 | 0.86517808 | 0.03114653 |
| Audio | 11,166 | 54.28% | 1.14673480 | 0.53640444 | 0.07067944 |
| Video | 400 | 31.50% | 1.33925248 | 0.70061116 | 0.08419295 |
| Macro | — | 30.71% | 1.68653328 | 0.74018041 | 0.08424858 |

The first attempt compared CPU/GPU scores with an absolute tolerance of
`2e-6` and stopped after the complete forward pass when the observed maximum
difference was `3.53e-5`. The rerun used `5e-5`, chosen after seeing that
first result. Therefore that tolerance was **post hoc**, not a preregistered
acceptance gate. The evidence to retain is the observed score drift, zero
top-option disagreements, and near-identical metrics; do not describe this as
exact floating-point parity. The first attempt's partial output was kept in
the separate `native-common-cpu-full-replay-01` directory; the second attempt
used a new directory and did not overwrite it.

## Timing and machine constraints

The full CPU scoring pass took `58.80 s` (`414.0 examples/s` overall). The
first three 256-example windows measured `652.7`, `454.2`, and `680.7`
examples/s, averaging `595.9 examples/s`. That average predicted about 40 s
remaining after the initial three samples; actual remaining time was about
58 s from scoring start, approximately 19 s longer as later throughput fell.
Waiting for 30 s and then about 20 s matched the observed completion window.
This is a useful local wait estimate, not a stable benchmark.

At the post-run check, Orbitune PID 22136 remained alive and RTX usage was
15,821/16,384 MiB at 99% utilization; the replay itself used CPU only. C: had
about 0.72 GiB free. No process or existing model/data/checkpoint was changed.

## Interpretation

This closes only the CPU execution check for the trained readout over frozen
cached features. It does not package or run the multimodal encoders, prove raw
input preprocessing parity, alter the 196,254,874 pre-quantization parameter
count, meet the provisional package-size target, establish product quality, or
justify promotion. End-to-end raw-input runtime and independent task-quality
gates remain open.
