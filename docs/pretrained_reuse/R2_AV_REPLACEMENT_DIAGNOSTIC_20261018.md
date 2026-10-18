# E0G1 AV replacement diagnostic — 2026-10-18

## Run and integrity

- Status: `PASS_POSTNORM_AV_REPLACEMENTS`.
- Source branch/commit: `codex/r2-av-postnorm-stability`, `216b16bde7299c60915052159631f1cc3868e602`.
- Command: `python C:\CodexArtifacts\pretrained-reuse-r2\score_after_orbitune_yield_20261018.py --arm E0G1 --attempt 11 --max-seconds 7200`.
- The external launcher SHA-256 is `9d06c367870cbebe94ad65b679f4b117d2b3a05461134c93aa045cc77a9718f7`. It held a small CUDA reservation until no other GPU process had appeared for five seconds, then invoked the committed scorer in the same process. The process check excluded only its own PID; other GPU processes remained guarded. No repository source was modified for the run.
- Fixed arm: E0G1, selected step 24,576; checkpoint SHA-256 `9df53a55e610f4e473dc5f56528edd86610a72683c8f6c40145e9d87d899d87d`.
- Frozen validation inventory: 8,875 questions; 44,375 model forwards across the five recorded views; 17,750 exact self replacements. Pairing SHA-256 `4960fb4829c9f87df832ae2044b06ef96cca62d65ce4ff56f62374372eac58c4`.
- Input digest: `00aa90bed282446e77ae834afc4d9b1d1c1adeedcbaf73e47a49f084c1838868`.
- Final integrity verification covered 2,755 input pins and 1,120 source pins.
- Sealed audit and calibration data were not loaded. Optimizer and encoder updates were both zero. Cloud cost was zero.
- Wall time: 733.37 seconds, including pin checks and final verification. Peak working set: 1,961,504,768 bytes; peak CUDA allocation: 116,765,696 bytes. Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 3080 Laptop GPU.

## Results

The scorer compares natural-input predictions to predictions after replacing one media stream with the frozen donor pairing. Rates below describe how often the argmax changed and how much the probability distribution moved; they do not measure correctness.

| Readout | Donor stream | Questions | Argmax flip rate | Mean total variation | Mean JS divergence |
|---|---|---:|---:|---:|---:|
| Baseline | Audio | 8,875 | 0.367549 | 0.329036 | 0.131192 |
| Baseline | Video | 8,875 | 0.582310 | 0.539991 | 0.302920 |
| Query-only | Audio | 8,875 | 0.380056 | 0.334018 | 0.134566 |
| Query-only | Video | 8,875 | 0.561915 | 0.525610 | 0.293329 |
| Full residual | Audio | 8,875 | 0.380056 | 0.334018 | 0.134566 |
| Full residual | Video | 8,875 | 0.561915 | 0.525610 | 0.293329 |

Full-residual argmax flip rates by task were:

| Task | Questions | Donor-audio flip rate | Donor-video flip rate |
|---|---:|---:|---:|
| Action | 2,987 | 0.384332 | 0.192166 |
| Material | 2,978 | 0.458697 | 0.889187 |
| Reaction | 2,910 | 0.295189 | 0.606529 |

The full-residual aggregate sensitivity is identical to query-only at the reported precision. The baseline also changes substantially under media replacement, especially for video. This establishes that the fixed model's outputs are sensitive to these replacements on this inventory. It does **not** establish that the donor input is semantically valid, that changes are correct, or that either stream improves task accuracy. The artifact explicitly records `counterfactual_truth_verified: false`; this is a sensitivity diagnostic, not a causal or quality evaluation.

## Artifacts

Output directory: `C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-postnorm-replacements-E0G1-11`.

| File | SHA-256 |
|---|---|
| `result.json` | `d3961fe145667a3dcfdd271a328786e7206ac6aea2b1e9823d1aa37d29734ed5` |
| `predictions.jsonl` | `eea8d109d90032800b198cca87b3dd9dfd0206f02d42115153c0799027ad016e` |
| `sensitivity.json` | `f6287e4c6437d70ce1142db82347fd8a652b76e5ebb1ecbab70cbc2c3c040454` |

`predictions.jsonl` contains exactly 44,375 rows, and both self-replacement checks passed for all 17,750 required self cases. Existing attempts 08–10 remain preserved; only attempt 11 is a complete scorer run.

### Authorized retry attempt 12

The later authorization requested a fresh inference-only retry named attempt 09.
That output directory already existed as a preserved `INCOMPLETE` preflight
from 2026-10-17 with zero forwards, so it was not overwritten. A new directory,
`C:\CodexArtifacts\pretrained-reuse-r2\greatest-hits-postnorm-replacements-E0G1-12`,
was used instead. The committed scorer started its CPU integrity preflight,
then failed closed before model inference when a foreign GPU process appeared:
`GPU occupied; scoring not started`. Receipt: `INCOMPLETE`, 0/44,375 forwards,
0 optimizer updates, 0 encoder updates, `sealed_audit_loaded=false`,
`checkpoint_selection=false`, and cloud cost 0. The `result.json` SHA-256 is
`07b834fa9870829686ed191819009e726740af32776150b84d288c757b722c1`.

The Windows Orbitune monitor log shows the trainer cooperatively yielded after
detecting external GPU activity. No process was stopped or restarted manually,
and no checkpoint, training data, or prior attempt was changed. This retry adds
no model-sensitivity evidence. It can be resumed only in a GPU window where the
full pinned-input CPU preflight can finish before Orbitune reacquires the GPU.
The follow-up runner change moves the CUDA acquisition to after CPU preflight,
then waits for Orbitune's cooperative yield before model transfer and scoring;
details are in
[`R2_AV_REPLACEMENT_RUNNER_PREFLIGHT_20261018.md`](R2_AV_REPLACEMENT_RUNNER_PREFLIGHT_20261018.md).
Attempt 12 remains unchanged; a fresh attempt is required to execute with the
updated runner.

The first scheduler retry, attempt 13, also remains preserved. It stopped at
the initial GPU gate in 0.10 seconds after observing Orbitune trainer PID
`6744`; it performed zero forwards and did not enter the input-pin preflight.
Its `result.json` SHA-256 is
`8439b7ee52f55463619557cdd9ba9399c270cddb3243de7db37d4f0902590593`. The
current runner moves GPU acquisition until after CPU preflight and waits for a
clear device window, so attempt 13 is not being reused or overwritten.

## Interpretation and next step

This result weakens the narrow hypothesis that E0G1 ignores AV inputs entirely, but it does not answer whether the inputs are used *correctly*. The next useful test is a truth-bearing, source-paired task evaluation or a carefully constructed modality ablation where the replacement has a known expected outcome. Do not use these replacement flip rates as accuracy or as grounds to claim multimodal quality.
