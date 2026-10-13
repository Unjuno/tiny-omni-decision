# Frozen VideoPrism Torch port: numerical gate failed

This bounded runtime optimization is **not adopted**. Full native-token parity
failed; no corpus features, readout or product artifacts use the Torch port.
The native JAX CPU runtime remains the extraction reference. No training,
quantization, dependency upgrade, legacy artifact edit or sealed-audit access.

## Fixed implementation and checks

Source `google-deepmind/videoprism@c6eba0ece3dd49a24ffc44582966997a4a59f1d8`,
Apache-2.0 operators; all40 loaded native arrays exactly equal the NPZ source.
Checkpoint SHA`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`;
114,365,184 FP32 frozen parameters; native16x288 RGB profile. No interpolation.
The port keeps scale+1 LayerNorm, centered variance/epsilon1e-6, exact GELU,
query scaling,50*tanh(logits/50), stacked projections and chronological output.
Shape/schema/dtype/finite checks reject partial loading, training, autocast and
CUDA TF32. Its module explicitly labels it an unqualified research backend.

Fresh read-only review found the inherited Torch state-dictionary loader bypassed
constructor validation: wrong dtypes silently cast and missing keys could leave
partial copies. Eight regression cases were observed failing before the fix.
The reload API now preflights every key/shape/FP32/finite tensor before copying,
rejects non-strict/assign modes and non-FP32 destination weights. Valid reloads
remain supported. Direct manual parameter edits are outside this controlled API;
requires_grad=False does not make a Torch object tamper-proof.

Independent NumPy attention and native tiny Flax model checks pass (tiny maximum
absolute difference about3.58e-7). Those checks alone do not establish parity for
the pretrained full model. CPU tests856 passed,1 skipped before review; after
the reload fix864 passed,1 skipped, with Ruff passed. Fresh review found no
Critical/Minor issues and one Important reload finding, fixed with the eight
regression cases above. Commit`b71378fbb6df5d41021e3588ef4d6732f8ba62e9`
push/PR CI both succeeded.

Review dispositions: full-model parity repair/adoption remain deferred(cost:
native CPU time); the reviewer checked saved hashes/tiny outputs rather than
rerunning heavy CPU/CUDA/CI(cost:no independent full-device replication); direct
manual parameter/gradient re-enabling remains outside the controlled loader
(cost:freezing is not a tamper-proof security boundary). No deferred Minor findings.

## Predeclared real-model gate and observed failure

Identical saved preprocessed inputs and full native outputs from four **training**
videos; float32, TF32 off, seed17, no label inputs. Before comparison the full
token gate was `allclose(atol=1e-4,rtol=1e-4)` and the framewise spatial-mean
maximum absolute difference had to be<=1e-4. Neither condition was relaxed.

The original gate stopped on first CPU clip`s16_l_bgah2n`: full-token maximum
delta0.00666654, framewise mean delta0.0000490025, full allclose **false**.
Single forward2.616s; same-backend repeat bitwise. The gate never reached its
CUDA phase or saved/reloaded the proposed deployment checkpoint.

A separately labelled diagnostic subsequently measured the four CUDA forwards:

|Training source|Full-token max delta|Framewise max delta|Seconds|Full gate|
|---|---:|---:|---:|---|
|s16_l_bgah2n|0.00505063|0.0000222623|0.3991(cold)|FAIL|
|s16_p_bgah2n|0.00366056|0.0000152364|0.2401|FAIL|
|s22_l_sgbe5s|0.00253683|0.0000182390|0.2387|FAIL|
|s22_p_sgbe5s|0.01335426|0.0000516567|0.2399|FAIL|

All repeats bitwise, all finite. Peak CUDA allocated839,142,400bytes,
reserved903,872,512bytes on RTX3080Laptop16GB/Torch2.6.0+cu124.
Warm speed is promising but does not override failed parity. Mean pooling hides
some token discrepancies; passing that statistic cannot replace the full gate.

## Localization and limits

The first patch projection already differs by about6.71e-6. Spatial encoder
max delta0.00333381 and final output0.00666654. Feeding native positioned patches
to the rest of the Torch implementation still leaves CPU final delta0.00161353
and CUDA0.000900745. Diagnostic addmm, folded additions and FP64 patch-dot variants
reduce some errors without meeting the original gate; none is adopted.
Most token errors are much smaller than the maxima. Cancellation/low-variance
background patches may amplify backend rounding, but this is a hypothesis, not
a proved sole cause or justification to loosen the threshold.

Keep the negative result and stop this optimization branch before a full port
job. Continue the scientific comparison with native CPU features instead of
spending further iterations on an unqualified backend. Quality remains unmeasured.

## Reproduction and evidence

Runners and preserved negative/successful diagnostic artifacts are under
`C:/CodexArtifacts/pretrained-reuse-r2/videoprism-torch-*` and
`.../videoprism_torch_*.py`. The adjacent JSON embeds exact result records and
runner/result hashes. The initial gate's FAILED artifact remains unchanged.
Run CPU code checks with `python -m pytest tests/test_videoprism_torch.py -q`;
real checkpoint/reference tests are intentionally outside normal CI.
Native source and preprocessing are documented in
[the visual smoke report](R2_LOMBARD_GRID_VISUAL_SMOKE_20261014.md).

Execution ruling: a completed negative parity experiment satisfies the bounded
diagnostic's evidence requirement, **not** the backend acceptance gate. A full
Torch extraction remains prohibited. This costs CPU time for the native route,
and leaves future backend parity work open. The full Omni goal remains active.
