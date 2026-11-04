# Frozen FSD50K audio feature smoke

All eight previously fixed training clips produced finite384-dimensional
EfficientAT dymn04_as embeddings, with exact repeated inference in eval mode.
Used the same pinned1,961,239-parameter checkpoint and model source as STARSS23.
Decoded native mono floats using PyAV, resampled44.1kHz to32kHz with torchaudio,
and applied official eval mel preprocessing. No model updates or thresholds.

Measured extraction plus repeated inference:0.711s CPU, excluding loading.
Full clip durations1.13–20.38s were retained; no clip label was assigned to
arbitrary5s subwindows. Final feature SHA256:
`69e03e7b8aec0aaa36e893c8054c662ee9d57021cc065ed2b1fc9044059f141f`.

Nearest cosine against all2,720 already-extracted STARSS development features
ranges .7381–.8652. Exact nearest IDs/splits are saved without reading their
labels or selecting a model. This is semantic similarity triage, **not** an
audio fingerprint, an independent-source guarantee, or a license to exclude
validation examples. Whole-clip FSD embeddings and5s STARSS embeddings differ
in duration. Cross-source near-duplicate verification remains open.

Initial attempt failed at import because the default Python path lacks
torchaudio. Preserved its source in `fsd50k-frozen-feature-smoke-01`; no features
were written there. The second attempt used the already-working dependency
directory from the original STARSS extraction. No packages were installed or
upgraded, and no old output was overwritten.

```powershell
$env:PYTHONPATH='C:/CodexArtifacts/pretrained-reuse-r1/python-deps'
python -m scripts.probe_fsd50k_features
```

Outputs: `C:/CodexArtifacts/pretrained-reuse-r2/fsd50k-frozen-feature-smoke-02/`.
Input media report, checkpoint, source and reference feature hashes are recorded.
Ruff passes. This confirms feasibility of reusing the existing audio path for
licensed additional positives; it does not establish improved decision quality.
No sealed audit, training, GPU job or existing run was accessed or changed.
