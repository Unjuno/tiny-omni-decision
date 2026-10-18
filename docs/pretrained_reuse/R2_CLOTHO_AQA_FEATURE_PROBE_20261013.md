# Frozen audio features: actual 16-clip probe

Ran `python -m scripts.probe_clotho_audio_features` with the existing process-local
`PYTHONPATH=C:/CodexArtifacts/pretrained-reuse-r1/python-deps`. No installation,
training, model update or package upgrade. Strict checkpoint load and pinned
EfficientAT source revision checks passed. Actual model count1,961,239 parameters.

All16 licensed training files passed recorded SHA verification. Native mono
float PyAV decoding, torchaudio resampling to32kHz and official eval mel frontend
were reused from the prior FSD/STARSS path. Frozen eval-mode whole-clip features
are16x384, all finite, with bitwise-identical repeated inference for every clip.
Extraction plus repeat and comparison took2.694597s CPU excluding model load.
FeatureSHA256 `88e84347e13d29455754a14dc87b70798a77a2868926fc74967aa237c9fefe4b`.

No question labels were scored. These pooled features do not by themselves
establish temporal-order preservation or audio-question quality. Nearest-feature
cosines against existing STARSS are diagnostic only, not waveform leakage checks.
The unresolved filename and full waveform provenance gates remain open; this
16-clip probe does not replace or reduce the frozen development corpus.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-frozen-feature-probe-01/`
contains features.npy and result.json with input/model/code hashes, source
revision, IDs, sample counts and runtime. The compact report is copied alongside
this document. Ruff passed for the new probe. GitHub CI for753a9ba passed both
checks before this change. Remaining work: full media identity resolution,
acquisition/integrity, text representation, and matched conditional/unimodal
experiments; no quality improvement is claimed from this execution check.
