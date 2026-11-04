# Native training cache preparation: unchanged eight-frame video features

This bounded extension persists an existing, verified extraction profile for
the unified native candidate. It is not another readout/quality experiment.
The historical456-scene run kept full tokens only in RAM; only pooled features
and raw-token digests survived. Native conditional readout training needs the
full tokens, so reproduce them once into a reusable, immutable local cache.

## Fixed inputs and operation

- Same original256 TRAIN /200 development-validation CLEVRER scenes, source
  manifest order and historical token SHA-256 from `video-token-readout-01`.
- Same original frozen VideoPrism tree, `_decode_video` / `_preprocess`,8frames,
 288x288,2048x768 FP32 token arrays. These are the same preprocessing functions
  used by the integrated native candidate. No16-frame pooled Lombard substitution.
- Validate every media hash, split and underlying filename/path identity before
  model execution. Every output must exactly equal its historical raw-token
  digest and reproduce after save/reload. Never drop a failed scene.
- Existing worker `scripts.audit_token_ram_transport --worker --live` runs in
  its existing CPU JAX environment. No dependency changes or model fitting.
- New output only; each tensor file is created exclusively. Incomplete runs
  stay INCOMPLETE and retain completed files. An observation timeout polls the
  same child; it never triggers a replacement extraction.

## Bounds and verification

Whole extraction cap1200s; process-tree sampled RSS cap6GiB; disk reserve3GiB.
Expected tensor files2,868,962,304bytes. Check free space for the whole corpus
plus reserve before starting. Only this invocation's new worker can be stopped
on resource/hash failure; existing runs and artifacts remain untouched.

Tests cover complete scene accounting, identity leakage, inconsistent duplicate
media, dtype/shape/nonfinite/hash rejection, exact reload and overwrite rejection.
Run targeted tests/Ruff before extraction and full CPU/manifest checks before
publication. Record code/config hashes, environment, command, time/memory, all
456 file digests and historical parity. No sealed audit or calibration payloads.

This makes video features available to the next shared-readout fit. Full common
supervision, language query/token caches, joint-input coverage, task quality,
retention, calibration and deployment gates remain separate outstanding work.
