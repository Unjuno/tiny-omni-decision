# VideoMAE global-feature count separability probe — 2026-11-10

## Frozen question and limits

This one-shot diagnostic asks whether the existing 384-dimensional, globally
pooled VideoMAE V2 representation carries linearly accessible object-count
information on the frozen Clevr-4 count task. It follows the count-error
diagnosis in `R2_VIDEOMAE_CLEVR4_100K_PATH_A_RESULTS_20261110.md`.

The predeclared candidate is a single closed-form multi-output ridge readout
with regularization `lambda=1.0`, fit on all 72,010 train images and evaluated
once on the same 1,000-image selector pool used by the earlier fixed Path A
experiment. Input coordinates are standardized with train-only means and
standard deviations (zero-variance coordinates use scale 1). Targets are
centered one-hot count classes 1–10; prediction is the highest ridge score.
No hyperparameter or seed search is performed. The primary metric is selector
count Accuracy, with macro recall, per-class recall, and confusion matrix as
diagnostics. Ridge scores are not treated as calibrated probabilities.

This selector was already used to select the earlier MLP checkpoint, so this
is an exposed development diagnostic, not a blind evaluation. It must not be
used to make a product-quality claim or to reopen the 5,296-image one-time
reporting pool. No reporting-pool features or labels, official validation, or
sealed-audit data are inputs. The existing checkpoint, corpus, and result files
remain read-only.

## Inputs and isolation

- Base branch/commit: `codex/path-a-videomae-clevr4-100k-20261110` at
  `7d65f11e`.
- Source archive: existing Clevr-4 100k v1 archive, 36,057,238,315 bytes;
  source receipt pins SHA-512
  `7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c`.
  Each consumed image is checked against its frozen individual SHA-256.
- Train media/label hashes: `e48687af785a74864d2d4eb6e632a50d381d821a3ade8c2227eda7624647f410` /
  `45de9d36d557619789dd6160e808f7ff0c531f06a298f0b2af80c9720103b695`.
- Selector media/label hashes: `4c551e91d4e02bd896380476d0e59f1348a01ed3f95a5eed7e0d39cb9a311dee` /
  `c08af3839089638d081dd31c1118be7256418e6c2440a70a55f79e50456fb920`.
- Encoder: OpenGVLab VideoMAE V2 ViT-Small K710, frozen, 22,153,286
  parameters; source SHA-256
  `d132c97369f25207e380e78c59294fd00b93d2f6aa0fb5c4ded19eff26a5a3cb`,
  checkpoint SHA-256
  `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`.
- The new extraction cache contains only train and selector IDs. It is written
  to a unique external attempt directory on D:; no existing artifact is
  overwritten.

## Interpretation rule

If the fixed ridge readout is materially better than the selected MLP on this
same selector, the global feature may contain more accessible count signal
than the MLP used; this does not demonstrate fresh generalization. If it is
similar or worse, that is consistent with a limitation of the pooled feature
or the linear probe, but does not prove VideoMAE or its spatial tokens lack
count information. Either result is diagnostic only; no model is promoted.

## Execution record

Pending. Results, code/input/output hashes, measured runtime and peak VRAM will
be appended after the isolated train+selector extraction and single fixed
readout finish. No quality result is claimed by this protocol.
