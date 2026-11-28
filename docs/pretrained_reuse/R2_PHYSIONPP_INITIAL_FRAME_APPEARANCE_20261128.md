# Physion++ initial-frame appearance diagnostic — 2026-11-28

## Question

Can a fixed low-level appearance representation of the first video frame
predict the existing Physion++ train label, without seeing later motion? This
is a narrow train-only diagnostic for static appearance and initial-state
signal. It is not a video model-quality score and cannot by itself identify
rendering leakage, invalid labels, or causal shortcuts.

## Frozen protocol

- Input: 652 rows from the pinned train manifest only; manifest SHA-256
  `0de1bbbb1cc54000936c1771ea54e375c214fedda36734536f31a44c4f64932b`.
- All 652 video paths and SHA-256 values were verified before decoding.
- Decode only the first frame from each train clip; convert to RGB and average
  pixels in an 8×8 spatial grid (192 values normalized to [0,1]).
- Fixed classifier: Gaussian naive Bayes, variance floor 0.05; no hyperparameter
  search or checkpoint selection.
- Five deterministic folds assigned by
  `int(SHA256(source_asset_group_id)[:4])/modulo 5`; all rows from a source asset
  group remain in one fold.
- Compare pooled out-of-fold predictions against the same fold-trained class
  prior and the existing recorded-factor-only diagnostic.
- Validation/sealed/test rows were not read. CPU only; no GPU or pretrained
  checkpoint used.

## Results

| Train-only grouped OOF readout | Accuracy | Balanced accuracy | NLL | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|
| Fold-trained prior | 0.5199 | 0.5000 | 0.6951 | 0.2510 | 0.0003 |
| First-frame 8×8 RGB Gaussian NB | 0.5353 | 0.5395 | 1.6843 | 0.3908 | 0.3616 |
| Recorded factors-only NB | 0.5982 | 0.5978 | 0.6849 | 0.2416 | 0.0873 |

The first-frame classifier has a small Accuracy lift over the prior but poor
probability quality. The much higher NLL and Brier show that its fixed Gaussian
assumptions are a poor calibrated readout for these features. Recorded factors
alone carry more grouped OOF signal than this static appearance baseline.

## Interpretation and limits

This result does not show that the target is visually unobservable: the
diagnostic discards temporal motion and uses a deliberately small color-grid
representation. It does not rule out richer first-frame cues, unrecorded
simulator settings, or renderer-specific cues. The recorded-factor model's
stronger result remains a shortcut-risk signal, not proof that a video model
uses those factors.

Retain the preselected blinded 75-pair, 32-frame review pack for an independent
human observability review. Analyze within-stratum rendering statistics before
capacity changes, and keep any conclusions limited to this train corpus.
These observed rows cannot support a blind generalization claim.

## Reproduction artifacts

- Script: `scripts/audit_physionpp_initial_frame_baseline.py`
- Tests: `tests/test_audit_physionpp_initial_frame_baseline.py`
- Output directory:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-initial-frame-20261128-attempt-01`
- `audit-report.json` SHA-256:
  `fb4849e978f4152d6171d0e1d7b6bb43842af130416d86242f7ea316760bfdcf`
- `oof-predictions.jsonl` SHA-256:
  `2f56f496fa253f97deb59af520996cc3753b8e4eaa0b5572ed42c7c77d2bb5c4`
- `decoded-media-hashes.jsonl` SHA-256:
  `517a3d14bc973312c905b678deee4001d52d3f614e23ddb6827b90a8f6e083e8`
- Script SHA-256:
  `bddaA2bf3574cf2f7634b99528242e66056d7907fa534c5f3392dacba590fbd9`
