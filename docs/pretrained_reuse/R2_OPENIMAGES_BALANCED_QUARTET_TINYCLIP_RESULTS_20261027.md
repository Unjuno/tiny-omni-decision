# TinyCLIP on fixed same-option train quartets

The [fixed protocol](R2_OPENIMAGES_BALANCED_QUARTET_TINYCLIP_PROTOCOL_20261027.md)
used the existing 15 disjoint train-candidate quartets, all 60 JPEGs, the
audited frozen TinyCLIP checkpoint and the previously fixed prompt on CPU.
All source media hashes and page declarations were checked. The later 12
source-disjoint candidate quartets were not evaluated: two still lack media.

| Input | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|
| Correct image | 0.4500 | 1.4162 | 0.6942 | 0.2401 |
| Fixed deranged image | 0.2000 | 2.7420 | 1.1877 | 0.5074 |
| Uniform processor-mean image | 0.2500 | 1.6731 | 0.8854 | 0.2933 |
| Deterministic option-only predictor | 0.2500 | not fitted | not fitted | not fitted |

Correct-image predictions changed correctness on 29/60 rows under image
derangement and 26/60 under blanking. Blank-image Accuracy is exactly 25%
because all four rows in each quartet have identical ordered options and each
option is correct once. Aligned mean confidence was 0.6901 versus Accuracy
0.45, consistent with poor calibration on this small diagnostic.

The task was constructed after earlier train-candidate inspection and contains
only seven of 11 classes. It is not an independent validation set, not an
estimate of general image understanding, and cannot establish multimodal
quality or the 95–98% product target. Page declarations do not independently
verify ownership, and overlap with TinyCLIP's YFCC pretraining data is unknown.
No training, model selection, prompt search, calibration fitting, or sealed
evaluation occurred. The existing evaluations were not altered.

### Paired within-quartet uncertainty diagnostic (2026-10-28)

The saved prediction file was rehashed against the pinned SHA below and all 15
four-row groups were checked for identical options and one correct occurrence
per option. Aligned-image minus blank-image Accuracy was +20.00 percentage
points. Resampling the 15 complete quartets with replacement 10,000 times
(NumPy seed 17, percentile 95% interval) gives [+6.67, +33.33] points. This
interval describes this small, previously inspected development sample under
quartet resampling. It does not establish independent photographer or source
groups, resolve pretraining overlap, or measure production performance. The
deranged-image control is excluded from this bootstrap because donor images
cross quartet boundaries. No new model inference or checkpoint selection ran.

Raw SHA-256 for
`C:/CodexArtifacts/pretrained-reuse-r2/openimages-balanced-quartet-tinyclip-20261027-attempt-01/summary.json`:
`ad55c0dc20a65307604dc26ef0e24877d3c5ad29a53e9492f3af1a9cf24bb29d`;
for `private-predictions.jsonl`:
`812b4994961cf9049e3ba6e19d5a4203ef9e9c9e1c479a1e1b9a3661662cf64c`.
The summary records model, processor, task, page, code and protocol hashes.
