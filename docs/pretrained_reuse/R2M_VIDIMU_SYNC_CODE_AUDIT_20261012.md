# VIDIMU synchronization code audit: verify published offsets before fusion

Inspected upstream revision `649fdf07513fddcf5b0462a1bd888a5b13319c9c`.
Source copies are preserved under
`C:/CodexArtifacts/pretrained-reuse-r2/vidimu-code-audit-02`.
No upstream GPL code was executed or incorporated into the product.

## Observed execution semantics

- The estimation notebook uses 180 samples at 30 Hz (six seconds), and searches
  integer shifts 0 through 14, separately in each direction.
- For each subject/activity, the utility chooses the first existing trial among
  T01 through T05, then breaks. This code does not establish synchronization for
  every trial. Missing metadata cannot automatically mean zero offset.
- Joint-angle signals are downsampled/interpolated, smoothed and centered before
  RMSE minimization. This is estimated alignment, not hardware synchronization.
- `getSamplesToSynchronize` returns both independently optimal directional shifts
  and flags identifying the lower-RMSE direction. The plotting/alignment path
  zeros the nonselected shift. However, CSV logging at lines 274-286 tests the
  original positive shifts rather than the selected flags. If both directions
  improved over the unshifted baseline, both can therefore be logged despite
  only one being used for the reported aligned RMSE.

This is a code-level discrepancy, not evidence that any particular distributed
record has incorrect offsets. Inspect actual `infoToSync.csv` rows for paired
positive offsets and trial coverage before interpreting corrected sensor data.
Do not silently repair/re-estimate offsets on validation data. Establish a fixed
alignment rule and record any excluded or unresolved trials before evaluation.
Original MP4 files are not trimmed by the modification notebook, as documented
in the acquisition report; corrected sensor and original video origins differ.

## Evidence hashes (SHA-256)

- EstimateFileSynchronization.ipynb:
  `a88c3bb86a27587ca30d99f820643bb48e27500e58f62b035174143e9c1e4865`
- syncUtilities.py:
  `f6f598ce9cd4644efde6beab41060d4fbc3be2e196c50e37e41741d11c7ab6c5`
- README.md:
  `57e411c34151e4a23d9a76278017a7d3d0304dd6eac5ea88b1de5d8358e2a331`

The dataset download remains in progress at this audit. No sensor payload,
held-out split, model training, or quality result is claimed here.

Source: [pinned upstream utility](https://github.com/twyncoder/vidimu-tools/blob/649fdf07513fddcf5b0462a1bd888a5b13319c9c/utils/syncUtilities.py).
