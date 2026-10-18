# CLEVR-4 count training-only diagnostic — 2026-10-19

## Scope and integrity

This is a read-only CPU diagnostic of the selected checkpoint from
`textfree-image-decision-512-01`. It did not fit weights, select a checkpoint,
or score the internal development or official validation examples. The
analysis reconstructed the original seed-17, 10% asset-hash split and used only
its 7,582 training images. The 842 internal-development rows were excluded;
official validation and sealed audit were not opened.

Input hashes matched the earlier run record:

- Feature cache: `f36b9cf17efed6a5137eb48ffcaf40ff960015fc07bc194fb532bf5562d88fab`
- Training labels: `00f03f27982e2785636e85a50f4ecf258c0ac3ee483d7242e297cee14b50ec2e`
- Training media manifest: `2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010`
- Selected readout: `e62711ca1b8fd68df5037cd135266ed7f556e0ad006359305f10e6665789a47d`
- Reconstructed train-ID order: `3df06d4fd9e902fd73fe990804f4a71aec0fe7e7e029b20d463c905609543021`

The machine-readable matrices and per-class counts are preserved outside Git
at `C:\CodexArtifacts\pretrained-reuse-r2\clevr4-count-train-diagnostic-01\training-only-diagnostic.json`
(SHA-256 `217e42c369de0765d8e6d30bc535477223acf218adfce89b349da58576742eb6`).

## Findings

| Taxonomy | Saved linear head, train accuracy | In-sample nearest-centroid accuracy | Majority-class accuracy |
|---|---:|---:|---:|
| Color | 100.00% | 82.64% | 10.60% |
| Texture | 97.86% | 55.55% | 10.96% |
| Count | 69.12% | 41.16% | 10.55% |
| Shape | 100.00% | 95.54% | 12.25% |

For count, most linear-head errors are among adjacent count labels. The largest
directed confusions are 5→6 (171/759), 9→10 (246/768), 7→6 (176/775),
7→8 (158/775), and 4→3 or 4→5 (109 or 110/728). Count 1 recall is 100%;
count 10 recall is 88.79%, with predictions overproducing label 10 (1,010
predictions for 758 examples). The full confusion matrix is in the external
JSON receipt.

The linear head materially exceeds the count majority prior, so the frozen
feature has useful count signal on the training examples. Its limited
training accuracy and the weaker in-sample centroid result do not isolate
whether the main limitation is the pooled feature, the linear boundary, the
task labels, or generalization. The previously reported spatial bilinear
readout's 70.51% count accuracy on reused development validation is consistent
with spatial information being useful, but that validation was used for model
selection and is not independent evidence.

## Decision

Do not add training steps or promote the current image head based on this
training-only diagnostic. If CLEVR-4 count remains a useful development task,
the next experiment should test an explicit object/spatial readout on a
predeclared, uncontaminated split, with the existing global and spatial
readouts as fixed references. Keep the result narrow to synthetic CLEVR-4
counting; it does not establish natural-image reasoning or product quality.
