# DINOv2 spatial-information control

Previous frozen CLS+mean baseline count Accuracy53.71%, overall85.06%; eight
epochs did not prove a plateau. Test a representation intervention with matched
readout capacity, not a claim that global pooling is already established as cause.

Keep the pinned backbone, whole-image224x224 preprocessing, all8,424train and
512selection images,40taxonomy options, frozen MiniLM embeddings, seed17,
8epochs,AdamW1e-3/wd1e-4,batch64questions and earliest min validation NLL.
Use only the same observed selection validation; no reporting/audit access.

Extract final16x16patch grid. Average each nonoverlapping4x4patch block to yield
4x4spatial cells. Store CLS plus16cells, FP32. Verify CLS and mean-of-cells
reproduce the prior CLS+patch-mean cache within1e-5. Verify all media/split hashes.

Compare two new readouts, with identical initial tensors and sample orders:
1. global-control: repeat the global patch mean at all16cell positions.
2. spatial: retain the16cell means in fixed row-major order.
Both concatenate CLS384, flattened16x384cells divided by4, and candidate384.
The1/4factor makes the repeated-global patch block have the same L2norm as its
single vector. It does not force spatial/mean feature distributions to match.
Both have6912input dimensions,128hidden,884,993trainable parameters. No backbone
updates; total stack40,331,393parameters. Capacity-matched control is primary;
the previous smaller147,713parameter readout remains historical reference.

Separate16-image/one-epoch smoke tests feature/layout guards and reload. Full run
uses new dinov2-spatial-control-01 outputs, with separate best/final checkpoints
and per-condition histories. Do not extend budgets or search seeds. Compute
inputs in bounded batches rather than materializing all expanded candidates.

Report all epoch train/validation gaps and four-task Accuracy/NLL/Brier/ECE,
chosen epochs, examples/assets, gradients, runtime/VRAM, hashes and parameter cost.
After selection, compare both matched images and cyclic+1image controls on all
512images, retaining chance label matches. Paired512image bootstrap10,000draws,
seed17, percentile95% Accuracy difference spatial-minus-global. No selection
based on bootstrap or rotated metrics. This observed development comparison is
not a blind95% claim. No generic-question/video capability inferred from taxonomy
classification; no product replacement, compression or old artifact changes.
