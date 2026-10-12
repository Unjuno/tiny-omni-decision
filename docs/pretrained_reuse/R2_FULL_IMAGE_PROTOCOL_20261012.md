# Whole-image preprocessing comparison

Motivation: the existing640x480 training images are resized384x288 and cropped
to288x288, discarding25% of width. The second inspected training image has an
object intersecting the left crop boundary. This is an information-loss risk,
not proof of the cause of counting errors. No validation-error inspection is
used to choose examples or preprocessing variants.

One new preprocessing policy: TensorFlow bilinear resize of the full RGB image
directly to288x288, divided by255, no crop/pad/augmentation. This retains field
of view while changing aspect ratio and effective horizontal object resolution.
The comparison identifies the effect of the complete preprocessing policy;
it cannot separately attribute crop removal, distortion and sampling density.

Keep the pinned VideoPrism weights, spatial intermediate, single-frame batch8,
17x768 cells, MiniLM option embeddings, existing8,424 train images and512
selector images/IDs/order. Do not read1595 reporting images or sealed audit.
Fresh cache `videoprism-full-image-cache-01`, preserve all prior caches.

One fresh seed17 global-arm readout: 13,056 observation dimensions,384 candidate
dimensions,128 interaction width;1,720,576 parameters. Same initial weights and
shuffled question order as `shared-spatial-image-readout-01/global` must match.
AdamW lr0.001,wd0.0001,batch64 questions,24epochs, earliest min validation NLL.
No spatial arm, extra seeds, learning-budget extensions or readout alterations.
Separate smoke first, then `shared-full-image-global-01`; keep best and final.

Primary reference is the original same-capacity cropped global arm. Report
all task Accuracy/NLL/Brier/ECE, train/validation gaps, complete learning curves,
paired512-image uncertainty and image-rotation control. Exact counting is the
predeclared weak task; retain all four tasks without changing labels/options.
No new blind-evaluation claim, arbitrary95% threshold claim or product promotion.
Mixed or negative outcomes are retained without further preprocessing search.

Before full extraction, actual TensorFlow synthetic side-band smoke must show
that both edges are retained and output is finite float32[288,288,3] in[0,1].
Before learning, require completed cache hashes and exact record order. Use
the existing local CPU encoder/GPU readout environments without upgrades.
