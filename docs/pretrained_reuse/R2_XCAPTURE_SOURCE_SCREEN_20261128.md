# X-Capture multisensory source screen — 2026-11-28

## Disposition

**Source license: `ALLOW` for a task-scoped cross-sensory retrieval experiment.**
The official project page links directly to the Hub dataset, whose pinned card
declares MIT; the Hub owner is a paper coauthor. MIT permits commercial use,
derivative training and redistribution, satisfying the repository's declared
license policy for this dataset. The project page does not repeat the license,
so the source of that grant is the pinned, author-linked dataset card. This
does not admit X-Capture for general physical reasoning or broad product-quality
claims. No media payload was downloaded or opened.

## What is promising

The authors describe 600 real objects, six manually aligned capture points per
object, RGB-D, impact audio, tactile readings, and 3D data. Their paper defines
cross-sensory retrieval as matching a reading to the same object among 100
objects. Object directory IDs provide the target for this specific retrieval
task; absent descriptions prevent semantic object/material-decision claims,
but do not prevent reproducing the identity-matching task. Use one fixed
object-disjoint train/validation/evaluation partition, with every capture point
from an object in the same partition. Compare audio-only, image-only,
tactile-only, and prespecified joint input with matched choice sets and
input-permutation controls. The paper's out-of-the-box ImageBind average
top-5 retrieval was 6.9% against 5% chance; after training on their data, the
cross-sensory-loss configuration reached 47.2%. These are paper-reported
retrieval results, not results for this repository or a 95–98% product gate.

## Pinned metadata-only audit

- Dataset repository: `swistreich/XCapture`, revision
  `b4468bf999aac250f34ef1b02459d902492a068f`.
- README blob: 2,667 bytes; Git blob OID
  `42202b1c3bf359802c65f80289152cb523f19338`.
- Archive: `XCapture_data.zip`, 17,924,194,875 bytes, LFS SHA-256
  `71855c2d9cdfb6d54ed379ff4059b6c370e6f7ef040830b0ac19063159c477c2`.
- A range request fetched only the 11,527,647-byte ZIP central directory; its
  SHA-256 is `3d8e546f1ca55e52a2b8e6c72aab99d14815e9da631589cb4d144f1179790f17`.
  It contained 94,973 records and 600 numeric object directories. The archive
  has no root-level CSV, JSON, or text label/split file. The index contains
  3,599 point directories rather than 3,600; object `00132` lacks point `004`.
- Paired-file coverage from the index: RGB plus normalized audio has 3,599
  points across all 600 objects. Adding a 10N tactile image leaves 3,592
  points, still represented across all 600 objects. A 20N tactile image is
  present for only 2,352 points across 399 objects, so it should not silently
  define the primary comparison.
- The indexed selected files for RGB, normalized audio, and 10N tactile total
  about 3.35 GB compressed and 6.41 GB uncompressed. The current C: free space
  observed during the screen was 12,349,992,960 bytes. Downloading and
  extracting the selected payload together would leave little working margin;
  a future run would need streaming/per-object feature extraction or a fresh
  capacity check. The full archive cannot fit in current free space.

## Frozen split and remaining gates

The source split is now frozen in
[`xcapture-object-split-seed17.json`](../../manifests/pretrained_reuse/xcapture-object-split-seed17.json):
400 train / 100 validation / 100 evaluation objects, seed 17, ranked by a
documented SHA-256 rule. All six capture points for an object stay in its one
partition. The evaluation partition is represented by IDs only; its media and
features remain unopened. The split builder and serialization are covered by
four focused CPU tests.

## Bounded media inspection

On 2026-11-28, a fixed SHA-ranked sample selected 10 complete points from
train and 10 from validation (20 distinct object IDs; one point per object).
The pinned archive central-directory index was read by HTTP range, then only
the selected RGB, normalized audio, 10N tactile image, and point-cloud mask
files were fetched. All 80 files passed recorded size and CRC checks. Their
sample manifest SHA-256 is
`17bacf80527a1aa0e1496eb330637d7f81b039ca68ffa28ccfabf9b33b240c0f`; the
masked-crop report SHA-256 is
`6c37406b3e8011bab5fd564470b5b8d312d4016966906a1f2683f723d4453c84`.
Evaluation object media and labels were not opened.

The paper describes generating the point clouds by applying a SAM mask queried
around the centered target point. In the 20 viewed RGB/mask pairs, coverage
ranged from 4.60% to 59.78% (mean 18.98%). The mask suppresses much of the
surrounding scene, but the manually reviewed crops still show readable package
text/barcodes or brand marks in at least 5/20 examples; some crops also retain
small disconnected mask fragments. These are visual-screen counts, not an
automated OCR measurement or an estimate of corpus-wide prevalence. Text on a
composite/labeled target is a real visual cue, not automatically leakage; it
must be stress-tested with a frozen text-occlusion control. A single point per
object and the object-disjoint split prevent exact object IDs from crossing
partitions.

The crop routine is implemented in `src/tiny_omni_decision/xcapture_preprocess.py`.
It requires aligned binary masks, zeros outside the mask, adds 8% padding to
the mask bounding box, and emits measurement and hash receipts. Five synthetic
CPU tests pass. This is input sanitation, not proof that all background or
lexical shortcuts are absent.

Artifacts are external to Git under
`C:\CodexArtifacts\pretrained-reuse-r2-local\`:

- Sample files: `xcapture-sample-audit-20261128-attempt-04/`
- Masked crops/contact sheet: `xcapture-masked-crops-20261128-attempt-02/`

No feature extraction, fitting, checkpoint load, or GPU work was performed.

## Remaining gates before model fitting

1. Implement a narrow object-identity retrieval task. Do not invent material
   labels or treat numeric IDs as natural-language descriptions. Use fixed
   candidate sets and compare unimodal, joint, shuffled-input, and text-occlusion
   controls.
2. Preserve the missing capture point (`00132/004`) as missing; do not impute
   or silently duplicate it.
3. Keep product readiness and the 95–98% multi-task objective open; this source
   can support only an object-centric cross-sensory retrieval result.

## Sources and access boundary

- Official project page: <https://xcapture.github.io/> (links to the author
  dataset card and states approximately 50 GB uncompressed).
- Pinned Hub card: <https://huggingface.co/datasets/swistreich/XCapture/blob/b4468bf999aac250f34ef1b02459d902492a068f/README.md>
  (declares MIT; says there are no train/validation/test splits).
- ICCV paper: <https://openaccess.thecvf.com/content/ICCV2025/papers/Clarke_X-Capture_An_Open-Source_Portable_Device_for_Multi-Sensory_Learning_ICCV_2025_paper.pdf>
  (object split and task details cited above).

Repository/card metadata, the paper, archive index, object IDs, and only the
20 train/validation sample points described above were accessed. Evaluation
media remains unopened; no checkpoint or GPU was used and no data was added to
a training corpus. X-Capture passes the dataset-license policy for this narrow
task. Feature extraction and measured cross-sensory evaluation remain open.
