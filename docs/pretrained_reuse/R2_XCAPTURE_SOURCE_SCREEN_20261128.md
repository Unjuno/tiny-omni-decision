# X-Capture multisensory source screen — 2026-11-28

## Disposition

**`REVIEW_NOT_ADMITTED` for Decision training or quality selection.** X-Capture
is a promising object-centric audio, image, and tactile research source, but
the official project page does not state a dataset license, the linked Hub card
is the only inspected place declaring MIT, and the archive exposes no
machine-readable object-description/target table or official split manifest.
No media or annotation payload was downloaded or opened. Do not treat the
research paper's retrieval results as product-task quality.

## What is promising

The authors describe 600 real objects, six manually aligned capture points per
object, RGB-D, impact audio, tactile readings, and 3D data. Their paper defines
cross-sensory retrieval as matching a reading to the same object among 100
objects. This is a plausible narrow sensor-fusion Decision probe: a query from
one sensor could select the matching candidate among supplied object choices,
with unimodal and joint controls. The paper's out-of-the-box ImageBind average
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

## Gates before any data fit

1. Obtain an authoritative dataset-use statement from the project authors or
   an explicit license file. The Hugging Face card labels the dataset MIT, but
   the official project page links to the archive without displaying that
   license; this screen does not resolve the scope of the grant.
2. Verify whether the archive actually contains the paper's brief object
   descriptions. The central directory showed no separate machine-readable
   label or split file. Without descriptions, the only directly derivable
   target is anonymous object identity, which is a retrieval benchmark rather
   than broad physical reasoning.
3. If admitted, preregister a task-specific object-identity choice protocol
   and split by object ID, never capture point. Keep every point from an object
   in one split; construct validation before fitting and reserve a one-shot
   object-disjoint test. Compare audio-only, image-only, tactile-only, and
   prespecified joint inputs, plus input-shuffle controls. Report calibration
   and group-level uncertainty in addition to accuracy.
4. Audit capture environments and image backgrounds before selecting any
   product-directed evaluation subset. No visual contents were opened in this
   metadata-only pass.

## Sources and access boundary

- Official project page: <https://xcapture.github.io/> (states approximately
  50 GB uncompressed; no dataset license is displayed there).
- Pinned Hub card: <https://huggingface.co/datasets/swistreich/XCapture/blob/b4468bf999aac250f34ef1b02459d902492a068f/README.md>
  (declares MIT; says there are no train/validation/test splits).
- ICCV paper: <https://openaccess.thecvf.com/content/ICCV2025/papers/Clarke_X-Capture_An_Open-Source_Portable_Device_for_Multi-Sensory_Learning_ICCV_2025_paper.pdf>
  (object split and task details cited above).

Only repository/card metadata, the paper, archive size metadata, and ZIP central
directory ranges were accessed. No media, labels, checkpoint, or GPU was used;
no dataset was added to a corpus. This screen leaves X-Capture as a candidate
for follow-up, not an admitted source.
