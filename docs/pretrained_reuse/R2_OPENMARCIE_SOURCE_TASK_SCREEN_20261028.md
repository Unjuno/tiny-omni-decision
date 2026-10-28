# OpenMarcie multimodal source and task screen (2026-10-28)

## Decision

**Highest-priority candidate for follow-up rights and split verification; not
admitted for training yet.** The official project page describes a natural,
synchronized dataset with video, audio, wearable sensors, two assembly settings,
and 36 participants. It is substantially closer to R2-M than the currently
admitted synthetic or single-modality tasks. However, the official project page
and dataset README inspected here do not specify the data license or the
participant release terms. The GitHub repository's Apache-2.0 applies to its
code; it is not evidence that the separate human-recorded dataset is covered
by Apache-2.0. No dataset payload was downloaded, and no fitting or evaluation
was performed.

## Task fit and unresolved integrity questions

- The official project page describes two settings: bicycle assembly and
  disassembly (12 participants), and guided 3D-printer assembly (24 valid
  participants). It reports over 37 hours of synchronized egocentric and
  exocentric video, audio, wearable sensing, and other sensor channels.
- The stated benchmarks include human activity recognition, open-vocabulary
  captioning, and cross-modal alignment. These provide concrete task-design
  candidates, but the project goal requires a Decision task with supplied
  choices/probabilities and matched unimodal, metadata-only, and perturbation
  controls. The benchmark labels alone do not establish that every modality is
  needed for a particular decision.
- The official public materials inspected do not establish participant- or
  session-disjoint training/validation/test partitions, repeated-session
  relations, or immutable archive checksums. These remain **UNKNOWN**.
- The project page links both a password-protected single-link download and
  four Kaggle parts. No download was attempted. The Kaggle dataset's asserted
  data license and version identity were not confirmed from the reviewed
  official materials.

## Admission gates before any download or fit

1. Confirm explicit data-use rights covering model training, derivative model
   retention, and intended product/commercial use; code licensing is separate.
2. Pin the archive version and per-file checksums, and determine total archive
   and extracted size before reserving local storage.
3. Establish participant/session-level grouping across both experimental
   settings and identify any shared cameras, rooms, object instances, or
   procedural templates that could leak across splits.
4. Pre-register one task where an observation stream is expected to be
   necessary, then verify this using metadata-only, matched unimodal, and
   modality-swap controls before checkpoint selection.
5. Preserve any official test split; use only train and development groups for
   initial selection.

## Read-only sources inspected

- Official project page: <https://hymalaidfki.github.io/OpenMarcieCVPR/>
- Official code and dataset README: <https://github.com/HymalaiDFKI/OpenMarcie>
- Official CVPR paper: <https://openaccess.thecvf.com/content/CVPR2026/papers/Bello_OpenMarcie_Dataset_for_Multimodal_Action_Recognition_in_Industrial_Environments_CVPR_2026_paper.pdf>
- Linked Kaggle dataset ID:
  `4942046dbb2608f5a0c3e76c9bd53c6b9b2784d738da7db37f7739b89c2c7f95`.

The supplement PDF was not accessible to the current web reader, and the
Kaggle page was not readable from the browser. This screen does not infer
license or split terms from inaccessible materials.
