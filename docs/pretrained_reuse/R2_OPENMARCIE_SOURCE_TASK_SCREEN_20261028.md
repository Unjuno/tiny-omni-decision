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

## 2026-10-29 metadata and split follow-up

Public Kaggle dataset pages were fetched as small metadata documents only; no
Kaggle API request, subscription acceptance, download request, archive, or
media payload was used. Their JSON-LD now exposes these versioned declarations:

| Part | Kaggle dataset ID | Version | Declared archive size | Displayed license | Subscription required |
|---|---:|---:|---:|---|---|
| [OpenMarcieA_Main](https://www.kaggle.com/datasets/4942046dbb2608f5a0c3e76c9bd53c6b9b2784d738da7db37f7739b89c2c7f95) | 7391133 | 14 | 139,274,818,467 bytes | CC BY-SA 4.0 | yes |
| [OpenMarcieB1](https://www.kaggle.com/datasets/9707fa8bcc1441bc7794afd43c64af8462487db326448e24a015740e218af2cb) | 7387960 | 15 | 131,915,306,620 bytes | CC BY-SA 4.0 | yes |
| [OpenMarcieB2](https://www.kaggle.com/datasets/106440bd5392b49fc4b732a8205f768af76db808cd70a94f4aa1b7ded115eb21) | 7389510 | 14 | 171,538,118,613 bytes | CC BY-SA 4.0 | yes |
| [OpenMarcieB3](https://www.kaggle.com/datasets/4a3e662eeea0e2fbeec209f9c617da36aa057186221b017ec39564ba74d781bf) | 7389984 | 11 | 148,562,565,029 bytes | CC BY-SA 4.0 | yes |

The combined declared archive size is 591,290,808,729 bytes (550.68 GiB). The
local C: volume had only about 0.38 GiB free during this check, so downloading
these archives is not feasible in the current environment. The license value
is the dataset host's displayed metadata; it is not treated as confirmation of
participant consent, third-party media rights, model-derivative obligations,
or compatibility with the project's product policy. Keep the source
`REVIEW_NOT_ADMITTED` pending those checks and a feasible storage plan.

The CVPR paper describes a 12-class HAR experiment with subject-disjoint
splits. This is useful evidence for that reported experiment, but the paper
does not provide a pinned participant-to-split manifest in the inspected
materials, and this statement does not establish grouped splits for the other
benchmarks or both experimental settings. The paper's reported acoustic track
is instrumental; do not interpret it as a speech/audio-understanding dataset.
The paper also reports I+V as stronger than adding audio for its HAR fusion
results, so OpenMarcie is not evidence that all modalities are jointly
necessary for our product task.

The split and modality statements refer to the [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Bello_OpenMarcie_Dataset_for_Multimodal_Action_Recognition_in_Industrial_Environments_CVPR_2026_paper.pdf)
and its [official supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Bello_OpenMarcie_Dataset_for_CVPR_2026_supplemental.pdf).

The [official code repository](https://github.com/HymalaiDFKI/OpenMarcie) was
rechecked at commit `fd72cf1ff11961520aafebc1f52587fd5bc2e74f`, tree
`a73e02927d5e1cbfad17b326e59e761b7cceeedd` (22 blobs, complete recursive
tree). Its `LICENSE` is Apache-2.0, which applies to code. The pinned tree has
no path containing `split`, `partition`, or `fold`, and the 5,151-byte code
README has no split/participant description. Thus the paper's split statement
cannot be reconstructed from this pinned repository snapshot alone.

The three repeated small-page retrievals for OpenMarcieA_Main were 12,295
bytes each and took 0.57, 0.31, and 0.38 seconds (mean 0.42 seconds). This
measures only public metadata-page response time, not dataset download
throughput; no large transfer ETA is inferred from it.
