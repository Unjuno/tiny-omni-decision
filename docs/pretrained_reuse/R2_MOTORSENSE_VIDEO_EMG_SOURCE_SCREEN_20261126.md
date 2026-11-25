# MotorSense video–EMG source screen — 2026-11-26

## Disposition

**`REVIEW_NOT_ADMITTED`; metadata-only.** MotorSense is a strong scientific
lead for a narrowly scoped temporal video-plus-sensor Decision probe, but the
current evidence does not admit its human-recorded data for product-directed
use. Do not download annotations or media, load its checkpoints, or fit a
readout until the rights, privacy, split, and representation-exposure gates
below are resolved. No annotation rows, media, checkpoint, or sealed-audit
content was accessed for this screen.

## Pinned source identity and what was inspected

- Official project page: <https://www.cs.umd.edu/~edessale/motorsense/>.
- Official code repository: <https://github.com/edess442/MotorSense>, observed
  `HEAD` `536180c611834ae81a8dea14716896284057da77` on 2026-11-26. Its `MIT`
  license applies to code; it does not grant rights to the separate dataset.
- Official Hugging Face dataset repository:
  <https://huggingface.co/datasets/edessa/motorsense>, revision
  `3fec7844694f692a11d2818735aec535a001b7f7`, API `cardData.license` is
  `other`, public and ungated. This is not an explicit reusable data license.
- The recursive Hub file listing reports 1,864 entries. Metadata and action
  annotation files are small (e.g. `metadata/recordings.csv` 18,012 bytes,
  `annotations/actions/MotorSense_train.csv` 527,768 bytes and
  `MotorSense_validation.csv` 239,039 bytes); contact annotations are
  27,587,504 bytes. These are metadata sizes only; their contents were not
  fetched or opened. The complete release is reported as approximately 0.9 TB
  (885 GB on the Hub page).
- Primary documents reviewed: the official project page, pinned repository
  README / `docs/DATA.md` / code `LICENSE` at the repository revision above,
  the v1 arXiv paper and appendix (`arXiv:2610.41127v1`), and Hub API/card
  metadata, README and file listing at the pinned dataset revision.

## Task fit

The source pairs egocentric video with bimanual forearm EMG and IMU at 500 Hz,
with frame timestamps on the sensor clock. The publisher describes action
segments with verb/noun/acting-hand labels and hand-contact intervals. A
possible Decision task is to select the active action or next action from
time-bounded video and sensor windows, comparing video-only, sensor-only, and
joint readouts plus shifted, shuffled, and missing-stream controls. This would
be a limited video-plus-sensor task, **not** an audio task or general Omni
benchmark. The paper says audio is included in the release, but the pinned Hub
file listing exposes no audio-named paths; count audio as unavailable until
the publisher provides a pinned file-level inventory.

The source does not itself establish supplied-option choice sets, uniqueness
of a correct next action, or that each label requires both video and sensor.
The action annotations are retrieval of observed human actions, not proof of
goal completion, safety, or task success. A frozen task contract would need to
define candidate construction and label semantics before any fit.

## Blocking integrity and rights issues

1. **Data rights:** the dataset Hub declares `other`, not an explicit license.
   The paper says the data is released under research-use terms in the
   appendix / data package, but those terms are not exposed in the pinned Hub
   README or file tree. The official code repo's MIT license is code-only.
   Product/commercial use and derivative-model retention are therefore
   unresolved.
2. **Privacy/consent:** the paper reports IRB approval and written consent for
   research use, public dataset release and reproduction of frames and clips.
   It says recordings were screened and de-identified with manual review and
   EgoBlur, with no bystanders, incidental speech removed and location/GPS
   omitted. The pinned Hub card, however, says frames are not face-blurred.
   This conflicts with the paper's EgoBlur statement; clarify the exact
   released-media revision before use. Consent is documented, but the media
   discrepancy and terms remain open.
3. **Group independence:** action annotations cover 65 train and 20 validation
   recordings, but the official `docs/DATA.md` says Aria validation
   participants also occur in training. Smartphone recordings are split by
   recording but their participants are unattributed. Thus the available split
   does not establish participant-disjoint action validation. The paper's
   held-out-participant motor-code prediction is a different task; the authors
   explicitly note downstream action recognition was not evaluated with
   subject-disjoint splits.
4. **Pretraining exposure:** the released video–EMG/IMU contrastive checkpoints
   are pretrained on all 370 recordings, including action/contact validation
   recordings. They are not valid frozen representations for a claim of
   untouched validation generalization on this source. Reusing them would
   contaminate the proposed validation comparison even though labels were not
   used in pretraining.
5. **Scale and modality mismatch:** the complete data is approximately 885 GB,
   infeasible for the current C:-only local workflow. A metadata-only small
   download is not a substitute for rights admission. Audio is not established
   in the distributed schema, so this source cannot fill the audio gate.
6. **Count convention:** the project page reports 10,418 action units and 9,400
   contact units, exactly twice the paper/Hub counts of 5,209 action segments
   and 4,700 contact intervals. This is consistent with per-hand expansion,
   but the counting unit must be frozen before building examples; use the
   pinned annotation schema as the source of row identity.
7. **Audio exposure:** the paper's checklist says the release includes audio,
   while the pinned Hub README and file listing expose no audio-named path.
   Aria capture included audio, but the exact released audio payload is not
   established from the Hub tree. Do not count this as an available audio
   dataset without a pinned file-level inventory.

## Source-screen correction — 2026-11-26

The initial metadata-only screen did not include the paper appendix and
incorrectly described consent as unestablished, treated the Hub's
"not face-blurred" note as settled, and called the action-count difference a
discrepancy. After reading arXiv v1 and the pinned Hub README, the record above
was corrected: the paper reports consent/IRB/de-identification; the Hub and
paper conflict on face blurring; and the count difference is consistent with
per-hand expansion. The Hub `other` license and the exact research-use terms
remain unresolved, so the admission decision remains unchanged.

## Next gate

Reconsider only if the publisher provides explicit dataset and human-subject
terms compatible with the intended use; the Hub/paper privacy discrepancy is
resolved; a new group-disjoint split can be verified from participant/session
identity; row-counting and label semantics are frozen; and the selected
representation is demonstrably not pretrained on the validation groups. Then
pre-register a small, source-disjoint task and
metadata-only / video-only / sensor-only / joint controls before accessing any
validation labels. Do not use the released MotorSense-pretrained checkpoint
for this validation study.

This source screen is not an R2 admission, quality result, or reason to start
training. No GPU, cloud compute, purchase, or dataset payload was used.

## Primary sources

- [Official MotorSense project page](https://www.cs.umd.edu/~edessale/motorsense/)
- [Pinned code repository](https://github.com/edess442/MotorSense/tree/536180c611834ae81a8dea14716896284057da77)
- [Pinned data documentation](https://github.com/edess442/MotorSense/blob/536180c611834ae81a8dea14716896284057da77/docs/DATA.md)
- [Pinned code license](https://github.com/edess442/MotorSense/blob/536180c611834ae81a8dea14716896284057da77/LICENSE)
- [Pinned dataset card](https://huggingface.co/datasets/edessa/motorsense/tree/3fec7844694f692a11d2818735aec535a001b7f7)
- [Pinned v1 paper and appendix](https://arxiv.org/html/2610.41127v1)
- [Hub API record](https://huggingface.co/api/datasets/edessa/motorsense)
