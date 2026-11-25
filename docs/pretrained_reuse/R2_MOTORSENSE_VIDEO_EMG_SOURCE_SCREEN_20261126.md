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
  and Hub API/card metadata and file listing at the pinned dataset revision.

## Task fit

The source pairs egocentric video with bimanual forearm EMG and IMU at 500 Hz,
with frame timestamps on the sensor clock. The publisher describes action
segments with verb/noun/acting-hand labels and hand-contact intervals. A
possible Decision task is to select the active action or next action from
time-bounded video and sensor windows, comparing video-only, sensor-only, and
joint readouts plus shifted, shuffled, and missing-stream controls. This would
be a limited video-plus-sensor task, **not** an audio task or general Omni
benchmark. Audio is mentioned in high-level project text but is not present in
the reviewed released file layout; count audio as unavailable unless the
publisher documents and exposes it.

The source does not itself establish supplied-option choice sets, uniqueness
of a correct next action, or that each label requires both video and sensor.
The action annotations are retrieval of observed human actions, not proof of
goal completion, safety, or task success. A frozen task contract would need to
define candidate construction and label semantics before any fit.

## Blocking integrity and rights issues

1. **Data rights:** the dataset Hub declares `other`, not an explicit license.
   The official code repo's MIT license is code-only. Commercial/model-training
   rights and derivative-model retention are therefore unknown.
2. **Privacy/consent:** the publisher says frames are not face-blurred. The
   reviewed materials do not establish participant consent and publicity/model
   training scope. Do not infer permission from public download access.
3. **Group independence:** action annotations cover 65 train and 20 validation
   recordings, but the official `docs/DATA.md` says Aria validation
   participants also occur in training. Smartphone recordings are split by
   recording but their participants are unattributed. Thus the available split
   does not establish participant-disjoint validation.
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
6. **Count discrepancy:** the project page reports 10,418 action segments,
   while the pinned code README / data documentation and Hub action split
   describe 5,209 annotated segments (3,584 train + 1,625 validation). Resolve
   this with the publisher before relying on sample counts.

## Next gate

Reconsider only if the publisher provides explicit dataset and human-subject
terms compatible with the intended use; a new group-disjoint split can be
verified from participant/session identity; counts and label semantics are
reconciled; and the selected representation is demonstrably not pretrained on
the validation groups. Then pre-register a small, source-disjoint task and
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
- [Pinned Hub API record](https://huggingface.co/api/datasets/edessa/motorsense/revision/3fec7844694f692a11d2818735aec535a001b7f7)
