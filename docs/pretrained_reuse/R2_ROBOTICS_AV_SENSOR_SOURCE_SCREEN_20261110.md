# R2 robotics and synchronized AV source screen — 2026-11-10

## Decision

No newly screened source is admitted for product-directed Path A fitting or
quality evaluation. This was an official-source metadata/license screen only;
no dataset payload, annotation archive, model, or GPU was accessed.

## Candidates

| Candidate | Official evidence | Gate result |
|---|---|---|
| ClevrSkills | Qualcomm reports about 300,000 simulated robot episodes over 33 tasks, with video, actions, text labels, success labels, and seeded episode directories. Its current Dataset License Agreement grants internal use solely for non-profit research and explicitly excludes commercial use, including processing for results used in a third-party product/service. | Exclude for this product-directed path despite its scale and useful modalities. The primary page links a binding research-use agreement; do not accept it or download the data. |
| CASTLE | The official project page reports 10 participants, synchronized audio/video and IMU/GPS/biometric streams, over 600 hours, and an 8.22 TB total. Its license is CC BY-NC-SA 4.0 and its terms expressly prohibit commercial use. | Exclude: noncommercial restriction and only 10 participants, below the 200 independent-group floor. |
| MMCSG | Meta describes synchronized multi-channel audio, video, accelerometer and gyroscope from two-sided conversations, with human annotations and participant consent. The publisher says access/use is under its Data License and intended for research purposes. | Keep `REVIEW_NOT_ADMITTED`: the exact Data License terms were not resolved, so product-directed rights are unverified. No access terms were accepted. |
| VITRA teleoperation / VITRA-1M | Pinned repository commit `b35517202b39d32a753fdd42014b2cc3c41fab58` documents one-to-one synchronized episode HDF5 state/action and MP4 video for teleoperation, but the teleoperation data page states no dataset license. The repo's MIT license is for software. Its separate VITRA-1M metadata refers to Ego4D, Epic-Kitchens, EgoExo4D and Something-Something V2 videos, and says source-license restrictions prevent redistribution; metadata alone is about 100 GB. | Keep `REVIEW_NOT_ADMITTED`: no explicit rights grant for the teleoperation data was found; the VITRA-1M component inherits source access/rights constraints. No metadata or media was downloaded. This screen does not establish adequate independent-group counts or a frozen both-input task. |

## Primary sources

- [Qualcomm ClevrSkills page](https://www.qualcomm.com/developer/software/clevrskills-dataset) and its linked [Research Use Data License Agreement](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/ClevrSkills-Dataset-Research-License-6.11.2024.pdf).
- [CASTLE official project page and terms](https://castle-dataset.github.io/).
- [Meta MMCSG page](https://ai.meta.com/datasets/mmcsg-dataset/).
- [VITRA pinned teleoperation data description](https://github.com/microsoft/VITRA/blob/b35517202b39d32a753fdd42014b2cc3c41fab58/data/teleoperate_data.md), [pinned VITRA-1M data instructions](https://github.com/microsoft/VITRA/blob/b35517202b39d32a753fdd42014b2cc3c41fab58/data/data.md), and [pinned repository software license](https://github.com/microsoft/VITRA/blob/b35517202b39d32a753fdd42014b2cc3c41fab58/LICENSE.txt).

## Next gate

Continue source admission only where the exact intended data has an applicable
rights grant, at least 1,000 examples and 200 independent groups, a defensible
group split, and a predeclared decision task that requires both proposed input
streams. The current RoboMNIST AV-match validation is already exposed and its
NLL/calibration fail the joint-input gate; it must not be retrained or
recalibrated against that validation. If no fresh source passes, keep the
quality gate open and progress runtime/package engineering without claiming
new multimodal quality.
