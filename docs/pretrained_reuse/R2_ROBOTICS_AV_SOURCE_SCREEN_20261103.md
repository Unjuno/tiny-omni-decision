# Robotics AV / sensor source screen — 2026-11-03

## Disposition

Three publisher-facing source descriptions were screened without downloading
datasets, opening media, reading annotations, or loading models. None is
admitted for the current R2 task. No GPU work or training occurred.

The R2 gate is a synchronized, rights-cleared task with labels that establish
both-input necessity and enough independent groups for a meaningful grouped
evaluation. Robot action trajectories alone do not supply such a decision
target. Dataset scale does not compensate for missing modalities, unclear data
rights, insufficient independent identities, or a task mismatch.

## Candidate findings

| Source | Publisher evidence | R2 disposition |
|---|---|---|
| BridgeData V2 | The project describes 60,096 trajectories, 24 environments and 13 skills, with language instructions and RGB-D / robot observations. The project page states CC BY 4.0. The published download page lists 100 GB raw images or a 34 GB MP4 subset. | Exclude from the synchronized AV gate: no audio stream is documented. The 24 environments also do not establish the required number of independent scene groups. Storage exceeds current free space. Could be reconsidered only for a separately specified video-plus-state task with valid outcome labels and a frozen group split. |
| RH20T | The project describes over 110,000 synchronized manipulation sequences with RGB, audio, force/torque and robot state. Its publisher warns recordings may include volunteer faces and voices and requests use solely for model training. It reports a 40 TB original corpus; even the listed 320×180 RGB Cfg3 archive is 4.4 GB. RH20T-C (scenes 0001–0005) is CC BY-SA 4.0; RH20T-NC (scenes 0006–0010) is CC BY-NC 4.0. | Do not use. The accessible license mix, share-alike/non-commercial terms, identifiable people, training-only restriction, storage pressure, and scene-level split constraints do not fit this product evaluation gate. No personal data or license acceptance was attempted. |
| RoboSet | The publisher describes 30,050 trajectories, including 9,500 teleoperated trajectories, four camera views, 12 skills and 38 tasks; the page says rollouts use varied/rearranged scenes. The page labels the work MIT, but does not clearly establish that the data itself is licensed under MIT. The TFDS catalog estimates 179.42 GiB. | Exclude from the current AV/sensor gate: the source description does not establish synchronized audio, decision-outcome labels, or rights for the data payload. The full dataset exceeds local capacity. |

## Storage and handling

At screen time C: had 7.41 GiB free. No dataset was downloaded and no existing
artifact was removed. RH20T's 4.4 GB Cfg3 RGB archive would leave little room
for extraction, derived features, and run outputs, and its associated license
and evaluation-use restrictions remain disqualifying independently of size.

## Decision

Do not start another joint fit from these sources. Continue only after a
source/task passes rights, modality synchronization, both-input necessity,
independent-group, target-definition, and local-storage gates. This screen
does not establish a representation or model-capacity limit and does not close
R2.

## Sources

- [Official BridgeData V2 project page](https://rail-berkeley.github.io/bridgedata/)
- [BridgeData V2 download and contribution page](https://sites.google.com/view/bridgedata/using-and-contributing-data)
- [Official RH20T project page: modalities and download sizes](https://rh20t.github.io/)
- [Official RoboSet teleoperation dataset page](https://robopen.github.io/roboset/teleoperation.html)
- [TensorFlow Datasets RoboSet catalog](https://www.tensorflow.org/datasets/catalog/robo_set)
