# R2-M follow-up screen: recent multimodal action datasets — 2026-10-31

## Decision

**None admitted for product-directed data use.** Three recent or prominent synchronized human-activity datasets expose useful sensor/video task designs, but their published data terms restrict use to non-commercial research or require a non-commercial Data Use Agreement (DUA). No access request or agreement was submitted; no dataset payload, sample media, or labels were accessed. This is a metadata-only screen.

## Official-source findings

| Source | Dataset/task evidence | Data-use and practical gate | Disposition |
|---|---|---|---|
| [ActionSense](https://action-sense.csail.mit.edu/) | Synchronized wearable and environment sensors, RGB/depth video, audio, and kitchen activity labels; official site reports 10 participants and about 778 minutes in its published dataset card. Candidate: decide activity/object-state from video plus motion/tactile/audio signals. | The dataset site states CC BY-NC-SA 4.0. Its paper describes very large per-subject camera payloads; only nine subjects have complete wearable streams in the cited card. A new product-directed fit is not admitted under the published non-commercial terms. | `REVIEW_NOT_ADMITTED`; no request or payload access. |
| [OctoNet](https://github.com/aiot-lab/OctoNet) | The official project describes 12 synchronized modalities and 62 activities across 41 participants, including RGB, thermal, depth, radar, skeleton and IMU. Candidate: temporal activity/state decision with video and sensor controls. | The official NeurIPS paper's supplementary states the code and dataset are CC BY-NC 4.0; identifiable RGB has an additional request/permission route. This conflicts with product-directed use and does not grant commercial rights. | `REVIEW_NOT_ADMITTED`; do not request identifiable RGB or download. |
| [CUHK-X](https://github.com/openaiotlab/CUHK-X) | The official README describes seven aligned streams, 30 participants, 64,267 annotated samples, and action selection / next-action reasoning tasks. The challenge uses cross-subject splits and excludes RGB. | The README states dataset files are available only for non-commercial research under a DUA and are not redistributable; only derived annotations/splits are CC BY 4.0. Code's MIT license does not license the raw data. | `REVIEW_NOT_ADMITTED`; do not accept the DUA or download data. |

## Task-quality limits

These sources are worth remembering for future research-only work, but their existence does not establish a valid both-input-required Decision task. Activity labels may be predictable from a single stream; the specific task would still need option/metadata priors, matched unimodal baselines, and shuffled/missing/stale/contradictory controls. ActionSense and OctoNet also have relatively few independent people, so repeated clips cannot be counted as independent participants. CUHK-X's cross-subject challenge split is promising metadata, but access restrictions remain decisive for this product-directed objective.

## Next gate

Continue with already-admitted sources and frozen Path A evidence. Reconsider any source only if the rights holder publishes terms compatible with the intended product use, or the user explicitly authorizes an appropriate non-commercial research-only scope. Before any fit, pin the exact data revision, confirm source/participant/scene splits, audit label and option leakage, and freeze matched modality controls. No model, data, checkpoint, prior result, or sealed audit was modified or opened.
