# FZI-AURA source and access screen — 2026-10-28

## Decision

**Status: HOLD — do not train on, evaluate with, or redistribute the locally
retrieved sample unless dataset access and terms are resolved through the
provider's intended flow.** This report records a source/task screen and an
access-control discrepancy, not an admitted R2 task or a model result.

The FZI-AURA SDK README says to accept the dataset terms on Hugging Face and
sign in before downloading. The pinned dataset card declares CC BY-SA 4.0 and
describes an acknowledgement gate. The SDK download used an anonymous public
transport path without accepting terms or signing in. The downloaded sample
must therefore be treated as unapproved for project use. It remains in its
external artifact directory for provenance and is not included in Git. No
further reads of scene payloads, feature extraction, training, or evaluation
are authorized by this screen. Resolve access with the provider / expected HF
flow first; do not infer consent from successful HTTP access.

## Pinned sources

- Dataset: <https://huggingface.co/datasets/fzi-forschungszentrum-informatik/FZI-AURA>
- Dataset revision queried: `3404bd6b8fcd6eed53a0ec7610650a6393aabb49`
- SDK: <https://github.com/fzi-forschungszentrum-informatik/fzi-aura-sdk>
- SDK commit inspected: `94e02035a5edf11fc37bc67aba21cabbf97b2b3e`
- SDK source license: Apache-2.0.
- Dataset card license: CC BY-SA 4.0. The release also carries third-party
  notices for OSM-derived metadata (ODbL 1.0) and weather context. This project
  did not use those context fields.

The inspected card reports 2,473 scenes and publisher-declared scene-level
geographic split separation (1,979 train / 247 validation / 247 test). Its
metadata includes 2,481 entries because eight `consumer_excluded_scenes` are
identified as quality/synchronization faults. The currently visible preview
was 1,081 entries including those exclusions, or 1,073 usable scenes (859 /
107 / 107). These are source metadata counts, not independently audited
global media counts. The provider describes 10 Hz samples, 2 Hz box annotation
keyframes, eight cameras, 4–12 LiDARs, and 0–3 radar sensors; object IDs are
scene-local.

## Candidate task and limitations

A possible derived task is same-frame object association: given an annotated
RGB object crop and a set of same-class candidate 3D boxes/point-cloud crops,
score which candidate corresponds to the image object. This is a task proposal,
not an official benchmark or verified label protocol. It would require
scene-grouped splits and fixed image-only, sensor-only, paired, and modality
swap controls, with candidate order randomized and hard same-class distractors.
It does not include audio, so it cannot satisfy the project's four-modality
gate or R2-M natural AV gate by itself.

No task labels were constructed and no quality evaluation was run.

## Acquisition and storage evidence (quarantined; not project-use data)

The SDK was cloned to
`C:\CodexArtifacts\pretrained-reuse-r2\fzi-aura-sdk-audit` at the commit
above. One requested train scene (`2025-06-11-14-31-00|41`) selected its full
seven-scene storage block (six neighboring scenes) and three layers:
base keyframes, camera keyframes, and motion-compensated LiDAR keyframes.
The SDK completed with `--verify`, then deleted downloaded archives after
extracting. It reported one requested scene, six block neighbors, three
extracted chunks, and no split-boundary crossing for the selected block.

The external directory is
`C:\CodexArtifacts\pretrained-reuse-r2\fzi-aura-crossmodal-task-smoke-20261028-01`.
It contains 4,037 files totaling 3,970,119,382 bytes; the `scenes` subtree
accounts for 4,014 files / 3,967,136,134 bytes. The local `dataset.json` is a
one-scene subset, not the full corpus. Retained metadata hashes:

| File | SHA-256 |
|---|---|
| `available_data.json` | `4CE662B9CF177F300CFE9D5FED875FAE5FFFE9E8AB0DE5485CBEF0F3B822DC68` |
| `dataset.json` | `C40D36889383A0C80CAE449181A9AD45C3B9ECC4D24834E35F9D42E7ADE61A3D` |
| `LICENSE` | `23EE78C8BAE49CF08EA2F0C84945C66B987EBE4520881FB51B3DAD4FB43D07C2` |
| `THIRD_PARTY_NOTICES.MD` | `ECDAD55BE986C02C6974423A8B80C82BEC80DAB9730517B53046E07802D1AE7F` |

SDK `--verify` validated the source chunks before deleting archives:

| Layer | Bytes | SHA-256 |
|---|---:|---|
| base | 20,418,560 | `f8f536850bb560969dbe6d37f983f4b02f08ac51e64c1053fa35db6d748c1ced` |
| camera | 1,367,398,400 | `ac3eeff686abe71742ff94b87ed5824ed45fd4d4454edd705ccedf835e98b8bf` |
| motion-compensated LiDAR | 1,222,523,272 | `a0a3a34ea4aea740ec8e0d30c9bf50c987d32c7f0e14f76f1f6838eb8f440502` |

Before recognizing the acceptance requirement, a one-scene SDK schema/read
smoke was performed. It reported 200 frames, 40 annotated keyframes, a
1200×1920 RGB front-camera frame, 76,554 points in one motion-compensated
LiDAR frame, and 15 boxes/projections (11 visibly projected). This only
establishes that the SDK can parse the sample technically. Because access had
not been accepted, it is retained here as an incident/provenance record and
must not be treated as an approved data-validation result. Do not repeat it.

The SDK preview dry-run estimated the full train camera + motion-compensated
LiDAR keyframe payload at 462,837,583,384 compressed bytes (~462.84 GB); the
entire repository was listed at ~2.81 TB. The smallest listed validation block
was 2,665,034,032 bytes and the next was ~8.75 GB. At the last local check,
C: had ~7.07 GiB free, so no further download was attempted. These payload
sizes are not needed to establish the access hold.

Download wall time was approximately 236 seconds for ~2.61 GB of compressed
chunks, about 11.1 MB/s including setup and per-file delays. Three 10-second
file-size samples showed no change because the SDK was finalizing files
atomically; they are not three independent throughput measurements and should
not be averaged as such.

## Runtime/process constraints and next action

At 2026-10-28 21:29 JST the RTX 3080 Laptop GPU was at 99% utilization with
15,821 / 16,384 MiB allocated by Orbitune PID 22136. No Tiny Omni GPU work was
started and no process was stopped. The next safe action is to resolve dataset
access with the provider and, separately, continue CPU-only R1/source audits
that do not touch this quarantined payload. If the provider's access flow
cannot be completed, exclude FZI-AURA and select another source with clear
project authorization.
