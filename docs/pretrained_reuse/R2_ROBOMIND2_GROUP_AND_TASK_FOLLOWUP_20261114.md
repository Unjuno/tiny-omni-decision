# RoboMIND 2.0 Tienkung simulation group/task follow-up — 2026-11-14

## Decision

Keep `122-rotate_pot_handle/success_episodes` at `REVIEW_NOT_ADMITTED` for
R2-M-2. Do not fetch its HDF5/media payloads or fit a Decision model from this
candidate. This is a metadata-only follow-up; no data, checkpoint, process, or
GPU state was changed.

## New official metadata checked

- The [official ModelScope task directory](https://www.modelscope.cn/datasets/X-Humanoid/RoboMIND2.0-Tienkung-sim/tree/master/data/tienkung_sim/122-rotate_pot_handle/success_episodes)
  labels the Tienkung-sim dataset Apache License 2.0 and reports a total
  repository size of 194.21 GB. This improves the published license evidence,
  but does not itself document provenance or rights for every embedded
  simulator asset and component.
- The [official RoboMIND-Sim README](https://github.com/Open-X-Humanoid/RoboMIND-Sim)
  says each task has 50 scenarios and that one is selected randomly for each
  simulation run. It does not expose a mapping from those scenario identities
  to the 303 HDF5 success-episode pointers in the pinned dataset task path,
  nor a train/validation/audit assignment for either collection.
- The same README describes the HDF5 data as collected in Isaac Sim 4.5 and
  provides a Linux Isaac Sim workflow. Its benchmark instructions do not
  provide a supplied-option Decision label contract; policy inference is left
  for the user to implement.

## Gate reconciliation

| R2-M-2 requirement | Current evidence | Result |
| --- | --- | --- |
| Rights for selected payload/components | The dataset card says Apache-2.0; component-level provenance and scope for embedded simulator assets are not independently specified | Unresolved; do not ingest |
| Independent groups and split | 303 timestamped episode pointers span six dates; README separately mentions 50 scenarios, with no episode-to-scenario/session map or split manifest | Fail closed; neither 303 files nor six dates establish independent examples |
| Minimum group support | ROADMAP's provisional planning floor is 200 independent groups when available; 50 scenarios would be below it if those are the independent units | Not met or not demonstrated |
| Supplied-option Decision labels | The source provides successful demonstrations; no fixed typed-option target or candidate set is published for this task | Unresolved |
| Both-input necessity | No matched image/video-only, proprioception-only, and joint controls are defined for this Decision task | Unresolved |
| Local execution | The full subrepository is 194.21 GB; task-only payload size is unknown. The published simulation workflow is Linux/Isaac Sim-specific | Not established; no download justified |

The dataset remains a plausible robotics source, but this task cannot yet
support a clean paired video+proprioception Decision experiment. Do not infer
the product's multi-input capability from the existence of synchronized
streams alone.

## Reopen conditions

Reconsider only if a pinned publisher artifact provides (1) an episode-to-
independent-scenario/session mapping and an explicit split with adequate
independent groups, (2) component-level rights evidence for the selected
payload, and (3) a predeclared typed-option task whose target demonstrably
requires both visual and proprioceptive evidence. The experiment must then
include matched best-unimodal, missing, shuffled, and time-offset controls.

## Evidence boundary

The ModelScope listing and RoboMIND-Sim README are official primary sources;
they establish only the statements above. No LFS object, episode payload,
label, benchmark run, or model output was accessed. The disposition from the
2026-11-12 screen is unchanged.
