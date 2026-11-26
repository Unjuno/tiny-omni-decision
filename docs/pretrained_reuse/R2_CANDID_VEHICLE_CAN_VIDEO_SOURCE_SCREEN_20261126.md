# CANdid vehicle CAN/video source screen — 2026-11-26

## Disposition

**`REVIEW_NOT_ADMITTED`; metadata and paper review only.** CANdid is a potentially
useful source for a narrow vehicle-state investigation, but it does not currently
provide a defensible product-quality evaluation for a general Omni Decision model.
No dataset payload, video, GPS trace, or CAN recording was downloaded or opened;
no model was run.

## Pinned source facts

- Dataset record: University of Adelaide, CANdid, edition 2,
  DOI [`10.25909/29068553.v2`](https://doi.org/10.25909/29068553.v2).
- The [Research Data Australia record](https://researchdata.edu.au/candid/3887827)
  reports CC BY 4.0 and describes CAN traffic, GPS, timed action labels, driver
  video, and instrument-cluster video. The provider's Figshare landing page was
  not accessible in this metadata-only check, so the exact provider-side license
  text and any supplementary conditions remain to be verified before acquisition.
- The [VehicleSec 2025 paper](https://www.usenix.org/system/files/vehiclesec25-howson.pdf)
  reports ten vehicles, with one roughly 25–65 minute public-road drive per listed
  vehicle and separate stationary, scripted control captures. It describes video
  and CAN/GPS capture on a shared device clock and timestamped control annotations.
- The paper reports post-processing distortion of identifiable faces in video.
  It does not establish from the reviewed passages that all route-identifying
  GPS data or other identifying details have been removed. GPS can reconstruct
  routes, so privacy handling needs a specific review before media use.

## Data and task quality assessment

The source is real-world and has useful cross-sensor synchronization, but its
independent support is small for generalization: only ten listed vehicle
captures are described. Repeated CAN messages, video frames, and timestamped
actions within a capture are not independent examples. Splitting those rows or
windows across folds would leak capture/vehicle identity. A vehicle-held-out
evaluation would therefore have at most ten top-level vehicle groups, far below
the project's provisional 200-group quality-evaluation floor.

The clearest paper-supported labels are controlled actions (indicator, lights,
brakes, gears, steering, engine) repeated in stationary captures, plus general
driving captures with GPS. These support a potential narrow state-recognition or
signal-identification diagnostic. The reviewed materials do not define a
user-supplied-option Decision task for which video and CAN are both necessary.
For example, a control-state label may be directly visible in the instrument
cluster or directly encoded by CAN, making a joint score insufficient evidence
of complementary multimodal reasoning. Any future task would need a frozen
target definition and CAN-only, video-only, joint, shuffled/missing-input, and
capture/vehicle-grouped controls.

The recordings are not licensed or authorized here for training merely because
an aggregator lists CC BY 4.0. Before any payload access, verify the provider's
current terms, commercial and derivative-model scope, and privacy constraints
for faces, vehicle interiors, and reconstructable GPS routes. Do not use this
source for product-directed fitting until that review and a group-aware task
protocol pass.

## Decision

Do not acquire or fit CANdid for the current R2 quality gate. It may be revisited
only as a clearly labeled exploratory diagnostic if provider terms are verified
and the project explicitly accepts its small number of vehicle groups; such a
pilot cannot establish broad vehicle generalization or close the multimodal
gate. Prioritize a larger rights-cleared source with independent recording,
scene, or participant groups and a task whose target demonstrably needs both
video and sensor input.

## Sources

- [University of Adelaide dataset record, edition 2](https://doi.org/10.25909/29068553.v2)
- [Research Data Australia metadata record](https://researchdata.edu.au/candid/3887827)
- [Howson et al., CANdid, VehicleSec 2025](https://www.usenix.org/system/files/vehiclesec25-howson.pdf)
