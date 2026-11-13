# Native sensor smoke protocol

Bounded task: HARTH activity-code candidate probabilities from calibrated6-channel
acceleration and timestamps, with no text. Two temporal convolutions32 channels,
kernel5, mean pooling, linear12-code readout. Fresh seed17; all parameters trained.
This small sensor-specific stem is explicitly allowed by R2-M; it is not a new
foundation model or a pretrained sensor representation.

First8 saved training windows only; 2 AdamW updates at1e-3, weight_decay0.01.
For this functional smoke use the observed label at index64 as the target; retain
mixed-label windows. Check finite nonzero gradients, parameter mutation, finite
normalized JSON output, exact save/reload and option permutation consistency.
No validation/calibration/audit, no model-selection or quality claim. Save initial
and final states in a new directory, preserving all source artifacts.

Missing samples cause abstention in this initial strict inference policy; this is
not calibrated risk/coverage. Full-window activity classification and transition
behavior need an independent, predefined evaluation before adoption. Dynamic
candidate IDs must map to supported typed activity codes; unrestricted task
semantics are outside this probe. All12 codes remain in training loss.
