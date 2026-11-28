# Physion++ visual-observability spot check — 2026-11-28

## Scope

This was a qualitative first-pass review of three existing train-only,
exact-recorded-factor-matched contact sheets from attempt-06:

- `friction_collision_pp.png`
- `mass_collision_pp.png`
- `bouncy_platform_pp.png`

Each pair shows eight sampled frames per clip at 128-pixel letterboxed
resolution, with the existing No/Yes label. The sheets were chosen by the
pre-existing deterministic selection, not selected after seeing their
appearance. Validation, sealed/test rows, or additional media were not opened.
No labels, corpus rows, or media were changed.

## Observations

In multiple pairs across these three scenarios, the sampled sequence shows
object motion and interaction over time: relative position changes, approach
or separation, collision-like contact, and passage around/behind barriers.
This suggests that some outcomes may be visually inferable from temporal
evidence in the rendered clips.

The review does **not** establish that every label is observable. At thumbnail
scale, exact contact with the target zone and brief events are difficult to
judge consistently. Some clips also differ in lighting, camera-visible
background, pose, or scene realization despite matching all recorded source
factors. Those differences may be legitimate consequences of unrecorded
initial conditions, nuisance variation, or rendering cues; this review cannot
separate them. Exact matching on the available factor metadata does not control
unknown simulator variables.

No independent human physics annotator performed a frame-by-frame review, and
no per-pair correctness count was produced. This is an AI-assisted visual
triage only. Do not treat it as a relabeling audit, proof against shortcut
learning, or a quality evaluation of any model.

## Consequence

The evidence is mixed but useful: temporal visual signal is present in at
least some reviewed clips, while observability and nuisance balance remain
unverified. Do not conclude that the task is impossible or that model capacity
is the bottleneck. Before another model sweep, conduct a blinded human review
of a predeclared subset with full-resolution clips and the official contact
definition, recording annotator agreement and ambiguous cases. In parallel,
compare low-level rendering statistics across labels within each exact-factor
stratum. If labels cannot be consistently judged from video alone, revise the
task/data rather than tuning a model against ambiguous supervision.

This report covers only the three named train contact sheets; it does not
close the R2 data-quality gate.
