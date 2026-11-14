# SigmaCollab multimodal source screen — 2026-11-14

## Disposition

Keep SigmaCollab `REVIEW_NOT_ADMITTED` for Path A quality fitting or evaluation.
The official dataset card declares CDLA-Permissive-2.0, and describes aligned
audio, egocentric video, depth, pose/gaze streams and task annotations. However,
the available task-level cohort is only 85 sessions from 21 participants, with
no publisher-defined train/validation/test split. This is below the roadmap's
provisional minimum of 200 independent source groups. The session count must
not be substituted for independent participants.

The source therefore cannot support the planned participant-generalization
gate or a robust estimate of a multimodal Decision model. No training,
evaluation, annotation inspection, media download, or model run is justified.

## Task and group feasibility

- Pinned source: `microsoft/SigmaCollab` at commit
  `a637688c0ccfd3289c7828cb937fd3475c74db7f`.
- The publisher's dataset card reports 85 task-execution sessions from 21
  participants, plus 8 expert demonstration sessions performed by one author.
  Keep the expert sessions separate; they do not add independent participants.
- The paper states that it provides no specific train/validation/test split and
  that custom splits should stratify by attributes including participant ID.
- The paper's outcome labels are session-level (correctly completed,
  incorrectly completed, abandoned, system failure). Its 1,583 sub-step
  executions do not constitute 1,583 independent labeled groups.
- The available record does not demonstrate that audio and video are both
  necessary for any frozen supplied-option task. Transcripts, displayed task
  instructions, and system-generated speech would need to be isolated as
  possible shortcut channels before defining such a task.

## Source evidence

- The pinned [Microsoft dataset card](https://github.com/microsoft/SigmaCollab/blob/a637688c0ccfd3289c7828cb937fd3475c74db7f/DatasetCard.md)
  declares CDLA-Permissive-2.0, lists the available streams, and identifies 21
  participants and 85 sessions.
- The [dataset paper](https://arxiv.org/html/2511.02560) states that no official
  splits are defined, describes participant-group stratification, and reports
  the 85-session / 21-participant cohort and task-level outcomes.

## Next gate

Do not acquire the payload or fit a readout for this source. Reconsider only if
new, rights-cleared participants are collected and the combined cohort supports
at least 200 independent participant groups, with a predeclared task whose
labels are not derived from the same input stream being tested. Otherwise keep
search bounded to a source that already satisfies the group and label gates;
do not relax the group threshold to turn this small cohort into a product claim.

This source screen is not a quality result and does not advance the R2 modality
or fusion quality gates.
