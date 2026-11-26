# Perception Test MCQA: joint-modality admission check

Actual pinned training archive reopened read-only;2184train videos,7392questions.
The complete question schema contains id,question,options,answer_id,area,reasoning,
tag. No explicit required-modality field. All tags/reasoning counts are preserved
in the adjacent machine-readable report. Case-insensitive substring screening of
question text for sound/hear/audio finds0; this is only a lexical diagnostic.

Do NOT conclude every question is visual-only, or that the dataset lacks useful
sound annotations. Existing sound-event records remain separately available.
The measured fact is that this MCQA schema cannot directly certify audio+video
necessity. Do not label it a complementary joint benchmark without task-specific
manual/structural evidence and unimodal controls. The previous recording-matching
and sound-visibility experiments remain closed; neither supplies semantic fusion
quality. No new training, media acquisition, validation/audit or model inference.

Reproduce: `python -m scripts.audit_perception_av_questions`. Source archive hash
validated, all source splits asserted train. Ruff passed. Next source work must
seek explicit sound/visual interaction supervision, not infer it from video files
merely having an audio track. Broad Omni goal remains active and unmet.
