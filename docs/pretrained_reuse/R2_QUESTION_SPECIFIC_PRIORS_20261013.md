# Fit-only answer distribution audit

Counted semantic answer text on624 fit questions only; no development answers or
model scores used. Exact counts are in the adjacent JSON. Largest answer shares
include count77.2%, identical72.2%, transparency77.2%, camera motion83.0%, and
hidden-object movement69.6%. These are within-training label frequencies, not
evaluated classifiers: an answer text may be absent from another row's options.
The single scene-description question cannot support generalization estimates.

Any future multitask result must retain task-level metrics and candidate-aware
question-only/mismatched-video controls. An aggregate increase could simply reflect
these easier/prior-heavy question types; it must not replace location-pair accuracy.

The pinned upstream README specifies cut frames to remove endings that reveal
answers. It does not certify every auxiliary question as answerable before the cut.
Static object properties are plausible candidates for pre-cut grounding, but that
is an inference requiring visual verification. Camera motion and hidden-object
movement may describe events outside the retained window. Those questions remain
REVIEW rather than silently being assigned to the existing truncated feature cache.

No labels, splits or existing benchmarks were changed. The780-row question-specific
corpus remains an audit artifact, not an admitted training corpus.
