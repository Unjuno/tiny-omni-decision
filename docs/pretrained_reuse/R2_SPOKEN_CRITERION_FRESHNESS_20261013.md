# Spoken criterion freshness gate fix

Found and reproduced a contract bug: `aligned_evidence_json` checked only
`required_sources`. A visual task with a spoken criterion could run despite
stale criterion audio, and the criterion source was omitted from selected indices.
Two new regression cases failed before the fix.

The gate now requires the spoken criterion's exact audio source to satisfy the
same causal/freshness policy. Fresh criterion indices are passed to the bound
predictor; unavailable criteria abstain without invoking it. If environmental
audio and spoken criteria use different sources, both are required: availability
is combined with AND rather than replacing one source with the other.

Four regression cases cover fresh/stale criterion audio for visual tasks and
distinct environmental/criterion audio. This is metadata correctness only. The
predictor still has to interpret the spoken criterion and verify actual payloads;
no speech comprehension or model-quality improvement is claimed.

Existing no-criterion behavior, weights, datasets and artifacts are unchanged.
Full CPU tests:734passed,1skipped. Targeted Ruff and diff checks pass.
