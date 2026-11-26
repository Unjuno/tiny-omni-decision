# Training-only video grounding coverage

The pinned CLEVRER training question file was joined to the 256 existing cached
training scenes. No validation or sealed-audit questions were loaded. There are
2,812 descriptive questions spanning 1,302 exact symbolic programs. Programs
contain literal attribute values, so this count is not 1,302 independent semantic
skills. All cached training scenes are found exactly once in the upstream file.

| Exact task | Training scenes | Label coverage limitation |
|---|---:|---|
| Total collision count | 55 | Counts 1–5; count 5 has one example |
| Number of entering objects | 53 | Counts 0–3 only |
| Color of last entering object | 43 | Eight colors; smallest class has 3 examples |
| Number moving at end | 40 | Counts 0–4; count 0 has one example |
| Shape of last entering object | 26 | Three shapes; smallest class has 5 examples |

This inventory changes the next experiment: an isolated fixed-question head on
these few labels would confound representation quality with severely limited
supervision. Do not infer encoder failure from that experiment or replace the
full video target with these easier tasks. First establish broader, permitted
training-only object/event supervision (or group compatible programs with an
explicit question-conditioned readout), preserve existing scene splits, then
compare matched, mismatched and temporally reordered video under a frozen
protocol. That additional supervision and probe have **not** been implemented
or run in this inventory.

This is a data-availability finding, not proof that insufficient data caused
previous failures. Existing pooled/token readout failures and frozen models are
unchanged. The complete per-program training label counts and source SHA-256s
are in [JSON](R2_VIDEO_GROUNDING_COVERAGE_20261012.json).

Reproduce: `python -m scripts.audit_video_grounding_coverage`. The script verifies
the pinned raw source hash, train-only manifest, unique scene records and complete
scene join, and refuses to overwrite results. No new model inference or training.
