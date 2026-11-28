# Physion++ official contact-event sampling coverage — 2026-11-28

## Scope and method

This train-only audit joined the exact official per-frame contact annotations
to the frozen 652-row Physion++ train manifest. The pinned official metadata
audit verified all 652 labels, frame timelines, and stimulus identities with
zero mismatches; this follow-up preserved all positive contact frame indices
instead of retaining only their count and first index.

For each clip, a fixed uniform policy selects `k` indices using
`round(i*(frame_count-1)/(k-1))`, including both endpoints. The audit measured
8, 16, 32, and 64 selected frames. Exact capture means at least one selected
frame index equals a frame marked positive for target-zone contact. Tolerances
of ±1, ±2, and ±4 frames are sensitivity analyses for nearby visual evidence;
they are not exact contact-frame captures. These policies are not nested, so
increasing `k` does not guarantee monotonic exact capture for every clip.

This measures index overlap only. It does not determine whether a model could
infer contact from pre/post-event motion, whether the event is visually
distinguishable in RGB, or whether this policy matches every encoder's actual
preprocessing. No validation/sealed data, model, or GPU was used.

## Results

| Uniform sampled frames | Positive videos with exact sampled contact frame | Exact positive-frame recall | Video capture with ±2-frame tolerance | Video capture with ±4-frame tolerance |
|---:|---:|---:|---:|---:|
| 8 | 199 / 313 (63.6%) | 347 / 19,930 (1.7%) | 224 / 313 (71.6%) | 240 / 313 (76.7%) |
| 16 | 217 / 313 (69.3%) | 674 / 19,930 (3.4%) | 228 / 313 (72.8%) | 241 / 313 (77.0%) |
| 32 | 254 / 313 (81.2%) | 1,384 / 19,930 (6.9%) | 288 / 313 (92.0%) | 305 / 313 (97.4%) |
| 64 | 277 / 313 (88.5%) | 2,767 / 19,930 (13.9%) | 301 / 313 (96.2%) | 313 / 313 (100.0%) |

Across the 313 positive clips, official contact is labeled on 19,930 frames.
The median positive-frame fraction per clip is 9.9%; 94 clips have multiple
disjoint contiguous contact runs. The median first-contact position is 70.3%
of the clip timeline, so uniform sampling can leave little temporal detail
after first contact even when it hits an event frame.

Exact 8-frame capture varies substantially by scenario: 13.6% for
`friction_collision_pp`, compared with 100% for `friction_platform_pp`. The
small per-scenario counts and non-nested sample grids limit comparisons.

## Interpretation

Uniform sparse sampling can omit explicit target-zone contact in a material
fraction of positive clips, especially in some scenarios. The 8-frame result
therefore supports temporal input coverage as a plausible limitation to
investigate, not as a demonstrated cause of any model score. Nearby sampled
frames may contain approach, collision, or post-contact evidence; this audit
does not judge that visual evidence. Do not change a frozen experiment's frame
count from this analysis alone. A separately versioned follow-up could
compare event-aware windows or denser uniform sampling only after blinded
review confirms the temporal cues are observable and the model path can afford
them.

## Reproduction artifacts

- Official-timeline audit script:
  `scripts/audit_physionpp_official_train_labels.py`
- Sampling analysis script:
  `scripts/audit_physionpp_temporal_sampling_coverage.py`
- Tests:
  `tests/test_audit_physionpp_official_train_labels.py` and
  `tests/test_audit_physionpp_temporal_sampling_coverage.py`
- Official per-row results:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-label-integrity-20261128-attempt-06\train-label-verification.jsonl`
  (SHA-256 `354175780c48929c1be1c137bd76b4af0520f10605a3ed1b75237c350068dfdb`)
- Analysis output:
  `C:\CodexArtifacts\pretrained-reuse-r2-local\physionpp-temporal-sampling-20261128-attempt-01`
- `audit-report.json` SHA-256:
  `837cde6e5ff46d667de70666f4715c70ea559de168b6b669d08b33c5b2e5bb5c`
- Per-video coverage SHA-256:
  `e841ebfefea1103c62d7aff7a6a25f38107ab89a7bcd9daf6805756d0540a604`
- Sampling script SHA-256:
  `b5c0dc41ec7bcae4257b45311ab5e8271f99141c5a59fc8fff36cc2e9191c4af`
