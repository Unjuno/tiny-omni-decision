# Eight fit-video midpoint witnesses

Inspected the first eight videos in the existing fixed fit order, one retained
midpoint frame each. No development, final evaluation, or sealed data was opened.
Local contact sheet and frame/media provenance:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-auxiliary-visual-audit-02/`.

| Video | Frame | Limited visual observation |
|---|---:|---|
| video_10103 | 229 | Two glasses visible; appearance differs. |
| video_10186 | 325 | Hand occludes a cup and right edge truncates another object; count and all handles cannot be certified from this still. |
| video_10212 | 525 | Three differently appearing vessels visible; hand and orientation limit handle inspection. |
| video_10387 | 254 | Several inverted vessels visible, one under a moving hand; small image scale limits exact attribute inspection. |
| video_1087 | 287 | Three white vessels visible; handles are not clearly visible. Upstream says “They probably all have handles.” This still cannot confirm it. |
| video_10886 | 316 | Three similarly patterned vessels visible; transparency cannot be confidently judged from this still. |
| video_10894 | 272 | Hand and image boundary obscure portions of the white vessels; full count and transparency need more evidence. |
| video_10903 | 279 | Three differently colored vessels visible, one held by the person. |

These are qualitative witnesses, not new ground-truth labels or a quantitative
annotation accuracy estimate. Failure to verify an answer from a single still is
not proof that the upstream answer is wrong or unavailable from the retained clip.
The inspection does show that static attribute questions are not automatically
unambiguous: visibility, viewpoint, image scale, and qualified answer wording matter.

All 780 question-specific candidate rows remain REVIEW. No target was rewritten,
no example was dropped, and no auxiliary training was launched. Any future admission
needs a task-specific retained-window rule applied consistently across the corpus;
these eight examples alone do not justify a new training corpus. Hidden-object
motion and final location cannot be established by this midpoint inspection.
