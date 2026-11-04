# Upstream label transformation recheck

All198 questions across99 training videos were joined to the original
train_annotations.zip/all_train.json by video_id and upstream_question_id.
Exact question text, option multiset and semantic target text match for every row.
Comparison uses target text after option remapping, not equality of numeric indices.
The adjacent JSON preserves all mapped/upstream indices and both source hashes.

This rejects a simple question/answer remapping error on the audited training
corpus. It does not certify upstream annotation truth, spatial viewpoint ambiguity,
visibility of the hidden object, or temporal information retained by frozen slots.
No validation data, model inference or new training was used.

Combined with the failed relational-head comparison, a further head-capacity
sweep is not supported. Localization/identity quality and task grounding remain
open; the previously measured sparse oracle-mask and tracking results remain the
relevant evidence. Full multimodal Decision quality remains unmet.
