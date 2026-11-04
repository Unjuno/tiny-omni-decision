# Descriptive grounding probe protocol

Freeze this protocol before model execution. Diagnostic only, not a replacement
for causal Video quality or the four-modality >=95% target.

Use `video-grounding-descriptive-inputs-01`: 1,337 train questions / 256 scenes,
2,200 validation questions / 200 scenes. All supported descriptive questions
were retained on validation. Exclude 1,475 training questions with identical
normalized question/options text on validation; this protects the existing
content-disjoint policy but creates a more difficult wording-transfer setting.
No held-out labels/scores informed removal. Existing artifacts remain unchanged.

Compare fresh identical small candidate scorers on frozen pooled VideoPrism
768-dimensional features plus frozen MiniLM question/option features versus
zero video. Seed17, AdamW lr1e-3 / decay1e-3, question batch8, 24epochs, no
encoder training. Keep full candidate taxonomies, no binary conversion.
Select earliest minimum equal-taxonomy validation NLL including initialization.
Keep final and best separately and check exact best reload. Record all curves,
Accuracy/NLL/Brier/ECE by taxonomy and temporal/static grouping, train metrics,
parameters, time and VRAM. After selection evaluate deterministic scene-cyclic
mismatched video without selecting or calibrating from that diagnostic.

Use a separate two-epoch smoke process first. Neither this probe nor pooled
features certify temporal order preservation; no reverse-frame features are
available in this cache. Do not infer a capacity ceiling or extend the budget
from failure. Positive visual contribution would justify finer temporal tests;
absence of contribution returns to representation/grounding investigation.
