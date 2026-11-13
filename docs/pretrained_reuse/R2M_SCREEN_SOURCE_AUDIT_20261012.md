# Native screen source audit

ScreenSpot is an instruction-conditioned grounding evaluation, not a direct
screen-state decision corpus. Do not silently remove its instructions and treat
its targets as uniquely determined from pixels. [Official task definition](https://github.com/njucckevin/SeeClick).

GroundCUA provides screenshots and approximate UI element categories under a
declared MIT dataset license. [Official dataset card](https://huggingface.co/datasets/ServiceNow/GroundCUA).
Pinned revision `5d6845b0116029d46ec762e734701c5b8ce207c3`. Three development-only
screenshots were fixed by lexical filename from7-Zip, Anki and Audacity, then
downloaded alongside annotations; no training or evaluation occurred.

Findings from actual files:

- 181 annotated regions across3 screens; every bounding box was within image bounds.
- Anki has47 empty category labels out of56 regions. Do not infer labels from
  annotation text or count blank annotations as an `Other` class.
- All3 PNG byte hashes differ from filename stems. The shared stem establishes
  pairing only; use independently computed byte hashes for integrity. This audit
  does not determine how upstream stems were calculated or claim corruption.
- Category labels are coarse and incomplete, as the publisher explicitly states.
- No synchronized sensor signals or action-success labels are provided by these
  inspected records. This source cannot establish a screen+sensor decision task.

[Exact file hashes and coverage](R2M_SCREEN_SOURCE_AUDIT_20261012.json).
Local source inventory, README,3 annotations and3 PNGs are preserved in
`C:/CodexArtifacts/pretrained-reuse-r2/groundcua-source-audit-01`.
These inspected screenshots must never enter a later blind split.

Candidate next bounded probe: classify the UI category of a supplied region from
screen pixels, using frozen DINOv2 and typed category IDs without annotation text.
Supplying a region is an explicit localization input, not full-screen autonomous
action selection. Before training, freeze platform/screenshot grouping and audit
category coverage and pixel duplicates; do not select easy categories by observed
model performance. This is only a screen-perception prerequisite, not closure of
M1-U action quality or M2/M3 fusion. Real paired screen/sensor source remains missing.
