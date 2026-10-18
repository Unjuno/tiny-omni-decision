# Train object-track annotation admission

Pinned official source3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf README lines200–209 documents boxes as [x1,y1,x2,y2], per-frame is_masked, and track is_occluder (cup-game-only semantics). is_occluder identifies the covering object, not the hidden target. Object labels/boxes remain supervision, never raw decision inputs.

Across all99 previously fixed train videos, exact frame-ID intersection with the dense sampler yields9,115 boxes. All have four coordinates in[0,1], x1<x2 and y1<y2. Arrays of frame IDs, boxes and masks have matching lengths. There are9,049 unmasked and66 masked boxes;672 unmasked boxes belong to occluder tracks. No interpolation, labels changed, examples removed, training, validation or sealed evaluation.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-track-admission-01.json`, including source/archive/report hashes and all selected annotation rows. Earlier availability counts are retained separately.

This supplies possible spatial/identity supervision, not proof that frozen features support accurate tracking. The exact hidden-question target mapping remains unresolved. A bounded next option is inference-only token correspondence initialized by an upstream tracking box, evaluated at later unmasked exact annotations. That would be an oracle-initialized tracking diagnostic, not the production Decision interface or a new answer-accuracy benchmark. It must retain displacement/static controls and cannot establish temporal reasoning from stationary-object success.
