# Preserved FOA direction errors by distance

No refitting, new source labels or held-out access. All54,738 previously scored
single-annotation training ticks retained. Original hash verified. Post-hoc fixed
centimeter bins and original raw-class strata; no best subset chosen.

| Distance | Ticks | Recordings | Mean error degrees | Within20 degrees |
|---|---|---|---|---|
| <=100cm |1210|14|51.82|37.69%|
| 101–200cm |28630|42|58.13|21.37%|
| 201–400cm |24863|44|64.74|18.69%|
| >400cm |35|4|70.35|8.57%|

Even the closest-distance aggregate is insufficient for reliable localization.
Do not filter to nearby events to claim success. The descriptive distance trend
is confounded by room/class/source and strongly correlated ticks; not a causal
distance effect or an independent sample confidence interval. The35far ticks are
especially insufficient. Per-class full metrics are in adjacent JSON.

This rejects treating raw mean-intensity vectors as verified spatial labels for
new AV training. Existing ground-truth angles can remain supervision, but source
visibility/visual-object association is still unannotated. No fusion-quality gate
is closed and no new training is justified merely by lower near-source errors.
A future spatial feature intervention needs its own fixed signal-processing or
pretrained-model protocol; do not fit an axis rotation on these outcomes.

Reproduce: `python -m scripts.audit_foa_error_strata`; Ruff passed. All original
predictions/audio untouched. No media download, model load or paid compute.
