# Frozen screen feature execution

Source commit ea2293f; pinned DINOv2-small weights from the existing local audit.
The same22,056,576-parameter visual backbone processed3 whole development screens
and the first8 annotated regions from each,27 inputs total. No text descriptions,
category labels or filenames were passed to the encoder. All parameters frozen.
Outward integer crops were resized bicubically to224x224 before normalization;
CLS plus mean patch features produced finite[27,768] float32 output.

Local RTX3080Laptop execution including model load measured0.660seconds;
peak allocated VRAM145,775,104bytes. This is a tiny warm-environment measurement,
not a sustained latency benchmark. Source PNG/annotation and weight hashes were
checked; processor/config hashes saved. Features and separate supervision records
are preserved at `C:/CodexArtifacts/pretrained-reuse-r2/groundcua-frozen-screen-smoke-01`.
[Exact evidence](R2M_SCREEN_FEATURE_SMOKE_20261012.json).

No classifier training or quality measurement occurred. Coarse resizing can
destroy small text, and cropped inputs receive upstream localization. Neither
full-screen action selection nor real screen+sensor fusion is established.
The3 inspected screenshots remain development-only. Next preparation must group
by application/screenshot, explicitly account for empty/unknown annotations and
measure duplicate pixels before fitting a frozen-feature category readout.
Do not call this prerequisite task a substitute for the full Omni objective.
