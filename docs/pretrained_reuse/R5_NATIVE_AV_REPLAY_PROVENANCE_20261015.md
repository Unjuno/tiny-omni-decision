# Raw AV replay dependency correction

Read-only review found that the frozen39eb23d producer's replay mode binds the
worker receipt bytes but does not recheck physical dependencies inside it or
compare the full native backend profile. Preserve its source and primary run.

Use `scripts.replay_native_av_correspondence` for fresh execution. It verifies
both process dependency/media/model/code maps and primary payloads before
launch, verifies the expected complete backend profile at runtime construction
before any raw forwards, then verifies them again after exact original native
array/JSON replay. A separate `replay-provenance.json` binds the wrapper and the
completed replay receipt. Regression tests reject physical dependency changes
and profile changes even if hypothetical numeric outputs would remain equal.

No scoring/backend/preprocessing/selection/tolerance or original producer changes.
Wrapper code is part of replay execution provenance; the frozen mathematical/
frontend/head profile remains identical to primary. No new quality claim.
