# R2 time-specific model-mask tracking — actual execution, 2026-10-14

## Result and scope

**PASS for the fixed input/tracking primitive; Omni quality remains unqualified.**
All1,070 model-produced NMS candidates across all32 times on the known public
Perception Test training regression scene `video_10903` were seeded at their own
original frames and tracked over the complete558-frame official cut. Keep the
76 repeated time/xy seeds as separate proposal identities:994 unique time/xy
locations are not994 unique objects. No GT selection, TopK, fitting, validation
scoring or sealed-audit access occurred.

Every primary/replay output, including all raw tracks, occlusion/expected-distance
logits,384 query-time appearance columns,512-column tokens, proposal/time IDs,
observation flags and CPU/CUDA RNG states, passed shape/dtype/byte readback.
A fresh model/process independently reproduced all15 arrays exactly. Both NPZ
archives have SHA-256
`8ba21ebd2fa28b7cc13a8bfc2f40846f8854173658186f0a7bf5a23c3c032732`.
This is reproducibility of one known training scene, not scene generalization,
semantic identity or Decision Accuracy.

## What changed and was checked

Source commit: `34bda1424450262fd6c80798b8d9602c40809dfa`.
[Fixed protocol](R2_TIMED_MASK_TRACKING_PROTOCOL_20261014.md).
The existing exact foreground COM-nearest mask seed becomes
`[original_frame,y,x]`; the original proposal ID is preserved. Full-video query
appearance uses that frame, rather than substituting frame0. Motion columns and
the previous32 original sample IDs remain unchanged in definition.

Two Important code-review admission gaps were fixed before GPU execution:
rebuild all seeds from pinned original proposal banks and compare every query
array/input/source/time record before model creation; admit a fresh replay only
after verifying primary result/config/bank hashes and identical effective config.
The review then found no remaining Critical/Important issue. The pre-review CPU
plan01 remains preserved; the corrected experiment uses plan02 only.

The20 new helper tests first failed because the helper APIs were absent, then
passed. The9 new binding checks first failed for the missing admission API and
passed after implementation;2 additional dtype regression checks passed.
Full CPU suite: **1,139 passed /1 skipped,16.41s**; Ruff and all CI manifest/catalog
commands passed. Model/GPU execution stays outside normal CI. No environment
upgrade, media download, model training, product promotion or main merge.

## Actual resources

| Process/body | Wall seconds | Forward seconds | Peak Torch bytes |
|---|---:|---:|---:|
| Preserved pre-review CPU plan01 |1.769925|—|—|
| Corrected CPU plan02 |2.087001|—|—|
| Primary full1,070-query tracking |63.015391|52.150294|5,849,186,304|
| Fresh full replay |63.476333|52.556005|5,849,186,304|
| Independent CPU audit |2.251860|—|—|

Sum of all measured bodies: **132.600510s**, including the preserved CPU planning
attempt. Peak Torch allocation is approximately5.45GiB; reserved approximately
7.75GiB (8,319,401,984 bytes). Each tracking archive is11,440,750 bytes. These
body/forward windows exclude initial module imports and are not an isolated
end-to-end Decision latency benchmark or GPU-active-time/electricity estimate.
Cloud cost0; local electricity cost not measured. GPU was idle after both
processes returned exit0; no live or unrelated process was stopped.

RTX3080 Laptop16GiB; native Windows PyTorch CUDA loader; Python3.11.9,
Torch2.6.0+cu124, CUDA12.4, NumPy2.4.6, PyAV18.1.0. Fixed FP32/TF32off,
CPU4, deterministic seed17, chunk16, strict TAPIR pyramid_level0/extra_convsFalse.
The frozen tracker has31,072,327 registered parameters, no new adapter/head.
Model-state SHA-256 before/after and in fresh replay:
`3c92ad70854f549bc46fde282f4248e949d733fadffcfa2b060afada2ea6dd4a`.
All parameters remain frozen with gradientsNone. This component count does not
establish the full resident Omni parameter/package gate.

TAPIR consumes CPU RNG for its query permutation even in eval. Before/after/final
states were saved and exactly replayed, not reported as unchanged. The subsequent
query-appearance extraction passed the unchanged-RNG gate. Resource limits were
checked after forward, not enforced by watchdogs; neither GPU attempt exceeded
the predefined600s body/12GiB allocation gate. Initial free GPU16,008,609,792 bytes
(approximately14.9GiB);
the≥14GiB and≥8GiB disk admission checks passed.

## Independent audit and limits

The separate CPU audit recomputed every mask's exact integer-distance seed from
the original32 proposal banks, without importing the production seed helper,
GPU/model construction or labels. All1,070 `(original_frame,proposal_id)` pairs,
all query arrays and all15 primary/fresh arrays matched. It independently
reconstructed the128 ordered motion/uncertainty columns from raw arrays and
verified the remaining384 columns match query appearance.

There are18,697 query/sample pairs on-or-after that candidate's observation and
15,543 before it. Both are retained. These flags are not visibility/confidence,
physical birth or actual causal memory: this offline model sees the entire clip
and can use future context. Query-time estimated tracks deviate from their input
seed by at most **1.101238 pixels**; outputs were neither clipped to the seed nor
represented as exact initial-mask membership. Keep this numerical limitation
when measuring later association.

The prior discovered visible-item mask was not singled out for tracking. No
semantic small-item selector, container binding, actor-relative relation,
begin/end-game answer or held-out improvement was measured here. Do not infer
those from seeds, model uncertainty or numerical replay. The previous expensive
native SAM discovery stage is still required for this raw input path; no product
latency improvement is claimed from cached TAPIR inputs.

Next measure input-only candidate memory and covering-object association with
all candidates/unknown cases retained. Source labels can enter a separate
posthoc diagnostic only after outputs exist; GT-selected inference and another
fit of the previously failed99-scene readout are not justified by this result.
Text/image/audio/video joint improvement and95–98% task/OOD quality remain open.

## Reproduction and local artifacts

Run in `C:/CodexWorktrees/pretrained-reuse-omni` at the source commit, using the
existing pinned model/dependencies and fresh output paths:

```powershell
python -m scripts.probe_timed_mask_tracking --output C:/CodexArtifacts/pretrained-reuse-r2/dynamic-timed-mask-plan-02
python -m scripts.probe_timed_mask_tracking --query-run C:/CodexArtifacts/pretrained-reuse-r2/dynamic-timed-mask-plan-02 --output C:/CodexArtifacts/pretrained-reuse-r2/dynamic-timed-mask-tracks-01
python -m scripts.probe_timed_mask_tracking --query-run C:/CodexArtifacts/pretrained-reuse-r2/dynamic-timed-mask-plan-02 --reference C:/CodexArtifacts/pretrained-reuse-r2/dynamic-timed-mask-tracks-01 --output C:/CodexArtifacts/pretrained-reuse-r2/dynamic-timed-mask-fresh-replay-01
python C:/CodexArtifacts/pretrained-reuse-r2/audit-timed-mask-tracking-01.py
```

The runner refuses existing output directories; preserve these historical paths
and choose new ones for reproduction. Source/model hashes must still match.
Small [publication JSON](R2_TIMED_MASK_TRACKING_RESULTS_20261014.json) records
every result path/hash, all measured summaries and references. Its SHA-256 is
`340e1102a8f69739efc8a12c15530252988656f24c3bdb90f75422ad2452324a`.
Primary config SHA-256:
`c13d3796889ef52e288ef84fba900cab7be4c85af039f27265564c6c4a710925`.
Primary result SHA-256:
`1ed0d7e4b3114be53fd6df7359d895f04f2e361579d052231f7c95b4e96d1c05`.
Fresh result SHA-256:
`9d0fee2b733a5773a4e3f9f8b45899d40cc102e09299e88a6350f7b90a581288`.
Audit script SHA-256:
`3f2f95d18d73d8cdb1caa73bc085a74e01721b988d252cf671c5a19d7d65f869`.
Weights/media/NPZ stay local outside Git; all legacy artifacts are unchanged.
