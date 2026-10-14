# Frozen native mask-memory comparison

## Question and unchanged goal

Does the released SAM2 video-memory path retain automatically generated object
seeds across the existing32 source observations better than repeating frame0?
This is a distinct pretrained recurrent mask representation, not another cosine,
nearest-point reassignment or readout-budget fit. The full <=200M native Omni
conditional/joint/OOD 95–98% product objective is unchanged and unmet.

Use all eight previously known train references in their original order:
10103,10186,10212,10387,1087,10886,10894,10903. They are observed/selected training
material, not blind, actor-disjoint or population evidence. Preserve all existing
artifacts. No optimizer, new weights, semantic prompts, target filtering, official
validation or sealed data. Automatic semantic discovery remains unresolved.

## Input and model contract

Reuse the existing immutable native-wrapper frame0 proposal banks. Keep **all696
NMS survivors** (55,186,43,168,67,64,73,40), in saved order; do not apply old top3,
GT matching, score/category threshold or group-size assumptions. All1,024 source
proposals and their NMS provenance remain protected. Empty future masks are retained.
Each seed receives an independent single-object session via public
`reset_tracking_data()`; object ID1 is bookkeeping, not a source track/class ID.

Same source media/cuts and rounded-linspace32 frame indices as the admitted spatial
bank. Decode raw RGB only at those times. Native processor retains its published
1024-square profile; mask output is256-square. This is a **subsampled** video,
not full dense original-frame tracking or a causal comparison to full-frame TAPIR.

Fixed released `facebook/sam2.1-hiera-tiny` revision
`de431c4043854a71d8101e17995dfe596bf101a5`, weightSHA
`48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7`.
All38,962,498 registered parameters frozen, strict local load, unchanged environment
Transformers5.6.2/Torch2.6.0+cu124/FP32/SDPA, seed17, deterministic, TF32off.
Weights and mask outputs are FP32; the installed native implementation stores
encoded mask-memory features as BF16 internally. Preserve and record that native
behavior, rather than silently claiming every intermediate is FP32.
No batched decoder replacement; each native forward handles one object.

Arms: actual32 RGB frames; identical32-length input repeating raw frame0. Same
initial automatic mask, options/cohort/time bookkeeping. The second arm controls
changing visual evidence, not a surgical ablation of learned memory weights.
No semantic Decision Accuracy/NLL/Brier/ECE is inferred from object-score logits.

## Public cache admission and scale gate

Use public `max_vision_features_cache_size=32`, GPU inference/state storage, CPU
video storage. `reset_tracking_data()` preserves only the shared frozen vision
cache while clearing object prompts/history. No private state patch or library edit.
Record actual vision-encoder call counts and complete empty tracking-state checks.
Source/model/processor/installed-library code/config hashes bind the contract.

Separate smoke, first two saved NMS seeds of first video, all32 times:

1. Actual input, native default cache1 reference.
2. Actual input, cache32 with tracking reset between seeds.
3. Initial-repeat input, cache32 with tracking reset.

Compare every actual mask/presence raw FP32 output by shape/dtype/bytes between
1 and2. A failure rejects caching; no tolerance relaxation or silent fallback.
Cache32 must invoke vision exactly32times across both seeds, rather than64; normal
and repeat sessions have separate caches. Preserve raw smoke logits, scores,
packed masks, boxes, raw stream digests and timings. Check freeze/flags/gradients/
CPU-CUDA-NumPy RNG. No source annotations or geometry scores influence admission.

Before main, validate actual smoke effective config/contract/split and all saved
artifacts. Project all696seeds/two arms from measured warm per-seed timings plus
cold/IO margin; start only if <=7,200s and disk-reserve/output budget permits it.
Main repeats each arm's first2 raw stream digests exactly, then completes the
predefined full cohort. No frame/seed/candidate reduction after seeing outcomes.

## Persistence, controls and scope of scoring

Each candidate/time keeps packed raw-threshold0 masks, FP64 raster bboxes, FP32
object scores, empty flags, original seed/frame IDs and a raw-logit/score stream
SHA. Save every candidate, not just positives; producer readback verifies each array
shape/dtype/bytes. Large full-corpus FP32 logits are hashed during inference rather
than duplicated on disk. They are not calibrated option probabilities.

Fresh separate process reloads the same checkpoint/contract and reruns fixedfirst2
seeds/all32times/both arms. Compare original raw stream digests and packed arrays.
This fresh numerical scope is2/696seeds, not full-corpus reinference.
Independent CPU audit checks all completed mask/cohort/time/geometry/hash products.

Only after inference/fresh admission, a separate CPU scorer may use pinned existing
training-only source group-box proxies at those times. Source role IDs/boxes never
enter neural inference. First-time unique IoU>=0.5 group assignments are **posthoc
oracle anchor eligibility**; report all unknown/ambiguous anchors and all source
objects, including unsupported ones. Future absence/ambiguity/missing boxes remain
in eligible-seed denominators. Report pooled and per-video identity proxy rates and
identity-preserved source-group coverage, with the unchanged0.5 gate. No pixel-mask
truth, hidden-target selection, actor-relative QA truth or blind quality claim.

## Safety, completion and next step

Idle RTX3080 Laptop16GiB;12GiBfree preflight, peak GPU6GiB/process4GiB,8GiBdisk
reserve. Smoke600s/main7,200s/fresh600s body caps checked per frame, seed and end;
no external watchdog claim. Numerical/hash/resource faults preserve incomplete
artifacts without automatic settings changes/restart. No cloud/legacy stop/merge/
quantization/promotion. No model head/capacity increase.

Read-only execution review before smoke; tests first RED→GREEN; source/protocol
commit before GPU. Preserve failed admission and all raw prediction metadata.
Ruff/full CPU/manifests and publication-head CI. Execute rather than only plan.
If scale is impractical or native identity evidence fails, report the exact
constraint/result; do not rerun rank/prompts/thresholds to manufacture improvement.
