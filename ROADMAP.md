# Roadmap — Pretrained-Reuse Omni Decision (proposed 2026-10-10)

> **Status: architectural redirection proposal in Draft PR; R0 inventory and initial R1/R2 audits/probes are underway as of 2026-10-11. A shared frozen-feature candidate-scoring core now serves the text/image/audio/video Path A probe runners; typed immutable observation caching has been exercised on 1,920 audio assets, 1,536 CLEVR-4 images, eight reused CLEVRER scenes, and reusable state/query/candidate text features, with saved predictions rechecked after cache reload. A byte-bounded LRU now serves the real frozen CLEVRER CPU scorer with exact direct-path output parity and measured evictions. A separate expanded CLEVRER descriptive readout used 256 train and 200 validation scenes; it improved NLL/Brier but not Accuracy over the prior small readout on identical new validation IDs and showed early overfitting. The initial frozen VideoPrism + MiniLM binary reasoning readout reached macro Accuracy 0.519 on reused development validation scenes. A new content- and scene-disjoint 200-scene control found near-chance Accuracy for fixed video+text, same-data text-only and shuffled-video readouts; shuffling scene video left NLL/Brier essentially unchanged, so this setup shows no clear useful scene-specific video contribution. A full-coverage static CLEVR-4 VideoPrism probe used 8,424 unique train images and scored 87.9% on the reused 512-image selector and 88.1% on a one-time 1,595-image validation subset; a fixed-task runtime replay matched 128 saved decisions on 32 images, while the 458MB bundle and 4.26GB peak CPU working set miss practical lightweight packaging targets. This is synthetic image taxonomy performance, not video quality. These are bounded development diagnostics, not blind audits or general video claims. Frozen synthetic-English text, CLEVR-4 image, narrow audio keyword, video, and a one-time near-uniform Japanese zero-shot text diagnostic are measured; synchronized audio-video quality, deployment-ready packaging, explicit V-JEPA checkpoint rights, and full R1/R2 exits remain open.**
> This roadmap replaces the *future direction* in the previous `main` roadmap, not the factual record of completed experiments. It does **not** start, stop, modify, resume, or declare successful any training job; merge or close any open PR; change an existing config, model, dataset, or checkpoint; or authorize paid compute.
>
> **One-sentence mission:** Build a locally deployable, genuinely multimodal, calibrated **Decision model**, preferably **100–200M total resident model parameters**, by reusing *already learned* video/world and cross-modal representations with as little new data, training, and inference work as possible. Neither Gemma 4 E2B nor EmbeddingGemma 2 is a required product backbone.

## 1. Product contract and research priorities

### Product behavior

- Accept **text, image, audio, video**, and later timestamped **sensor** observations; distinguish single-media operation from **simultaneous** audio+video+text/sensor support. Do not declare simultaneous fusion until tested.
- Consume a shared observation/state and independently authored typed questions with variable candidate sets: `choice`, yes/no (`noul`), and ordinal `score`. Return **option probabilities** and permitted abstention/escalation, not free-form generated answers.
- Reuse one encoded observation for **many questions** without rerunning a costly video tower whenever its state has not changed; represent time and modality/source identity when relevant.
- The persistent world-state cache is a means to reuse temporal evidence, **not a claim of a general, causal, action-conditioned world simulator**. Add recurrent state prediction or action-conditioned dynamics only if measured task requirements justify it.

### Constraints, with distinct status

| Constraint | Contract / status | How to verify |
| --- | --- | --- |
| Total inference parameters | **Target: 100–200M**, including every resident text/audio/visual tower, Lens, projector, embedding table, head, and adapter; <100M is a stretch goal. A larger system is permitted **only as an explicitly labeled research/reference baseline**, not a qualifying compact release. | Exact loaded `named_parameters` / deduped tensor accounting, not model names or paper-size approximations |
| High decision quality | Required, but no global percentage is meaningful without frozen tasks, labels, input lengths, and decision policy. The parallel 2026-10-06 product roadmap proposed **>=95% per text/image/audio/video task**; preserve this as a **stretch product target**, not a measured result or automatic criterion across arbitrary tasks. Define task-specific gates **before** evaluating a contender. | Held-out per-task quality + confidence intervals, not just a pooled score |
| Low retraining cost | Reuse publicly licensed pretrained checkpoints first; prefer feature extraction, cached probes and small readouts, then small adapters, then selective alignment, then model merging. No full world-model pretraining in the default path. | GPU-hours, energy/cost where measurable, trainable parameter count **and** backward FLOPs / peak memory |
| Local execution | PC validation first, iOS/ARM target as hardware becomes available. No cloud request must be required in the final inference path. | Package bytes, peak resident memory, warm/cold latency p50/p95, battery/energy where measurable, offline functional test |
| Deployment packaging | **Initial engineering target <=200 MB** including all model files, scales, higher-precision exceptions, tokenizers, lenses and optional adapter; aspirational <=100 MB. This may require low-bit storage and is **not** guaranteed by meeting the parameter target. | Serialized artifact size and exact runtime load |
| Development resources | Available historical reference: RTX 3080 Laptop 16 GiB VRAM, 32 GiB host RAM; target isolated CPU tests and bounded GPU runs. The old <=US$15 rented-compute budget is a *planning ceiling*, not an authorization to spend. | Before each GPU experiment record device, driver, software stack, wall-clock cap, samples, and cost |
| Reproducibility / rights | Immutable checkpoint/processor revisions, SHA-256 for selected inputs/artifacts, rights-aware data manifests, sealed evaluation, reproducible preprocessing and seeds. | Fail-closed manifest audit, split leakage checks and saved run ledger |

**Optimization order:** (1) usable quality on the required tasks, (2) zero/minimal training for comparable quality, (3) parameter/byte/latency feasibility, (4) stronger sharing/merging **only where it improves the measured Pareto frontier**. Never promote a smaller model solely because it is smaller.

## 2. Historical evidence to retain — not the new architecture

Completed on legacy `main` as documented in [`docs/DURABLE_TEACHER.md`](docs/DURABLE_TEACHER.md):

- Pinned Gemma 4 E2B Decision LoRA Teacher; 256 optimizer steps on a frozen 4,859-record train set, 2,901 validation and 2,917 final held-out evaluation examples.
- Reported overall held-out accuracy **0.7343**, NLL **0.6380**, Brier **0.3344**, ECE **0.0344**. By modality: text **0.6723**, image **0.5703**, audio **0.9492**, video **0.4176**. These are **dataset-specific historical measurements**, not cross-model guarantees.
- Original video data represent a small number of independent clips; multiple decisions from one clip are not independent scenes. [`docs/DATASETS.md`](docs/DATASETS.md) retains licensing/source restrictions.
- The historical teacher is useful as an **optional, separately measured reference or option-probability source**, not a required training objective. The unsuccessful embedding-level self-distillation attempt motivates an ablation against geometry imitation; it is **not proven** that the loss alone caused the failure.
- Keep pinned checkpoints, manifests, test fixtures and prior logs byte-identical. Do not re-use repeatedly inspected old test sets for model selection; define a new sealed final audit for the new research sequence.

**Parallel work: as checked 2026-10-10.** GitHub has open PRs **#9–#14**, including Teacher-quality work and EmbeddingGemma ternary/QAT work, several in draft/stacked branches. A separate `codex/roadmap-compression-video-specialization` branch describes another compression/video program. Their results, historical run locks and ongoing processes are **not canceled or reinterpreted** by this document. No auto-merge, auto-close, checkout/overwrite of live worktrees, or stop/resume decision is authorized. Evaluate each branch independently before deciding to archive, merge, or repurpose its code. Especially do not claim exact optimizer resume from a weights-only warm start.

## 3. Competing architecture paths (not three compulsory stages)

### Path A — Frozen modular reuse (**mandatory low-cost baseline**)

Use real, available pretrained visual/video, text and audio encoders, plus tiny readout/fusion modules. Extract and cache modality-specific **token or intermediate features**; preserve temporal positions. A small conditional candidate scorer consumes cached features.

- Candidate visual/video: **V-JEPA 2.1 ViT-B**, pretrained and already distilled from a larger video model. Source-derived inventory is about **86.8M** including separate image/video patch embeddings; **recount the exact downloaded checkpoint and remove training-only predictor from deployment tally**.
- Candidate text: a small multilingual pretrained encoder (e.g., Ruri-v3-30m as a *candidate*, not a verified plug-and-play choice), or a compatible already-aligned text tower. Check Japanese and English performance before selection.
- Candidate audio: a small pretrained environmental-audio model (e.g., EfficientAT) and optionally a distinct ASR reference. **Audio event understanding is not speech comprehension.**
- For the initial reference, keep independently trained towers rather than pretending they already share weights. Target combined loaded model <=200M when feasible. This path may already satisfy the product constraints and must remain a valid winner.

**Advantage:** no new large-scale pretraining; features can be cached; clear isolation of modality-specific errors.

**Cost:** more than one resident backbone, and fusion may not discover genuinely joint signals without paired examples.

### Path B — Shared pretrained video backbone + lightweight modality Lenses (**primary research candidate, conditional**)

Reuse the same video-trained Transformer weights for image/video and, **only if feasible**, audio/text/sensor tokens via small learned or already pretrained input Lenses. Keep separate modality/time/position metadata and optionally modality-specific side information. Preserve the video representation as an explicit quality constraint.

- Start with **frozen** video backbone; do not average unrelated backbone weights.
- Prefer reusing **actual released Lens weights together with their verified backbone** when possible. ViT-Lens demonstrates an approach, **not pre-validated compatibility with V-JEPA**.
- Probe lightweight audio/text tokens with verified preprocessing and exact model revisions. If a Lens precedes a frozen backbone, training it still requires **backpropagation through the frozen backbone to the input**, thus may require costly activations. Compare this cost against Path A feature caching; small trainable-parameter count does **not** imply cheap optimization.
- Do **not** require raw multilingual text to be relearned from scratch through randomly initialized embeddings. Keep a small pretrained language encoder as a side path or revert to Path A if needed.
- Do not force complete latent equality. Evaluate shared event information and modality-specific information separately.

**Advantage:** potentially fewer resident Transformer blocks and real capacity sharing.

**Failure modes:** negative transfer, modality token distribution shift, video forgetting, expensive input-side Lens training, language capacity loss, excessive attention cost.

### Path C — Weight-aligned model merging (**research option, not default**)

Investigate Model Soups/task vectors for common-initialization models; permutation/basis alignment or FS-Merge-type approaches for verified compatible different initializations. SSAM-like methods have additional common-base assumptions.

- **Prohibited shortcut:** raw element-wise averaging of independently pretrained V-JEPA and an unrelated audio/text model merely because they encode similar events.
- Produce a tensor-level compatibility report first: names, shapes, depth/width, attention and FFN layout, nonlinearities, normalization, RoPE/position schemes, patch/token embeddings, tied weights and checkpoint ancestry.
- Compare each source on its own tasks, the candidate merge **before and after alignment**, and an independent Path A baseline. Model merging can require data/optimization; account for it.
- If alignment is not justified by tests or compute budget, **skip Path C**, not the whole product.

**Advantage:** may remove redundant resident towers without full retraining.

**Failure modes:** weight-coordinate mismatch, destructive interference, loss of temporal or unimodal information, hidden training costs.

### Architecture selection rule

Run **A first**, then attempt **B only if expected savings/benefits are measurable**. Treat **C as optional** after proven tensor/functional compatibility. If A meets quality and on-device budgets and B/C do not improve quality-cost trade-offs, **ship A**. A single shared Transformer is a *research hypothesis*, not a product requirement worth degrading performance for.

## 4. Interfaces and data contract to design before model-specific code

### 4.1 Observation features

Proposed versioned internal structure (not implemented in the present `DecisionExample` schema):

- `observation_id`, `source_id`, `source_revision`, `modality`, `time_start/end` or sample timestamps; input/media references; checksum; processing version; availability/missingness; rights/provenance.
- Encoded token array with sequence length and hidden dimension, valid-token mask, modality/time/space location, model revision and precision metadata.
- Cross-observation state/cache containing feature tokens or selected slots with explicit history window, invalidation policy, and the **exact** input checkpoint/preprocessor identity.

`World state` initially means **reusable, time-aware pretrained observation features**, not a trained latent dynamics engine. Start without compulsory 32/64 recurrent slots; compare full tokens, intermediate-layer selections, pooled features, and 16/32/64-slot compression on **held-out tasks**. Do not call any slot count sufficient without evidence.

### 4.2 Typed decision output

- Stable string option IDs and exact ordered option text; variable-size candidate set and masking; `choice` and `noul` probabilities, ordinal `score` distribution/expected score; consistent semantics for abstain if enabled.
- The scoring head is replaceable and intentionally small. Probability calibration is fitted on a **calibration-only** split; report quality and coverage at a specified risk threshold separately.
- Teacher and student token IDs are not interchangeable. Optional teacher training targets are keyed by **actual option identity and order**, with detached probabilities/logits and source provenance.

### 4.3 Cache validity and information security

- Feature caches must include the full key: model/code revision, checkpoint hash, preprocessing, frame selection, tokenizer, input hash, modality/time metadata and dtype. Never cache arbitrary mixtures under a single key.
- Caches from validation/test examples may be used only for **read-only evaluation**, never fitted into the training pipeline.
- Do not export copyrighted media or private data with the feature cache. New synchronized audiovisual/sensor corpora require component-level rights review; existing OneJev aggregate remains REVIEW/excluded.

### 4.4 Software/runtime isolation

The current `pyproject.toml` ML extra targets the **legacy Gemma stack** (Torch 2.6.x, Transformers 5.6.x). Candidate V-JEPA, Lens, TensorFlow VATT and mobile runtimes may require different dependencies. Maintain **separate lockfiles/env profiles** until demonstrated compatible; never silently replace the active Teacher/QAT environment. Validate real model load, forward and checkpoint hashes on the intended device; a config-only load is not sufficient.

## 5. Data and measurement protocol

### 5.1 Data groups and leakage

- Retain historical data/license tools. Create a distinct new experiment namespace and **new sealed test set**; select model, layer, feature extraction, frame policy, compression and all thresholds using train/validation only.
- Group visual/video decisions by original clip/scene and audio by source event/speaker as appropriate; count **unique independent scenes**, not questions, in bootstrap uncertainty and minimum-sample declarations.
- Initial smoke may reuse approved, license-audited existing text/Clevr-4/Speech Commands/CLEVRER data in train/validation. These *single-media* datasets do **not** establish audio-video joint alignment. Add synchronized paired data only when a verified need, license and split exist.
- Add difficult counterfactual **evaluation**: frame-order swap, source-video same/audio swapped, missing modality, irrelevant cues, option-order permutations, unseen candidate descriptions, unseen scene conditions, delayed new questions over the same cached observation.
- Use labels/ground truth whenever available. Compare an LLM Teacher only as optional ablation, disclose noisy synthetic labels, and preserve disagreements for adjudication.

### 5.2 Core outcome metrics

- Per-task and per-modality **accuracy / macro-F1 / Brier / NLL / ECE**; separately report abstention coverage versus error risk; no blanket 95% claim.
- Temporal and world-state tests: time-order discrimination, event sequence, object persistence, long-gap recall, whether new questions can be answered from the same cached observation, and accuracy after state-slot compression.
- Genuine multimodal gain: paired versus shuffled signals and unimodal baselines, not superficial similarity.
- Efficiency: unique train examples, GPU-hours/total cost, peak allocated and resident RAM/VRAM, actual checkpoint/package bytes, activation memory, backward cost, input tokens/frame policy, warm/cold CPU/GPU/iOS latency p50/p95 and where available energy.
- Every performance figure states **HW, clock/power/thermal policy, backend, precision, batch=1 unless explicitly changed, inputs, warmup, repeats and sample counts**. Benchmarks on disparate hardware are not directly comparable.

### 5.3 Initial pre-registered acceptance hypotheses (proposal, not measurements)

**H1 — reusable latent quality.** A frozen pretrained temporal encoder plus small readout captures specified decisions on unseen scenes without requiring new video foundation pretraining.

**H2 — shared-backbone savings.** Path B maintains each predeclared relevant task within **2 absolute percentage points** of Path A, with a paired/grouped uncertainty analysis, **and** reduces at least one measured deployment cost (package bytes, peak memory or p95 runtime) by **>=20%**, while respecting <=200M parameters. Other costs must not regress beyond their individually specified tolerances. These are **initial experiment gates**, adjustable *before* looking at outcomes, not claims of achieved performance.

**H3 — no destructive merging.** Path C preserves temporal and individual modality scores within the declared noninferiority bounds. A merge which reduces checkpoint bytes but harms joint/time-conditioned decisions fails.

**H4 — optional persistent memory.** Adding cache/state slots beats a no-memory/pooled baseline on delayed-query and missing-observation conditions by a predeclared margin, without violating memory/latency limits.

**T — minimal test.** First a deterministic 8–32-example smoke for data/forward/shape/numerics; then at least **200 independent scenes/clips when available and legally usable** for the chosen temporal paired validation. This is a **provisional floor, not a power calculation**. Grouped bootstrap and confidence intervals must be computed; use the observed event variance and desired detectable difference for formal required sample size. Preserve a sealed final set untouched by hyperparameter selection. Initial screening can be run with one seed; repeat selected comparisons with multiple seeds.

**D — decision policy.** `PASS` when all pre-registered quality, cost, provenance and runtime checks hold, including the lower confidence bound on noninferiority for quality where required. `FAIL` if a hard bound is violated with sufficient evidence. `UNCERTAIN` if sample coverage, intervals, compatibility, or device measurements cannot support a claim. Do not convert an uncertain result into a win by selectively changing task weights.

**C — counter-hypotheses.** Frozen features may lack task-relevant information; feature extraction could destroy temporal structure; alignment might suppress modality-specific details; shared token growth could make B slower than A; merging may invalidate weights; an uncalibrated head may inflate confidence.

**U — uncertainty.** Record dataset sampling/label noise, paired-scene clustering, random seed, synchronization error, video frame selection, numerical dtype/quantization, backend/clock/thermal drift. Report interval method and effective independent sample count; only compute combined standard uncertainty `u_c` and expansion factor `k` when its component estimates are actually available.

## 6. Workstream phases: exit gates determine the next phase

### R0 — Evidence preservation and design lock (**first action**)

- [x] Archive a machine-readable inventory of `main`, current independent branch/PR revisions, benchmark/data manifest hashes and experiment checkpoint hashes without modifying their bytes (`docs/pretrained_reuse/R0_INVENTORY.json`; 2026-10-10 snapshot).
- [x] Reconcile the current remote PR head/base states and local worktree/process snapshot; preserve all independent branches and runs. Read-only contribution/dependency review for PRs #9–#14: `docs/pretrained_reuse/R0_OPEN_PR_REVIEW_20261011.md`. All remain open and unchanged; CI for PRs #13/#14 was not verifiable from the unauthenticated status lookup.
- [ ] Set the deployment device, task-specific quality/confidence and time/money criteria before training or promotion decisions; retain the historical >=95% product aspiration only as a specified challenge target. The user provisionally authorized the roadmap's 100–200M parameter and <=200 MB package figures for engineering work only; neither is a release gate yet.
- [x] Publish candidate matrix, evaluation plan and separate environment contracts (`docs/pretrained_reuse/R0_DECISION_MATRIX_20261011.md`). Parameter/package targets are provisional; deployment device, task thresholds and time/money approval remain open.

**Exit:** frozen legacy baseline and written constraints; no historical evidence altered.

### R1 — Pretrained checkpoint feasibility audit (**no training**)

- [x] Verify V-JEPA 2.1-B checkpoint URL, source commit, SHA, encoder/predictor split, strict EMA load, image/video forward behavior, tokens and parameters; see `docs/pretrained_reuse/R1_AUDIT.md`. **That exact checkpoint remains REVIEW.** A separate pinned official Google VideoPrism-B checkpoint passes the explicit rights and local-load gates: 114,365,184 parameters, 8-frame finite forward, CC-BY-4.0 stated for released weights by upstream and Apache-2.0 on the official Hub card. A matched CLEVRER development probe is recorded in R2; independent task quality and package fit remain unresolved.
- [x] Inspect ViT-Lens released backbone/Lens pairing, PE Core tiny/small/base, EfficientAT, small Japanese/English text checkpoints, VATT-MA and any relevant MJEPA *released* checkpoint. Mark `code only` / `pretrained and accessible` / `unknown` distinctly; see `docs/pretrained_reuse/R1_RELEASE_LICENSE_FOLLOWUP_20261011.md` and the source-of-truth inventory. PE asset access/implementation rights and VATT access remain REVIEW/UNKNOWN as recorded.
- [x] Load and hash Ruri v3-30m as one small Japanese text candidate; a single zero-shot JamC-QA-V2 dev diagnostic was near uniform chance. It is not a Decision-head result, has unresolved training-data contamination risk, and JamC-QA remains `REVIEW` under this repository's license policy; see `docs/pretrained_reuse/R2_PATH_A_PROBES.md`. This does not close the broader language-candidate audit.
- [x] Test the selected compact local candidate forwards for preprocessing, shape, parameter count and memory. VideoPrism-B's official loader and pinned checkpoint were exercised on a deterministic 8-frame input; ViT-Lens original-backbone execution remains untested and is outside the initial small model budget.
- [x] Create compact source-of-truth inventory with `passed_load`, `license_allowed`, `has_real_weights`, `runtime_supported`, parameter tensor bytes, source revision and primary citations in `docs/pretrained_reuse/R1_CANDIDATE_INVENTORY.json`. The minimum executable/legal video-candidate gate is now met by VideoPrism-B; task quality, model package fit and broader candidate screening remain open. V-JEPA 2.1 itself stays REVIEW.
- [x] Screen the small sensor checkpoint MOMENT-1-small as a frozen feature-source candidate. Its pinned MIT checkpoint passed a local 128-sample × 6-channel CUDA embedding smoke; no task quality was measured. UCI HAR was not used for model selection because MOMENT pretraining includes UCR/UEA and the available official sources do not identify every constituent dataset, leaving overlap UNKNOWN. See `docs/pretrained_reuse/R1_MOMENT_SENSOR_20261011.md` and the candidate inventory. This does not close the sensor quality or deployment gate.

**Exit:** at least one executable, legally usable video candidate and one viable small text/audio path; otherwise pause architecture selection.

### R2 — Frozen modular baseline A (**main low-compute milestone**)

- [x] Cache deterministic frozen text/image/audio features separately for train/validation; preserve sample IDs, source/media hashes and run artifacts outside Git. The typed immutable observation-cache core is exercised on 1,920 audio observations, 1,536 image observations, eight video features, and the state/query/candidate segments of the synthetic text probe; separate CPU tests cover content-key invalidation and payload integrity. Video cache-vs-fresh feature parity now covers eight scenes (16 questions) with zero feature delta on one backend; byte-identical repeat invocation was also confirmed for one scene. Cross-backend video parity and repeated fresh-encoder parity for audio/image remain open.
- [x] Compare final-layer masked-mean, middle-layer masked-mean, and final-layer first-token text readouts with the same variable-option head on identical train/validation IDs (`docs/pretrained_reuse/R2_PATH_A_PROBES.md`). The common `FrozenFeatureCandidateScorer` now lives in `src/tiny_omni_decision/decision.py` and is used by the Path A probe runners; the saved audio multi-query checkpoint reloads through this package head with exact prediction parity. This is one synthetic English task and one seed; learned attention pooling, image/audio pooling, and per-modality readouts remain untested.
- [x] Test many questions on one cached visual/audio event; count avoided encoder executions, memory residency and latency, not merely Accuracy. CLEVRER has 88 descriptive questions over 8 reused validation scenes (11 per cached scene), with 80 hypothetical encoder executions avoided and parity on the original 16 questions. The typed persistent cache roundtrip used those eight features; direct cached-vs-fresh parity now covers 8 scenes / 16 questions. A CPU-only run routes all 88 queries through the byte-bounded LRU into the real frozen candidate scorer: 80 LRU hits, 8 persistent reads, 4 evictions, and bitwise-identical outputs. Audio multi-query uses 640 fixed validation clips and four question types per event (2,560 queries); its packaged frozen runtime now passes parity at batch size 8 with 9.91 ms per observation and 380 MB peak inference allocation. The bundle payload is 227.1 MB, above the aspirational package target. CLEVR-4's 512 validation images each reuse one persisted visual observation across four taxonomy questions; all 2,048 validation predictions match after cache reload. Compact full multimodal packaging and full image/video encoder-plus-readout residency remain open. See `docs/pretrained_reuse/R2_PATH_A_PROBES.md`.
- [x] Measure whether the fixed Audio+Text multi-query readout depends on its cached audio observation using same-ID zero-audio and other-clip-audio controls. On all 640 validation clips and four queries per clip, original macro Accuracy/NLL/Brier/ECE-15 were `0.9785/0.0693/0.0323/0.0143`; zero audio changed them to `0.5250/1.2478/0.5713/0.1769`, and other-clip audio to `0.4961/4.0821/0.9709/0.4835`. A 10,000-resample paired bootstrap by 206 speaker groups excluded zero for all macro differences. This is a post-hoc fixed-readout dependence diagnostic on checkpoint-selection validation, not a blind or broad speech result. Details and artifacts are in `docs/pretrained_reuse/AUDIO_MULTIQUERY_FEATURE_DEPENDENCE_20261011.md`; joint audio-video quality remains untested.
- [x] Compare the unchanged Speech Commands Audio+Text runtime on 1,938 LibriSpeech dev-clean four-choice transcript queries with random and CMUdict-phoneme/lexical/length-matched distractors. Accuracy was `0.2652` vs `0.2637`; paired speaker-bootstrap intervals include zero for Accuracy/NLL/Brier/ECE. Both are out-of-distribution transfer conditions, not broad speech scores; the finding does not show an accuracy penalty from hard negatives because the model is near chance in both. See `docs/pretrained_reuse/LIBRISPEECH_HARD_NEGATIVE_TRANSFER_20261011.md`; joint audio-video and package-size gates remain open.
- [x] Assemble task-scoped Audio, CLEVR-4 Image, synthetic Text, and CLEVRER video-feature Decision components. The v4 package records `170,499,716` unique parameters and `687,254,772` bytes excluding the root manifest; the provisional 200 MB target is not met. Audio/Image/Text passed relocated 8-observation parity checks in v3; the v4 Video scorer replayed all 3,014 frozen validation choices over 200 scenes with zero prediction mismatches and max logit delta `4.77e-7`. The Video component consumes cached, pooled 8-frame VideoPrism features; raw-video extraction remains a separate runtime profile. This is packaging parity, not new quality evidence or a unified dispatcher. See `docs/pretrained_reuse/R2_PATH_A_MODULAR_RUNTIME_20261011.md`.
- [x] Report initial synthetic text, CLEVR-4 image, CLEVRER video, closed-set Speech Commands keyword audio, and one-time Japanese Ruri zero-shot dev diagnostics by source/task and independent source groups in `docs/pretrained_reuse/R2_PATH_A_PROBES.md`. The Ruri result is near uniform chance and is not a trained Decision model. Joint tasks and old Gemma reference comparisons remain open.
- [x] Increase CLEVR-4 image training coverage from the prior 1,024-image probe to 8,424 unique train images using frozen VideoPrism-B and the same 512-image validation selector. Selected same-ID Accuracy/NLL/Brier/ECE-15 changed from `0.5737/1.18360/0.53216/0.09024` (V-JEPA) to `0.8789/0.28781/0.15422/0.02187` (VideoPrism); a one-time report on 1,595 other validation images was `0.8806/0.29116/0.15361/0.01487`. The encoder and training coverage both changed, so this is not a causal encoder comparison. CLEVR-4 is synthetic; count accuracy is 61.2% on the report subset. The 131.9M resident parameters meet the provisional engineering range, while 528.3MB of weights exceed the provisional package target. This does not establish natural-image or temporal-video quality; full record is in `docs/pretrained_reuse/R2_PATH_A_PROBES.md`.
- [x] Measure whether more unique CLEVRER scenes alone improve the fixed frozen-video readout. A seed-17 256-train/200-validation-scene run completed at 8 frames with frozen encoders; a same-validation-ID comparison against the old 16-scene checkpoint showed improved NLL/Brier but lower Accuracy and worse ECE, while its learning curve showed early overfitting. This is a mixed development result, not a general-video claim or product promotion; source-archive full hashes and V-JEPA checkpoint rights remain open. Full provenance and outputs are in `docs/pretrained_reuse/R2_PATH_A_PROBES.md`.
- [x] Run a bounded task probe for the newly audited VideoPrism-B checkpoint on the same small frozen CLEVRER split as the existing V-JEPA probe. Accuracy tied at 0.375 on 16 validation questions; paired intervals were too wide to distinguish the point estimates. See `docs/pretrained_reuse/R2_PATH_A_PROBES.md`.
- [x] Expand the matched VideoPrism/V-JEPA descriptive comparison to the existing 256-train/200-validation-scene split. VideoPrism's Accuracy/NLL/Brier point estimates were slightly better; a 10,000-resample paired scene bootstrap favored its NLL, while Accuracy/Brier/ECE intervals included zero. Because the validation was reused for V-JEPA selection and selected the VideoPrism readout, this is suggestive development evidence only. Both readouts showed overfit at epoch 24. A later fixed-readout comparison on 200 previously unused scenes found near-tied Accuracy and a small VideoPrism Brier advantage; 259/400 question+option strings repeat across old examples despite zero video-scene/media overlap, so it is scene-held-out but not content-held-out or blind. Keep both as research candidates; neither is promoted. Full metrics, hashes, limitations and compute are in `docs/pretrained_reuse/R2_PATH_A_PROBES.md`.
- [x] Measure a binary-choice frozen-feature probe for CLEVRER explanatory, predictive and counterfactual questions using the same 256/200 video-scene split, excluding exact normalized train question/candidate duplicates from validation. The selected VideoPrism + MiniLM readout reached 0.519 validation macro Accuracy (95% scene-bootstrap CI 0.495–0.544), with per-task metrics near chance. This is a weak development result on validation scenes previously used for descriptive-readout selection; it does not demonstrate world reasoning or justify shared-Lens/merging work. Full result, limitations and hashes: `docs/pretrained_reuse/R2_PATH_A_PROBES.md`.
- [x] On a new 200-scene content- and asset-disjoint CLEVRER sample, compare the fixed VideoPrism+MiniLM scorer, a same-training-data text-only readout and a deterministic scene-video shuffle. All macro Accuracy point estimates were about 0.50; scene-shuffling changed NLL/Brier by less than 0.0002 and its paired intervals included zero. This does not establish useful scene-specific video contribution for this readout; see `docs/pretrained_reuse/CLEVRER_VIDEO_TEXT_CONTROLS_20261011.md` for data, metrics, bootstrap and limits.
- [x] Screen CREMA-D as a synchronized audiovisual source. Its acted-affect labels and access/rights review do not make it an approved proxy for general AV reasoning; no media was fetched or added. Keep it review-only and task-mismatched; see `docs/pretrained_reuse/R2_CREMA_D_AV_SOURCE_SCREEN_20261011.md`.
- [x] Reverse the order of the same eight frames with the fixed VideoPrism+MiniLM readout on those same 200 scenes. Forward and reversed macro Accuracy/NLL/Brier were `0.49899/0.69432/0.50117` and `0.49873/0.69433/0.50118`; paired scene intervals for NLL and Brier included zero, and baseline reload reproduced saved probabilities exactly. A read-only comparison of the saved 768D pooled feature caches found mean cosine `0.999759` and mean relative L2 change `0.02013` across the 200 validation assets. This is development-only evidence that pooled features change slightly but the current readout shows little order sensitivity; it is not a backbone capability limit. The subsequent binary-choice parent-question label audit found 0–4 correct statements per parent, confirming that statement-level `[wrong, correct]` metrics should not be collapsed into a single-winner distribution. Further readout tuning on these already-used validations is not warranted; a stronger next quality experiment needs a new task-matched source or task. See `docs/pretrained_reuse/CLEVRER_TEMPORAL_ORDER_CONTROL_20261011.md` and `docs/pretrained_reuse/CLEVRER_BINARY_REASONING_PROBE_20261011.md`.
- [x] Test one predeclared eight-position scalar temporal pooling rule with VideoPrism-B and the fixed MiniLM/scorer on the same 512 train questions and 400-row, 200-scene development holdout. Only eight weights were trained; all four macro metrics' paired scene-bootstrap intervals include zero, with one net extra temporal answer and worse ECE/Brier. This does not justify further tuning on the reused validation set or establish a backbone limit. Report, cache parity correction, hashes, metrics and restrictions: `docs/pretrained_reuse/CLEVRER_TEMPORAL_POOLING_CONTROL_20261011.md`.
- [x] Run the predeclared same-frame reversal control on 200 scenes with one temporal and one static descriptive question per scene, using the fixed expanded readout. Temporal Accuracy changed from `.435` to `.440` (one net item); NLL/Brier changes were near zero and paired scene-bootstrap intervals included zero or touched zero. Forward reload parity was within `1.79e-7`. This remains a reused-validation sensitivity diagnostic; it neither establishes temporal capability nor explains the earlier binary-choice result. Details, hashes, metrics and compute are recorded in `docs/pretrained_reuse/CLEVRER_TEMPORAL_DESCRIPTIVE_ORDER_CONTROL_20261011.md`. The fixed mean-pooled readout shows little evidence of frame-order sensitivity; inspect pooling/task construction before any layer-level attribution.
- [x] Screen synchronized AV sources. YT-NTU-AVQ has 1,620 rated clips, but inspected metadata lacks original source IDs and its labels measure subjective perceptual quality; it remains `review` with no media fetch. UnAV-100 describes over 10K untrimmed clips with time-localized AV events and a YouTube ID/timestamp/split manifest, but underlying video copyrights remain with owners and the project-page CC BY 4.0 claim conflicts with a Hugging Face mirror's CC BY-NC-SA 4.0 card. Qualcomm QIVD describes 2,900 spoken-question video/audio records but only 22 in its Audio-Visual category; its agreement limits use to internal non-profit research, excludes commercial use, and restricts third-party data combination. It remains `review`; no access was accepted and no files were fetched. None of these sources clears the product-aligned joint-AV gate. See the joint-AV screen reports and candidate manifests.

**Exit:** a reproducible, measured A baseline that meets stated functional requirements; otherwise diagnose missing features/labels before spending on new backbone training.

### R3 — Shared video backbone + modality Lenses B (**only if A has a measurable inefficiency**)

- [ ] Audit exact backbone compatibility for any pretrained Lens; first reproduce the published Lens-with-original-backbone behavior.
- [ ] Try Lens(s) into a frozen V-JEPA-class video backbone with small bounded data, while monitoring negative transfer and temporal probes. **Input-side Lens training needs gradients through a frozen backbone**; set peak activation and GPU-hour gates before launch.
- [ ] Compare B with A under matched tasks and measured costs. If text/ASR quality requires an independent small tower, retain it and count it; do not force shared-backbone purity.
- [ ] Ablate modality tags, timestamps, mid-layer readout and independent modality side channels. Prefer minimal intervention over full-backbone fine-tuning.

**Exit:** B passes H2 and is preferable on the measured Pareto frontier. If not, adopt A and stop sharing work.

### R4 — Selective cross-modal alignment / predictive repair (**only if diagnosed**)

- [ ] Identify whether errors are missing input information, lens–backbone distribution mismatch, timing/synchronization, poor small-head generalization, or genuine missing cross-modal knowledge.
- [ ] If synchronized rights-approved audiovisual examples exist, compare contrastive/event-alignment and MJEPA-inspired latent prediction as **small, bounded auxiliary losses**, preferably without rewriting a massive encoder.
- [ ] Keep unimodal retention tests, held-out audio-only facts and video temporal probes; avoid forcing identical whole embeddings.
- [ ] Only consider JEPA predictor, recurrent world update, LoRA or selective unfreeze when a counterfactual state test demonstrates need and the fixed resource budget allows it.

**Exit:** demonstrable gain on held-out joint decisions, without erasing unique modality information. If no paired data, record R4 as not testable and use A/B rather than fabricate alignment.

### R5 — State reuse and caching (**functionally driven, not assumed**)

- [ ] Establish observation caching correctness, event updates, invalidation and multi-query reuse on one observation first.
- [ ] If delayed queries need persistence, compare no memory, 16/32/64 compressed latent slots, and a gated update; retain time/location/missingness metadata. No default promise of action-conditioned physics.
- [ ] If a State Predictor helps specified future-state or sensor tests, train the smallest viable module using permitted temporal supervision; report forecasting quality separately from decision accuracy.

**Exit:** a minimal state representation chosen by actual temporal recall and runtime results.

### R6 — Compatible weight integration / structural reduction (**optional**)

- [ ] For any merge pair, verify shape, ordering, architecture, checkpoint genealogy and pre/post-merge source-task functional equivalence. Reject arbitrary averaging across different bases.
- [ ] Where justified, try aligned weights / FS-Merge-like folding with **small validation-safe alignment data**. Count its training and source towers used during merge.
- [ ] Compare merged output to A and B; keep the unmerged model as a reproducible fallback.
- [ ] Measure true resident byte/operation reductions after folding. A theoretical smaller checkpoint is not a deployed model.

**Exit:** H3 and hardware benefit; otherwise document negative result and skip merger.

### R7 — Precision, packaging and runtime (**after quality gate**)

**Read-only size envelope (2026-10-11):** all three main encoder weight files are FP32. A manifest- and hash-checked storage-only estimate for group-wise INT8 (group size 256; one FP32 scale and one INT8 zero point per group) is ~182.0 MB decimal versus the current 687.3 MB package, leaving ~18.0 MB against the provisional 200 MB target. No weights were quantized and no quality, device, kernel, memory, or latency result was measured; this does not close R7. Details and hashes: [Path A size envelope audit](docs/pretrained_reuse/R7_PATH_A_SIZE_ENVELOPE_20261011.md).

- [ ] Profile FP32/BF16/FP16 reference behavior **per backend**, with supported numeric combinations only.
- [ ] Test 8-bit and 4-bit deployment paths and their *actual kernels* before targeting BitNet-style ternary. Quantized storage without low-bit compute may not reduce runtime.
- [ ] If needed, measure architecture-compatible ternary-QAT plus recovery, preserving high-precision exceptions and optimizer/master state for training. This is **optional**, not a first-stage prerequisite.
- [ ] Count all exception tensors, tokenizers, scales, metadata and adapters in actual package bytes; verify export/reload parity and latency on the target device, not just CPU test fixtures.
- [ ] No unverified claim that a 1.58-bit name implies 1.58-bit whole-model storage or better speed.

**Exit:** measured quality, package, memory and latency on an identified backend/device.

### R8 — Locked release evaluation and handoff

- [ ] Freeze candidate architecture, all tensors, model/processor revisions, thresholds and run ledger **before** unsealing the final audit.
- [ ] Run one final untouched evaluation per frozen release candidate; aggregate by independent scene/source, show calibration, coverage, OOD, mixed-media, missing-media and temporal results.
- [ ] Publish package manifest, all parameter/byte accounting, trade-offs, license obligations, offline functional tests and device-specific benchmarks with reproducible conditions.
- [ ] Declare `PASS`, `FAIL` or `UNCERTAIN` separately for every target task and device. Do not use accuracy from unrelated public benchmarks to claim product task success.

**Exit:** release qualification is supported by a reproducible, unmodified audit record.

## 7. First bounded action queue (before any training or GPU purchase)

1. [x] Create `docs/pretrained_reuse/CANDIDATES.md` listing pinned candidates, rights, parameters and runtime checks.
2. [x] Create `docs/pretrained_reuse/EVAL_PROTOCOL.md` with task sources, group isolation, split rules and initial metrics.
3. [x] Prototype **read-only** inspection of a pinned V-JEPA-B checkpoint, verifying key names, image/video forwards, output tokens, frame behavior and parity (`docs/pretrained_reuse/R1_AUDIT.md`). Checkpoint rights remain unresolved.
4. [ ] Prototype **read-only** ViT-Lens original backbone + released audio Lens with published preprocessing. Its published pair exceeds 12.3GB, fails the current compact deployment/storage gate, and is not needed for frozen Path A; defer it unless B has a measured justification.
5. [x] Start Path A frozen-feature text, image, audio, and a bounded video probe with deterministic group splits. Text pooling variants, visual/audio feature reuse, multi-query scoring, and the four-modality dispatcher have been exercised; the dispatcher now has an eight-distinct-asset runtime distribution. Image/audio pooling variants and persistent state-update behavior remain untested, while no approved product-aligned joint-AV source is available. These gaps keep the full R2 exit open.

The reusable candidate-scoring core is integrated into `src/tiny_omni_decision/decision.py`, and the frozen-feature Path A probe runners call it. A typed immutable observation-cache core is exercised by text, audio, image, and CLEVRER diagnostics. A read-only pack/export API has now been exercised on the 94,636-entry text cache; it does not replace the source cache and is not wired into an end-to-end multimodal pipeline. The byte-bounded in-memory LRU is exercised in a CPU CLEVRER feature-retrieval-to-scorer path with exact output parity and bounded eviction. A hash-verified local v4 assembly contains task-scoped Audio, CLEVR-4 Image, synthetic Text, and a CLEVRER Video feature-input scorer. The Video scorer references, but does not duplicate, the VideoPrism checkpoint already stored for Image; its 3,014-choice validation replay has exact prediction parity to the saved readout within floating point tolerance. A one-video cross-environment smoke also ran the fixed 8-frame decoder/VideoPrism CPU profile and then the scorer on CUDA, exactly matching the cached video feature and saved decision to floating point tolerance. The four-modality dispatcher matched saved outputs on eight distinct validation assets per modality; second Video/Image requests reused the same frozen features. Batch-one latency, memory and parity records are in the [single-example smoke](docs/pretrained_reuse/R2_PATH_A_DISPATCHER_20261011.md) and [eight-asset distribution report](docs/pretrained_reuse/R2_PATH_A_LATENCY_DISTRIBUTION_20261011.md). These are runtime/replay results, not new quality results or service-level tail guarantees. Joint audio-video task quality remains unmeasured. The 687.3 MB package exceeds the provisional 200 MB target, and runtime profiles remain separate. The state/query text factorization probe is a separate readout configuration and was slightly below the original text baseline on the same validation IDs. Encoders and experiment orchestration remain research prototypes under `scripts/`. Existing `train-decision`, `configs/decision/durable_teacher.yaml` and `configs/recovery/probability_distillation.yaml` remain legacy Gemma-specific functionality. Do not suggest they execute R1–R7 as written.

## 8. Change-control / supersession map

| Existing artifact / branch | New status | Allowed action now |
| --- | --- | --- |
| `README.md` top-level Gemma description | Historically accurate implementation, **not new strategy** | Add pointer to approved roadmap in a separate documentation change |
| `ROADMAP.md` old "ternary first" contract | **Superseded as future product strategy** by this proposal | Review/approve new strategy before merging |
| `docs/DURABLE_TEACHER.md`, `docs/PHASE3.md`, `manifests/teachers/*` | Historical factual evidence | Preserve intact |
| `docs/DATASETS.md`, allowed-source manifests, source leakage checks | Reusable compliance and dataset infrastructure | Reuse; extend for synchronized media only after rights audit |
| `configs/decision/*` and existing Gemma teacher / EmbeddingGemma QAT implementation | **Legacy experimental track** | Keep reproducible, do not delete or silently repurpose |
| Open PRs #9–#14 and `codex/roadmap-compression-video-specialization` | Concurrent, potentially active and/or stacked research | Review one by one against this new direction; **no automatic merge/close/stop** |
| Proposed A/B/C implementations and R1–R8 | **Not built** | Author specs and bounded work packages after roadmap review |

## 9. References and why they belong here

Primary references and official implementations (paper merit does **not** imply immediately reusable weights):

- [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482), [official V-JEPA source and pretrained models](https://github.com/facebookresearch/vjepa2): small distilled temporal vision starting point.
- [MJEPA](https://arxiv.org/abs/2606.25225), [official code](https://github.com/facebookresearch/MJEPA): evidence for shared audio/video learning and cross-modal prediction, **not** proof of a released suitable small weight file.
- [ViT-Lens](https://arxiv.org/abs/2311.16081), [code/model zoo](https://github.com/TencentARC/ViT-Lens): pretrained ViT adaptation via modality Lenses.
- [LAVISH](https://arxiv.org/abs/2212.07983), [code](https://github.com/GenjiB/LAVISH): frozen visual backbone, parameter-efficient audiovisual adaptation.
- [Perceiver IO](https://arxiv.org/abs/2107.14795), [official implementation](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver): optional query/state bottleneck; **not** an off-the-shelf learned world model.
- [VATT](https://arxiv.org/abs/2104.11178), [official code and model links](https://github.com/google-research/google-research/tree/master/vatt): single-backbone multimodal reference; legacy TensorFlow stack.
- [Model Soups](https://arxiv.org/abs/2203.05482), [Git Re-Basin](https://arxiv.org/abs/2209.04836), [FS-Merge](https://arxiv.org/abs/2410.01483): averaging, alignment and merging under **different compatibility assumptions**.
- [SSAM](https://arxiv.org/abs/2603.21584): specialist merge with shared-base assumptions.
- [BitEmbed](https://arxiv.org/abs/2606.25674), [bitnet.cpp](https://github.com/microsoft/BitNet): low-bit representation and deployment after quality is established.
- [Project durable teacher report](docs/DURABLE_TEACHER.md) and [rights/corpus report](docs/DATASETS.md): actual historical measurements and constraints; not newly validated performance.

## 10. Disposition

This roadmap is a **model-agnostic, evidence-first design proposal**. Keep the full previous roadmap in Git history; it is not deleted from the project record. The requested product outcome stays constant—**small, high-quality, local, genuinely multimodal typed decision inference**—while the mandatory Gemma/EmbeddingGemma ternary-first implementation path is no longer the assumed route. Since this proposal was opened, a task-scoped frozen Speech Commands audio runtime bundle has been integrated on this branch and passed existing-validation replay parity; see the R2 runtime record. It does not close joint AV or full multimodal packaging.

**Next implementation action:** define R0 task/device criteria and R2 task-matched quality gates, including an approved synchronized audio-video task, before new training or product selection. Expand Path A task coverage only with approved sources and report results by task. The current four-component assembly is about 687.3 MB, above the provisional 200 MB target; any size reduction must preserve task-specific outputs and be measured.
