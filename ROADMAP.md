# Roadmap — Pretrained-Reuse Omni Decision (proposed 2026-10-10)

> **Status: architectural redirection proposal, documentation only.**
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

- [ ] Archive a machine-readable inventory of `main`, current independent branch/PR revisions, benchmark/data manifests and experiment checkpoint hashes without modifying their bytes.
- [ ] Reconcile each PR's actual state and dependencies; flag which belongs to obsolete Gemma-first exploration, which contributes reusable validation/quantization code, and which has a live process. **Do not close/stop anything automatically.**
- [ ] Approve the size, deployment device, task-specific quality and time/money criteria before training decisions; retain the historical >=95% product aspiration only as a specified challenge target.
- [ ] Publish candidate matrix, evaluation plan and separate environment contracts.

**Exit:** frozen legacy baseline and written constraints; no historical evidence altered.

### R1 — Pretrained checkpoint feasibility audit (**no training**)

- [ ] Verify V-JEPA 2.1-B exact upstream checkpoint URL, revision, SHA, license, encoder/predictor split, single-frame and video input behavior, intermediate token output and actual total params. **Known loader hazard:** upstream [issue #149](https://github.com/facebookresearch/vjepa2/issues/149) reports mismatches in distilled ViT-B/L predictor/loading setup; verify the `ema_encoder` source, strict tensor shapes and exact intended continuation mode rather than assuming the training predictor reloads or can be reused as a compact inference head. Pin source and checkpoint independently and inspect the hub download URL.
- [ ] Inspect ViT-Lens released backbone/Lens pairing, PE Core tiny/small/base, EfficientAT, small Japanese/English text checkpoints, VATT-MA and any relevant MJEPA *released* checkpoint. Mark `code only` / `pretrained and accessible` / `unknown` distinctly.
- [ ] For each candidate, test preprocessing, loaded forward, shape, supported precision, memory and license. Do **not** assume ViT-Lens Lens weights load into V-JEPA, or that model names state their true parameter counts.
- [ ] Create compact source-of-truth inventory with `passed_load`, `license_allowed`, `has_real_weights`, `runtime_supported`, parameter bytes, source revision and primary citations.

**Exit:** at least one executable, legally usable video candidate and one viable small text/audio path; otherwise pause architecture selection.

### R2 — Frozen modular baseline A (**main low-compute milestone**)

- [ ] Cache approved, **unaugmented deterministic** frozen per-observation token/intermediate features, separately for train/validation; check byte-for-byte repeatability and invalidation.
- [ ] Compare raw token, middle-layer token, mean/attention pooled token readouts with a variable-option lightweight head. Use per-modality and joint-fusion controls; record actual head parameter count and training budget.
- [ ] Test many questions on one cached visual/audio event; count avoided encoder executions, memory residency and latency, not merely Accuracy.
- [ ] Report by source and independent scene. Compare against the preserved old Gemma Teacher as a **reference only**, not an automatic distillation source.

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

1. Create `docs/pretrained_reuse/CANDIDATES.md` listing real model/checkpoint/processor revisions, licenses, params and basic runtime checks. **Proposed file, not yet implemented.**
2. Create `docs/pretrained_reuse/EVAL_PROTOCOL.md` fixing tasks, independent scene IDs, train/calibration/validation/sealed-test splits and initial acceptance thresholds.
3. Prototype **read-only** inspection of a pinned V-JEPA-B checkpoint, verifying checkpoint key naming, loaded image/video forward, output token dimensions, frame/resolution behavior and numerical parity.
4. Prototype **read-only** ViT-Lens original backbone + released audio Lens, with published preprocessing. Do not graft it into V-JEPA until this passes.
5. Only after (3–4), evaluate Path A cached-features decision probes with deterministic train/validation splits; print exact time/memory/bytes and reproducible source hashes.

The new pipeline is **not already implemented** in `src/tiny_omni_decision`. Existing `train-decision`, `configs/decision/durable_teacher.yaml` and `configs/recovery/probability_distillation.yaml` remain legacy Gemma-specific functionality. Do not suggest they execute R1–R7 as written.

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

This roadmap is a **model-agnostic, evidence-first design proposal**. Keep the full previous roadmap in Git history; it is not deleted from the project record. The requested product outcome stays constant—**small, high-quality, local, genuinely multimodal typed decision inference**—while the mandatory Gemma/EmbeddingGemma ternary-first implementation path is no longer the assumed route.

**Next decision before implementation:** approve/adjust the program-level parameter/quality/device constraints and select R1 checkpoint auditing, rather than committing to a new training run.
