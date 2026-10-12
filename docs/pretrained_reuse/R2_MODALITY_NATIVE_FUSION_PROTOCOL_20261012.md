# R2-M — Modality-native and cross-modal Decision experiment protocol

**Status:** new experiment design, not an executed result (2026-10-12). **Priority: main Tiny Omni Decision**. The independent <=10M audio-only project was **withdrawn by the user on 2026-10-12**; this fusion protocol is the active research direction. This plan corrects the earlier tendency to cast every task as a fixed *text question plus modality features*. The user requests direct decisions from audio, video, images/screens and sensor streams, as well as real joint decisions, in one compact multimodal system.

**Hard product trajectory:** <=200,000,000 total **resident parameters before quantization**, counting every simultaneously resident encoder, stem, vocabulary embedding, fusion/readout module and adapter; seek **95–98% decision accuracy** on predeclared *representative, independently grouped* tests **for every required modality and joint combination**; after quantization/recovery target **>=95%** on those same independently specified tasks. These are unachieved research goals. There is no single meaningful empirical percentage called "world understanding" without an explicitly fixed task distribution, evidence requirements, abstention policy and metric.

## 1. Correct functional contract

The primary operation is **state-conditioned Decision**, not "ask a fixed question in text":

`Decision(
  observations=[{modality, payload, timestamps, source_id, valid_mask, units, calibration}, ...],
  options=[{option_id, typed_action_or_choice_descriptor}, ...],
  criterion? = optional typed/text/spoken condition
) -> {option_ids, probabilities, selected_id?, abstain?, evidence_usage_metadata?}
`

- **Input presence:** Any nonempty subset of {audio, image, screen, video, sensor stream, text} is valid for a supported task, and some tasks **must require zero text input**. Screen is a UI screenshot modality, not assumed to be equivalent to natural photographic images.
- **Input roles:** Audio may carry **environmental evidence** (alarm, impact, footsteps) OR a **spoken criterion** ("which option is nearer?"); video/screens may carry observed state; sensor streams may carry acceleration, orientation, temperature, contact state, etc.; text is **optional context**, not the privileged default observation.
- **Candidate representation:** For fully text-free tests, candidate/action IDs and typed numeric or enumerated attributes are sufficient; no generated textual question, transcript, MiniLM query or hidden label/name is permitted. For user-instruction tasks, optional natural-language or spoken criteria and arbitrary candidate descriptions are a separate evaluation slice. Do not claim text-free reasoning if an unreported mandatory text tower supplies the question.
- **Output:** Stable option IDs and calibrated probabilities in a deterministic typed schema or JSON, with explicit "insufficient evidence / abstain" for unsatisfied or missing conditions. Do not add a language-generation Decoder merely to format JSON.
- **Modality absent:** Missing is different from zero signal (silence, black frame, zero acceleration). Use an explicit modality mask and timestamps, not a zero vector as the only indicator.
- **Safety:** This is a decision *recommendation* output, not automatic high-impact device actuation. Missing sensors, stale observations and low-confidence cases must be detectable.

## 2. Architectural hypothesis, not a finished model

`
Audio samples ─── pretrained lightweight audio encoder ─── temporal tokens ─┐
Video frames ──── pretrained video/world encoder ────────── spatiotemporal ──┤
Image/screen ──── shared visual stem where appropriate ───── spatial tokens ─┤
Sensor streams ── tiny typed/time-aware stem; physical units ── time tokens ─┤
Optional text ─── small pretrained text path ────────────── semantic tokens ─┘
                                     ↓
         Modality tags + true timestamps + validity/missingness masks
                                     ↓
           Small gated / cross-attention / latent-slot FUSION
                  (preserve source-specific information)
                                     ↓
                 Reusable temporally indexed World State
                                     ↓
          Tiny option/criterion-conditioned Decision Readout
                                     ↓
           Normalized option scores / JSON / abstain
`

Prefer frozen pretrained encoders and cached tokens first. Competing baselines must include modality-native linear/MLP heads **without** an extra text Question Encoder. Test raw token, time-preserving pooling and intermediate-layer readouts against global mean pooling. Insert a cheap sensor stem (e.g., calibrated numerical features + temporal convolution / small MLP) before considering a large standalone 35–38M time-series transformer. **The existing ~170.5M modular reference plus an additional ~37.9M MOMENT tower would exceed 200M** if simultaneously resident; do not conceal this count. Preserve a single shared visual backbone for images/screen/video when justified by loaded weights and measured performance. Freeze vs trainable status is separate from parameter count.

A fusion block is **not** assumed to learn from unpaired modality-only data without supervision. Synchronization, real co-occurrence, correct labels, information availability and loss construction are experimental requirements. Joint/shared-backbone weight merging is optional and must not substitute for checking whether fusion actually uses the signals.

## 3. Mandatory experiment matrix and clear controls

| ID | Input to model | Required test (no hidden text unless stated) | Minimum falsification/control |
| --- | --- | --- | --- |
| M0-T | Text alone | Criterion -> typed-option probabilities | Permute options; paraphrase instruction |
| M1-A | Audio alone | Environmental event/state -> decisions, no transcript/question needed | Replace/silence audio; balance classes; unseen recording/speaker |
| M1-V | Video alone | Motion, temporal order, object/event state -> decisions | Reverse frames; shift scene; use static-frame baseline |
| M1-I | Image alone | Spatial relation/state -> decisions | Swap image or relevant regions; unrelated distractors |
| M1-U | Screen capture alone | UI visual state -> selected action/option | Swap screen state; avoid filename/OCR-label leaks |
| M1-S | Sensor stream alone | Threshold, trend, event/state and temporal anomaly -> decisions | Time shuffle; unit/range/calibration shifts; static mean baseline |
| M2-AV | Synchronized audio + video | Evidence-linked event or action decision | Audio-only, video-only, matched vs mismatched pair, timing offset |
| M2-VS | Video/screen + sensors | Sensor/video coincidence, cross-evidence and contradiction | Both singletons, shuffled sensor trace, time shift |
| M2-AS | Audio + sensors | Sound and physical event co-occurrence | Both singletons, sensor replacement, audio replacement |
| M3-AVS | Audio + video + sensors | A task whose gold label truly depends on all required factors | Every singleton, every pair, each missing/replaced stream |
| M3-T | Any of the above + optional text or spoken criterion | New user criterion changes decision for same world state | Same observation / reversed criterion, option permutation |

Every row needs a task-specific label definition, valid option set, a source/license manifest, independently grouped train/calibration/dev/sealed splits and explicit task difficulty. **No single aggregate score may substitute for the per-row evidence.**

### Demonstrating actual fusion rather than concatenation

- **Complementary task:** Construct a controlled, predeclared task requiring two or more signals (e.g., choose an alert action only if video detects occupancy AND audio alarm is active AND contact/temperature sensor is above threshold). Generate balanced factor combinations so that **no unimodal signal, candidate name, timestamp, audio loudness, room ID or prevalence prior trivially solves the label**. The initial synthetic/control task proves wiring and conditional information use only; it is not real-world world-model competence.
- **Redundant task:** Also test a task where either video or audio can legitimately identify the event. The fusion model must match or exceed the better unimodal score or explain a measured degradation.
- **Interventions:** At a minimum evaluate all present/missing modality masks; same-label and conflicting streams; independently shuffled time-aligned streams; shuffled option order with output realigned by ID; event-time offset; irrelevant modality addition; stale sensor timestamps. Hold trained weights fixed during these controls.
- **Losses:** Start with supervised option CE plus optional Brier and separately tuned calibration; train only tiny native heads + small fusion. Add a supervised modality-consistency, missing-modality or cross-modal latent prediction objective **only if** a preregistered diagnostic identifies its need and legally usable synchronized data exist. Do not force entire modality embeddings to be numerically equal; different modalities have unique information.
- **Information accounting:** Log modality-specific gradients/ablations, validity masks, event identifiers and real paired assets. Matched encoders/readouts should be compared at equal evidence and task distributions. Reporting only "joint output is finite" is insufficient.
- **Evidence attribution:** Improvement over the **best** unimodal baseline, with grouped paired confidence intervals and domain-shift tests, is required to claim fusion benefits. In tasks intentionally requiring all signals, demonstrate that removing a necessary signal meaningfully degrades accuracy. Uncertainty from missing modalities should be reflected by calibrated lower confidence or abstention, rather than fabricated certainty.

## 4. Experiment sequence (minimal compute, no premature foundation training)

**M-0 — Task/data protocol and interfaces, NO training.** Define typed outputs, criterion-optional observations, sensor units/sampling/timestamps, no-text test mode and variable candidates. Audit 100–200M budget with sensor included. Select at least one independent labelled/rights-approved task per M1 row and a synchronized multi-signal task for M2/M3; if real paired video+audio+sensor data do not exist, report M3 **NOT TESTABLE**, not completed. Build a small strictly separated deterministic synthetic synchronization dataset for CPU smoke only if useful.

**M-1 — Direct modality-native decision baselines.** Keep current frozen encoders where possible; train tiny audio-only, video-only, screen-only, image-only and sensor-only Decision probes with text question absent. Include source-only M1-T text control. Preserve temporal features; compare pooled vs token/time-aware readout. Sensor-first simple threshold/trend labels should be compared with exact algorithmic baselines; succeeding at deterministic thresholding is not evidence of a learned world model.

**M-2 — Gated crossmodal fusion baseline.** Build a small time/mask-aware shared fusion over frozen features, first M2-AV and M2-VS/M2-AS when paired sources exist; then M3-AVS. Train on matched event-aligned examples; do not replace audio with text transcripts behind the scenes. Compare with concatenated mean-pool MLP and single-modality best. Save synchronized asset hashes, time offset, sensor calibration and event grouping. Evaluate model reliance with fixed modality interventions.

**M-3 — Conditional questions optional.** On the same encoded state, compare no criterion, typed criterion, text criterion, spoken criterion. The model should choose different options when a valid criterion changes and should not rerun the expensive video encoder solely because the criterion changes. This stage must **not turn text into a mandatory entry point** for the M1/M2 tests.

**M-4 — Capacity and quality gate.** Use new, independently grouped held-out tasks with representative event/condition diversity. Require every required modality and joint task to be measured, not covered by a text-conditioned proxy; report accuracy, balanced/macro accuracy, NLL, Brier, ECE, abstention risk/coverage and uncertainty. **Pre-quantization target 95–98%**, product not claimed until thresholds are met. Model and checkpoint selected on development data only, sealed audit untouched.

**M-5 — Quantization and measured recovery.** Only after the unquantized source passes the corresponding quality gates, test a deployable precision (8-bit, 4-bit, eventually ternary QAT where supported). Quantization-recovery goal: **>=95% on the same preregistered M1/M2/M3 task categories**, no systematic modality disappearance, no hidden switch to FP32 encoders. Measure real package bytes, total resident RAM and input-to-decision p50/p95 on target hardware and compare within exactly the same candidate architecture. Quantization never changes the parameter-count ceiling.

**M-6 — Generalization and audit.** Test new devices, sound conditions, lighting, rooms, frame rates, sensor-unit scales/calibration, missing channels and asynchronous updates. Freeze the model and protocol before the final audit. If broad 95–98% "world understanding" is unachieved, report specifically which tasks are unsupported.

## 5. Concrete work items for implementation agents

- [ ] Implement a versioned `ObservationBatch` protocol: variable present modalities, true timestamps, sensor value units/ranges, valid masks, stable source IDs, no-text sentinel and candidate/action IDs.
- [ ] Extend the existing native feature-cache API with a text-free single-modality Decision runner that consumes typed options and never instantiates MiniLM unless the experiment opts in.
- [ ] Add 5 text-free M1 smoke fixtures plus one M2 synchronized fixture; test that content/label files do not leak via filenames, option IDs or sampling identifiers.
- [ ] Add sensor numerical/time-series stem prototype with <1M params as an initial **budget hypothesis**, measured against simple deterministic rule baselines and an audited/pretrain-overlap-safe sensor dataset.
- [ ] Add fusion MLP control, gated/time-aware token fusion and explicit modality masks. No need for expensive new Transformer pretraining.
- [ ] Make one complete factorial intervention evaluator for M1/M2/M3 (drop/replace/shuffle/offset signals; candidate permutation) with sealed-split safeguards.
- [ ] Add per-cell accuracy/NLL/Brier/ECE/coverage/balanced-metrics reports and deduped all-encoder parameter/residency ledger. Per-cell preregistered acceptance thresholds, sample-size planning and grouped intervals required before model promotion.
- [ ] Add frozen-vs-quantized parity and >=95% quality-regression evaluation when a BF16/FP16 candidate first passes.

The initial work order is **M-0 -> M-1 -> M-2**, stopping for missing rights, label correctness or resource limitations. This document authorizes **no paid GPU, external data-license acceptance, interference with live worktrees or sealed audit access**. Do not overwrite historical outputs. Implementation is separate from this doc-only organization change.

## 6. Current measured boundary (not new claims)

- Prior frozen text/image/audio/video probes generally use a **text-encoded question/candidate pathway**. They do not establish modality-native no-text Decision performance.
- STARSS23 did **test synchronized audio/video**: validation macro event NLL **0.2704 audio-only**, **0.4108 video-only**, **0.3290 audio+video**; the joint model was worse than audio-only. Two validation rooms, imbalanced event labels and small head mean this is **negative evidence for the tested setup**, not a blanket disproof of fusion. See [STARSS23 report](R2_STARSS23_COMPARISON_20261011.md).
- Sensor prior is a pinned MOMENT-1-small **embedding smoke only**: a 6-channel/128-sample tensor returned finite tokens; **no sensor-to-Decision labels, calibration or quality evaluation**. Using that entire additional ~37.9M parameter tower with the existing ~170.5M reference would overrun the 200M limit. See [sensor audit](R1_MOMENT_SENSOR_20261011.md).
- The recent "joint vs independent" text readout experiment concerns **joint candidate/text token scoring** (not multimodal audio-video-sensor fusion); do not reuse its "joint" performance as proof of multimodal capabilities. See [joint readout report](R2_JOINT_LEARNING_RESULTS_20261012.md).
- No loaded five/six-modality <200M system, text-free sensor decision, AVS fusion quality or quantized >=95% product has been established by these records.

**Independent audio-only <=10M plan: WITHDRAWN, not paused.** It is an archived historical document only; no separate audio-only training, release or budget gate is authorized by that plan. Continue the main 100–200M Omni model with audio as a native modality (and text/image/video/screen/sensor capabilities and their fusion). Do not delete historical artifacts.
