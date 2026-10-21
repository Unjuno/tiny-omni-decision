# ARCHIVED — Standalone Audio Decision (withdrawn 2026-10-12)

> **WITHDRAWN by the user on 2026-10-12. This is NOT an active project, milestone, experiment queue, or delivery commitment.**
> The separate <=10M-parameter / 98%-accuracy audio-only model is canceled as an independent research direction. Do not start new training, model-selection, quantization, dataset acquisition, benchmarking, or packaging in response to the archived tasks below.
> **Only the main 100–200M (hard maximum 200M) multimodal Omni Decision project is prioritized**, with audio as one of its required modalities alongside image, screen/video, text, and sensors. The main model targets 95–98% quality before quantization and >=95% on the same representative tasks after quantization and recovery, all currently unachieved.
> This record is preserved for historical traceability; previous experimental results remain valid only for their original scoped tasks. Do not delete checkpoints, samples, or code as a side effect. Nothing here instructs changes to live jobs.

## Historical proposal below — superseded, not actionable

# Standalone Audio Decision — pretrained ~8M audio representation, <=10M total, 98% target

**Status:** independent research proposal (2026-10-12). This document defines a **separate project** and does not change or depend on Tiny Omni Decision, V-JEPA, audiovisual fusion, or the parent roadmap's completion. This is not a claim that a qualifying model already exists.

## Mission

Develop a locally runnable **voice-conditioned option-selection model**. A user speaks a question, priority, constraint, or instruction; the system chooses or ranks **arbitrary currently supplied candidates** and returns calibrated probabilities over the actual options, without generating long-form prose.

Example: A user speaks "この一覧で、一番安くて駅から近いところを選んで" while the application supplies candidate records with names, descriptions, prices and distances. Another spoken instruction may ask for the quietest, earliest or non-smoking option. **The output must change with the spoken criterion**, even when the candidates remain identical; it must also respond to changed candidate attributes and candidate ordering.

The intended input contract is:
- Raw speech audio (with language, sample rate, time/length/missing-data metadata).
- An independent candidate set of **2–N variable options**, each with stable identity, available text and/or structured attributes, and a defined mechanism for unavailable attributes.
- A scoring policy for multiple satisfied requirements, ties, uncertain evidence, and refusal/abstention when no available option satisfies a hard condition.

The model **does not** need to be merged into Omni or used as an Omni audio Encoder. The only relationship is historical research provenance: both may reuse separately permitted tools and papers. Do not count a successful audio result as progress toward full Omni quality.

## JSON interface contract — the complete independent voice-to-choice product

**Input:** Local application supplies raw microphone audio and an arbitrary *runtime-provided*, variable-sized candidate list. Each candidate has a stable string ID and optional typed fields / display name / description. The interface must not embed a fixed label taxonomy or rely on a pre-enumerated set of options. For a 10M model, distinguish **typed numeric/enum field predicates** (reasonable early scope) from unrestricted free-form document semantics (research target, not already solved).

**Output:** One well-formed, versioned **JSON object** per request, with no prose generation. The neural model can produce speech-derived criterion representations and option logits; a deterministic, locally implemented formatter serializes these values to JSON. The formatting code need not be a language model and contributes no neural parameters. Ensure keys and array ordering are stable; the option IDs must exactly refer to supplied candidates and probability values must be finite and normalized across eligible supplied choices when \`status="selected"\`.

Example application input (illustrative; not an achieved benchmark):

\`\`\`json
{
  "request_id": "demo-001",
  "audio": {"source": "microphone", "sample_rate_hz": 16000, "language": "ja"},
  "candidates": [
    {"id": "A", "name": "Option A", "distance_to_station_m": 420},
    {"id": "B", "name": "Option B", "distance_to_station_m": 180},
    {"id": "C", "name": "Option C", "distance_to_station_m": 650}
  ]
}
\`\`\`

Spoken audio: 「駅から一番近い候補を選んで」

Example structured response, **illustrative uncalibrated values**:

\`\`\`json
{
  "schema_version": "1.0",
  "request_id": "demo-001",
  "status": "selected",
  "selected_option_id": "B",
  "criterion": {"field": "distance_to_station_m", "operator": "min"},
  "options": [
    {"id": "A", "probability": 0.03},
    {"id": "B", "probability": 0.96},
    {"id": "C", "probability": 0.01}
  ]
}
\`\`\`

The **98% research target measures correct choices on unseen spoken conditions and candidate sets**, not JSON syntax-validity alone and not Smart Turn's original turn-end-detection accuracy. Report separately (a) speech criterion understanding, (b) candidate/field grounding, (c) choice correctness, (d) probability calibration, (e) JSON schema conformance, and (f) raw-audio-to-JSON latency. A speech-only classifier that picks a fixed label irrespective of supplied candidates fails the product contract.

**Robustness:** Reject unknown candidate IDs, duplicate IDs, missing required fields, nonfinite logits/probabilities, malformed audio, schema mismatch, unsupported language or unresolved criteria. Use \`status="abstain"\` with \`selected_option_id=null\` where no reliable answer exists; abstention must not silently select an invalid or unavailable option. Explicit confirmation is a UI concern for high-impact or irreversible actions. No private audio or transcript needs to be included in the returned JSON by default.

**Interaction distinction:** A mouse directly addresses screen coordinates. This project aims at **voice-conditioned semantic pointing** over an application-supplied candidate context. Its benefit depends on the actual supported criterion vocabulary, errors, speaking burden, environment noise, latency, privacy and confirmation workflow. Claims that it replaces mouse interactions or understands unrestricted spoken requests require separate evidence.

## Hard research targets and honest evidence rules

1. **Complete standalone model <=10,000,000 unique, loaded resident parameters before quantization.** Include the acoustic Encoder, all vocabulary/token embeddings, any language/option Encoder, adapters, pooling, scoring, and trained normalization. A model with 8M acoustic weights and an external 17M MiniLM **fails** this target. Non-neural application data, code and deterministic comparison operators have no model parameters, but their functional restrictions and memory/latency still must be reported.
2. **98% Accuracy target** on each task in a **predeclared, independently grouped, representative voice-criterion selection benchmark**; separately report macro/balanced accuracy, NLL, Brier, calibration and abstention risk/coverage. This is **a target, not an observed result**. The contract's breadth must be predeclared: a closed vocabulary of keywords, a single end-of-turn task, or a heavily imbalanced binary event set cannot substitute for arbitrary spoken-criteria selection.
3. **Minimal training first**: reuse a real released pretrained acoustic model, freeze its weights, and train only the smallest justified conditional scorer. Do not presume new world-model pretraining, Gemma Teacher, broad embedding MSE, or model merging.
4. **Offline local inference**, verified from raw audio and the user-supplied candidates, not from an external ASR/LLM/API. Distinguish cache-hit and live raw-audio latency; measure batch 1 and supported precision on target hardware. Quantization may reduce package bytes but does not reduce parameter count.
5. **Fail closed**: If general Japanese/free-form speech and arbitrary candidate descriptions cannot be solved in <=10M, report limitations and experiment-specific failure. Do not silently redefine the task as endpoint classification or raise the model size.

## Source candidates and the pretrained-latent hypothesis

### S: Pipecat Smart Turn v3 (~8M)

- Official code: https://github.com/pipecat-ai/smart-turn
- Official released weight package: https://huggingface.co/pipecat-ai/smart-turn-v3
- Official upstream describes ~8M model parameters, Whisper Tiny **Encoder**-based architecture, and ~8MB INT8 ONNX / ~32MB unquantized export; model card lists BSD-2-Clause. **These are published approximate figures; loaded tensor counts must be verified.**
- **Original training task is end-of-utterance / turn-completion detection (semantic VAD).** Its accuracy and latent information cannot be assumed to support spoken instructions or general speech transcription.
- Technical gate: inspect whether the released ONNX offers hidden-state outputs (not merely its completion score); if not, reproduce the original Encoder and intermediate features from verified released weights, with numerical parity to the intended checkpoint. Do **not** initialize a fresh Whisper Encoder and call it pretrained. Confirm Japanese phonetic coverage separately.

### E: EfficientAT lightweight acoustic representations

- Official code and pretrained release: https://github.com/fschmid56/EfficientAT
- Candidate `mn10_as` has ~4.88M parameters, while `dymn04_as` is ~1.96M (official model-family figures; recount actual weights). These are audio event/tagging systems, **not** automatically suitable for decoding spoken criteria. Their AudioSet mAP is not 98% Decision Accuracy.
- A smaller encoder leaves more budget for a local criterion/option Encoder, but may lose crucial linguistic details. Test retained phonetic, temporal and semantic information before selecting it.

**Additional candidate search must prefer released small speech-semantic/ASR Encoder checkpoints**, including Japanese if in scope, with rights, exact revision, load/reconstruction and latent extraction verified. Smart Turn and EfficientAT are *initial alternatives*, not selected solutions. A tiny end-of-turn model need not be the best base for spoken commands.

## Complete <=10M architecture hypothesis

```text
spoken audio ----------> verified pretrained acoustic Encoder (<~8M)
                              |
                  temporal acoustic features / masks
                              |          query/criterion derived from speech
                              |          (NO external ASR/LLM required)
candidate text + typed attributes
        |                     |
small *local* option/value representation & operators
        +---------------------+
                              |
             small criterion-conditioned scorer
                              |
       option-ID-aligned logits, calibrated probabilities,
          no-match/unknown handling where specified
```

**Initial allocation proposal**, subject to instantiated weight accounting:

| Component | Parameter allocation |
| --- | ---: |
| Pretrained acoustic Encoder | <=8.0M |
| Candidate text / attribute / criterion representation, including embeddings | <=1.5M |
| Conditional fusion, scorer, pooling, adapters and calibration | <=0.5M |
| **Total** | **<=10.0M** |

A deterministic typed-attribute comparator can handle *known structured* predicates (minimum price, maximum distance, hard exclusions, simple threshold comparison) once the spoken utterance has been correctly mapped to a criterion. This helps avoid teaching a tiny neural model arithmetic from scratch. **It does not constitute general arbitrary natural-language understanding**: tasks that require free-form semantic interpretation and open-vocabulary candidate text must be assessed separately. Any mandatory external text or speech model disqualifies the standalone <=10M claim.

Preserve temporal tokens or intermediate layer features. Compare their conditional utility against global pooling and the original pretrained task output. Keep one audio observation cache independent of questions; repeating a spoken question on *new audio* legitimately requires new encoding.

## Research phases

**A0: Reference audit, no training.** Pin released S/E (and better task-matched speech checkpoints if found) to exact revisions/hashes/rights. Recount all tensors and actual exported weights; verify original task output, candidate hidden-feature access, sample-rate/language/time-length behavior, and raw-audio processing. Define Japanese/English target language scope and representative categories *before* model ranking. Require an explicit option schema for the first benchmark, with a separate free-form text option track.

**A1: Frozen-latent feasibility.** For the same audio, train tiny probes that predict semantic criterion/operator and compare to a scorer that jointly reads acoustic features and candidate representations. Evaluate minimum/maximum, hard thresholds, negation, conjunction, swapped priorities, changed candidates, never-seen option names, and options reordered. Controls: silence, audio from a different request, criterion reversed, text-only/default majority, pooled versus time-token features. Use held-out speakers, prompts/templates, and candidate collections. If the Encoder cannot preserve spoken content, stop and choose a better pretrained Encoder rather than iterate arbitrary heads.

**A2: Small conditional Decision training.** Freeze all reusable Encoder weights initially; use labeled option decisions and criteria; train <=2M combined non-audio parameters. Optional selective unfreeze / LoRA only after an explicit diagnosed bottleneck and within **10M total**; existing weights becoming trainable do not increase the count, but optimizer/activation cost is still measured. For criterion extraction with a typed schema, compare a non-generative operator/slot parser versus direct option-scoring. Handle unsatisfied hard conditions by abstention. Record per-task micro/macro quality, minority-criterion coverage, sample counts and splits.

**A3: Independent runtime.** Package one self-contained offline audio-to-option model and processor, with no parent Omni runtime imports or mandatory data services. Verify feature cache, option-order invariance, numerical FP32/INT8 parity, exact model count <=10M, package bytes, batch-one and batched wall latency, and peak memory. Report cold/warm conditions and supported language/preprocessing; do not extrapolate published Smart Turn VAD timing to the new task.

**A4: Untouched target assessment.** Freeze code, checkpoint, tasks, criteria and thresholds before opening sealed evaluation. Score variable candidates from new speakers/prompts and separate candidate domains, including hard negatives. Report 98% pass/fail **separately for every promised use case** and its confidence interval. If results are below the stated goal or data insufficient, publish the limitation; no retroactive simplification or evaluation leakage.

## Historical record (not evidence of the new goal)

- A prior **55.49M** Whisper Tiny + MiniLM + decision-head stack achieved **97.85% development macro Accuracy** over four Speech Commands-derived question types; three question types were directly constructed from ten keyword labels. This shows a *narrow audio task can be learned quickly*, not arbitrary voice-conditioned selection or <=10M capability: https://github.com/Unjuno/tiny-omni-decision/blob/docs/pretrained-reuse-roadmap-20261010/docs/pretrained_reuse/R2_PATH_A_PROBES.md
- The unchanged keyword model performed near four-choice chance on LibriSpeech transcript/distractor transfer; absence of general spoken-condition handling is a measured risk: [transfer report](../pretrained_reuse/LIBRISPEECH_HARD_NEGATIVE_TRANSFER_20261011.md).
- Synchronized STARSS23 audio-only models were stronger than the tested joint A/V scorer; this **independent audio project is not required to perform or improve audiovisual fusion**: [STARSS23 report](../pretrained_reuse/R2_STARSS23_COMPARISON_20261011.md).

## Research deliverables, safety and scope

Deliver an independent tiny-Audio-Decision package, a frozen pretrained-checkpoint inventory, feature extracts/probes, train+calibration+validation+sealed split manifest, trained scorer with hashes, full-residency parameter and latency audit, and predeclared 98% quality verdict. Keep its branch/config/artifacts independent. Neither a successful keyword scorer nor a ~8M end-of-turn model alone closes the deliverable.

Historical Omni experiments, PRs, source data and checkpoint artifacts are read-only. No paid compute, live process manipulation, or benchmark unsealing is authorized merely by publishing this document. **This is a standalone plan, not an executed training result or a requirement to integrate into Omni.**
