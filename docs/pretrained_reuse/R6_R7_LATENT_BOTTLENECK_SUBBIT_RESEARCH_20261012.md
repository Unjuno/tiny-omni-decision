# R6/R7 research note — Decision-sufficient latent bottlenecks, pruning, and sub-1-bit factorization

**2026-10-12. Status: option under investigation; not adopted and not an experiment result.** Main target remains one pretrained-reuse, full-Omni audio/video/image/screen/text/sensor Decision model with <=200M underlying resident parameters, condition-free/native and optional-conditioned decisions, genuine crossmodal fusion, target 95–98% quality pre-compression and >=95% on the same locked tasks after quantization/recovery. The withdrawn separate <=10M audio model stays inactive.

## 1. User-proposed hypothesis, precisely separated

A stronger pretrained branch processes each observation and a much narrower, lightweight MLP/token bottleneck learns a **decision-sufficient state**. After function-level supervision, irrelevant token/channel/head/layer capacity may be pruned with selective fine-tuning and the remaining large linear layers reparameterized and/or quantized.

This idea decomposes into three **different** operations:

| Operation | What shrinks | What does NOT automatically shrink |
| --- | --- | --- |
| Learned MLP/TokenLearner/Perceiver-style bottleneck \`z=B(H)\` | Activation/state token volume, later fusion/readout computation, persistent cache | The original input Encoder's parameters or its cost **if still run** |
| Structured width/head/FFN/layer pruning | Resident parameter tensors and often FLOPs/latency (if actual dimensions/modules are removed) | Does not guarantee retention of video motion, modality information or generalization |
| Low-rank binary factorization (LittleBit/NanoQuant) | Stored weight bits and possibly inference arithmetic for supported kernels | Does not itself validate Decision accuracy, preserve arbitrary input features, or provide mobile-compatible kernels |

A small MLP output is not proof of information preservation. It can discard information required by unseen criteria and by rare but important sensor/audio/video events. If the bottleneck reads a 150M teacher Encoder on every device query, it **does not constitute** a deployable 50M standalone model. To remove the teacher, build or structurally reduce a compact raw-input student and verify teacher-free end-to-end execution.

### Hypothesis architecture

\`\`\`text
TRAINING ONLY — frozen/pretrained teacher (all required modalities)
  input --> teacher modality tokens/world state --------------------+
                    |                                              |
                    +--> small temporal/multimodal bottleneck       | teacher option scores,
                             (16/32/64 slots × 256 dims)           | paired event probes
                                      |                             |
                                      v                             v
                   student raw-media stems/backbone --> native/fused scorer
                            [width/depth reduced or reparametrized]
                                      |
                   task CE + optional teacher option KL + temporal
                  invariance + source-specific / crossmodal controls

DEPLOYMENT ONLY — teacher entirely absent
  raw audio/video/image/screen/sensor/optional text
    --> compressed student encoders + bottleneck + tiny choice head
    --> stable option probabilities / optional abstention
\`\`\`

The 16/32/64×256 slot sizes are **initial ablation proposals**, not proven memory requirements. Alternative: keep full pretrained backbone and compress only intermediate/persistent-state tokens, **without claiming** parameter shrinkage.

**Distillation safeguard:** Use supervised real labels and modality-native/fusion intervention scores as the primary target. Use teacher KL only on tasks where the teacher is demonstrably stronger and option IDs/order match. Never mandate matching full hidden-state coordinates from unrelated pretrained encoders; the project previously observed a failed embedding-level self-distillation approach. Detect concept collapse with same scene + changed criterion, shuffled modality, frame reversal, time offsets, rare positive labels and newly supplied choice descriptions.

## 2. Samsung's actual 0.1-bit research (not "0.1 megabit")

### LittleBit (NeurIPS 2025) and LittleBit-2 (ICML 2026)

- Samsung Research: [LittleBit](https://arxiv.org/abs/2506.13771); [LittleBit-2](https://proceedings.mlr.press/v306/lee26y.html); [official implementation](https://github.com/SamsungLabs/LittleBit).
- Original wide weight \`W\` becomes a low-rank product of two **binary factor** matrices (±1) plus row/column/rank scales, often a compensated residual path. The number called **bits per original dense weight (BPW)** is **amortized** over the original matrix, not a fractional physical encoding of each ±1 variable. The architecture's stored number of bits and factor rank are coupled.
- LittleBit: Dual-SVID initialization + QAT. LittleBit-2: internal latent rotation and Joint-ITQ initialization before similar inference-time factors. More stable at extreme compression than original LittleBit in evaluated LLMs.
- Concrete degradation caution: Samsung's original LittleBit blog reports Llama2-7B WikiText-2 perplexity about 5.47 full precision versus **15.92 at nominal 0.1 BPW**, not a fidelity-preserved 95%-accurate decision model. Its Llama2-13B FP16 storage 26.06GB shrinks to 0.84GB at 0.1 BPW (~31×), much larger than the naive 13B×0.1/8 theoretical storage due to uncompressed components and overhead. Extreme rate alone is not a product result.
- **License:** SamsungLabs/LittleBit repository source is **CC BY-NC 4.0**. Treat commercial deployment as unavailable without separate permission; use for restricted research/design inspiration only under applicable terms. Code explicitly names decoder-only LLM architectures, not JEPA ViT/video/audio/sensor encoders.

### NanoQuant (ICML 2026)

- [Paper](https://proceedings.mlr.press/v306/chong26a.html), [Samsung explanation](https://research.samsung.com/blog/NanoQuant-Efficient-Sub-1-Bit-Quantization-of-Large-Language-Models), [official code](https://github.com/SamsungLabs/NanoQuant).
- PTQ-style method (quantization/calibration, not new foundation pretraining) combining low-rank binary factors, KFAC-informed calibration, ADMM initialization, sequential block error reconstruction and scale-only global logit alignment. Its published 70B compression takes 13h on **one H100**; "PTQ" does **not** mean negligible cost.
- Official repo supports specific LLM decoder families and specialized **CUDA** prefill/decode kernels; it does not yet prove V-JEPA, VideoPrism, audio ConvNets, CoreML or smartphone hardware support. Porting encoder attention layouts and retaining video temporal representations is a separate implementation/research task.
- **License:** official SamsungLabs/NanoQuant source is **Apache-2.0**, subject to licenses on any reused foundation weights or dependencies.

### Quantitative matrix-rank check (illustrative theoretical budget)

For \`W\` of shape \`m × n\`, one binary factor path with rank \`r\` stores \`U(m×r)\` and \`V(n×r)\`, and FP16 row/column/rank scales. Ignoring residual paths, alignment, file metadata and packing alignment:

\`\`\`text
Stored bits ≈ r(m+n) + 16(m+n+r)
Effective BPW ≈ [r(m+n)+16(m+n+r)]/(mn)
\`\`\`

To target **0.1 effective BPW** under these optimistic assumptions:

| Matrix W | Implied approximate maximum r | Relative to min(m,n) |
| --- | ---: | ---: |
| 384 × 384 | 3 | 0.8% |
| 768 × 768 | 22 | 2.9% |
| 3072 × 768 | 45 | 5.9% |
| 1536 × 1536 | 60 | 3.9% |

This calculation **illustrates** why deep 0.1-BPW compression can be especially destructive for small transformer widths and modality-specific projections. It is **not** a measured LittleBit/NanoQuant layer configuration or quality prediction. Residual factors consume extra budget and can further reduce per-path rank; model-specific layer allocation is required. At such ranks it is entirely possible to remove information needed by temporally precise or minority-class decisions.

The pure (unrealistic) 0.1-BPW theoretical *weight* storage for 50M/100M/200M original dense parameters would be **0.625/1.25/2.50 MB**, respectively. This is **not** the real complete model, packaging, runtime RAM, activation, codec or metadata footprint and must never be advertised as a deployed-model size.

### Alternative size envelopes

| Original model | BF16 dense weights | 4-bit dense codes | Ideal ternary entropy (log2(3) BPW) |
| --- | ---: | ---: | ---: |
| 50M | 100 MB | 25 MB | 9.91 MB |
| 100M | 200 MB | 50 MB | 19.81 MB |
| 150M | 300 MB | 75 MB | 29.72 MB |
| 200M | 400 MB | 100 MB | 39.62 MB |

Decimal MB, excluding actual scales, exceptional tensors, activation memory, tokenizer, buffers and file/runtime overhead. Ternary quantization changes bit representation, **not** the model's count of effective dense parameters by itself. Factorized binaries explicitly change how matrices are represented/stored; record both dense-equivalent and physical stored variables.

## 3. Related capacity-compression evidence

- [Google TokenLearner](https://research.google/blog/improving-vision-transformer-efficiency-and-accuracy-by-learning-to-tokenize/) learns a few space-time tokens to shrink downstream vision transformer computation. Reusing its principle for a **fusion/state bottleneck** is plausible but is not yet validated for JEPA temporal relations or native sensor/audio decisions. Inserting it **after** the expensive pretrained Encoder does not speed up that Encoder.
- [Structural Pruning of Large Vision Language Models (IJCV, 2026)](https://link.springer.com/article/10.1007/s11263-026-02865-5) reports differences between widthwise and layerwise pruning, and low-data recovery (5% of original training data for *some moderate-ratio settings*). Its "95% retained" statements concern **relative performance retention**, NOT absolute 95% Decision Accuracy or success for a 100–200M JEPA-class Omni model. Aggressive pruning beyond 50–60% can be much harder to recover.

## 4. Proposed gated ablation sequence; no automatic training launch

1. **B0 — First establish a capable unquantized Omni reference**, with native audio/image/video/screen/sensor and mixed-modality tasks including text-free conditions. If the current text/video/sensor/fusion quality is below target, no compressor can be credited with preserving a 95% baseline that does not exist. Use rights-approved independently grouped evaluation and no option-label leakage.
2. **B1 — Bottle the *state*, not the model**, while Encoder weights are frozen. Compare mean pooling against 16/32/64×256 content/time/mask-aware slots, and preserve lightweight modality-private residuals. Train a small MLP/cross-attention readout on ground truth plus bounded teacher decision KL where teacher is competent. Check whether unseen criteria, frame reversal and shuffled audio/sensors still change answers correctly; loss of necessary event evidence is a fail.
3. **B2 — Structured pruning only after B1 works.** Create separate reproducible 10%/20%/30%/40%-pruned student candidates. Test widthwise attention heads/FFN neurons versus selected layer removals, with a *real* reduced model file and forward, not masks retaining all dense weights. Preserve first/last stems, critical synchronization/time pathways unless held-out ablations show they can be cut. A 170.5M reference → 50M is ~71% parameter removal; this is a research stretch target, **not** an assumed low-cost step.
4. **B3 — Limited quality recovery** on existing ground-truth decision tasks + selective option-KL + task-relevant event/time latent tests. Report accuracy, macro/balanced metrics, calibration, modality-dropout effects, compute, and overfit. Reject each compressed architecture if its predeclared quality floor fails. Do **not** re-fit on sealed audit or compensate by changing to easier tasks.
5. **B4 — Layerwise low-bit portability ladder:** 8-bit → 4-bit → 2-bit/ternary (~1.585 theoretical) → optional NanoQuant around 1.0/0.5 BPW → optional LittleBit-2 around 0.1 BPW as **extreme, potentially destructive reference**. All on the **same** precommitted modality/fusion battery; protect sensitive stems/embeddings and measure every higher-precision exception. If a precision tier fails >=95% quality, fall back to the last valid tier. Sub-1-bit conversion **requires porting and actual low-bit kernels**; CUDA speedups from Samsung's LLM decoding do not establish smartphone video/audio latency.
6. **B5 — End-to-end device audit:** Teacher absent at runtime, load raw modalities, emit candidate probabilities; physically verify unique resident parameter/factor counts, model bytes, peak RAM/VRAM/activations, cache keys, latency distribution and energy/thermal conditions. Compare baseline vs pruned vs each quantized package. Never claim "20–30M parameters after quantization" unless an actual structural reduction is demonstrated.

### Success decision

Retain the smallest **actual end-to-end** raw-input-to-Decision model that meets independent per-modality **and fused** >=95% accuracy after quantization, with confidence intervals, calibrated probabilities and target-device resource constraints. A compression-only number or strong original LLM benchmark score is insufficient. If the result cannot be reproduced, classify as **UNCERTAIN**.

**Scope / provenance:** This is a research-protocol document, no weights, datasets, jobs, existing model implementations, legacy PRs or sealed tests were modified. The withdrawn standalone 10M audio-only proposal remains archived; the sole priority is the main Omni target.
