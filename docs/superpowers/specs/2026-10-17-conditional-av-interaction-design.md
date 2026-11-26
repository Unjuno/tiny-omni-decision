# Conditional audio/video interaction: experimental readout

## Intent and scope

The user proposes bidirectional audio/video exchange followed by gates conditioned
on the question, candidate and observations. Preserve the current native Attention
baseline, the prequantization 200M total-parameter ceiling and probability JSON.
Earlier explicit authorization permits autonomous local design, implementation,
tests and publication. This stage implements an opt-in research module and CPU
contracts; it does not train or promote a new model. No old run, feature bank,
checkpoint, trainer default or sealed audit changes.

## Inspected evidence

- `native_candidate.py` has candidate-conditioned cross attention over concatenated
  input tokens. It already provides dynamic attention, but no token-to-token AV
  exchange or explicit conditional interaction gate.
- The current common corpus has single-modality observations. It cannot train an
  AV-only path. Its previous feature/label audit is not evidence of complementary
  paired supervision.
- VIDIMU gated experiments, including 1,000 epochs, failed adoption against the
  sensor-only control. Input replacement affected decisions without proving useful
  fusion. Do not repeat that closed comparison as a new success claim.
- Existing AV recording matching is narrower than semantic joint reasoning.
  Perception Test question metadata does not certify required modalities, and
  MUSIC-AVQA v2 media remains unapproved. Do not pair unrelated existing examples
  and call them naturally synchronized observations.

## Alternatives and selected implementation

1. Existing candidate Attention: immutable primary baseline, zero extra parameters.
2. Conditional gate alone: useful ablation; cannot by itself establish cross-modal
   matching or recover evidence absent from the encoder representations.
3. Bidirectional exchange plus conditional gate: proposed experimental path.

Implement (2)/(3) as a residual correction around `NativeCandidateReadout`, with
the ungated, no-exchange residual arm included. The residual preserves the original
path and permits exact initial baseline equality. Its gate controls the added AV
path, **not every contribution of audio/video through the baseline**. Gate values
must not be described as global modality importance or calibrated confidence.

### Exact path

- Reuse the baseline's validated projections, position embeddings and memory norm.
  Merge `speech` and `event_audio` into the audio memory; only `video` participates
  on the visual side. Image/Text remain available to the baseline.
- Use 8 learned queries per stream, width192, four heads, dropout0, and a shared
  pooling Attention. Preserve position features before pooling. This bounds exchange
  cost; pooling can lose detail and is an explicit design trade-off.
- From the original two pooled streams compute A<-V and V<-A in parallel using two
  directional Attention modules and residual LayerNorm. Do not let one direction
  consume the already updated other stream.
- Read each stream with the existing candidate query through a shared Attention.
- A shared MLP, `[q,a,v,a*v,abs(a-v)] -> 64 -> 2`, computes two sigmoid gates per
  candidate. They are independent, not a sum-to-one competition; both may be high
  or low. There are no option-ID/position, target or source-ID inputs.
- Score `[q,g_a*a,g_v*v,(g_a*a)*(g_v*v),abs(g_a*a-g_v*v)]` through
  `960 -> 192 -> 1` and add to baseline logits. Initialize only the last layer to
  zero for exact baseline parity. Upstream added-layer gradients are consequently
  zero on the first update; verify their reach after a controlled nonzero-head
  fixture, without representing that fixture as a trained model.
- Missing audio or video means skip the entire added path and return baseline
  logits exactly. Missing blocks are omitted; malformed/all-masked blocks fail
  under the existing baseline contract. Do not fabricate a missing stream.
- Store ablation policy in the state dict and reject cross-policy checkpoint loads.
  Return diagnostics only on explicit request. Default output remains logits for
  the existing JSON converter, with uncalibrated temperature1 softmax.

### Causal comparisons

Keep C0 existing Attention. Four new arms share the same residual/pooling path:
exchange off/on x conditional gate off/on. In a disabled gate arm use ones. In an
exchange-disabled arm use original pooled streams. All modules have identical
initial tensors for a given seed; report active and allocated parameters separately.
Compare within this 2x2 to isolate exchange and gate effects, and to C0 for net gain.
Parameter-matched extra capacity is not controlled by C0 alone.

## Data and scientific gate before real training

Require ALLOW provenance, underlying asset/participant grouping, immutable TRAIN
and validation IDs, verified synchronization metadata and task-level label evidence.
Include audio-sufficient, video-sufficient and both-required tasks; query-only,
audio-only and video-only baselines must be reported. Define clean, missing,
degraded, irrelevant-stream, relevant-stream-swap and time-offset strata before
training. Swaps that change truth need recomputed valid labels; unchanged old
labels on broken pairs measure sensitivity only, not correctness.

Contradictory inputs require a task-defined resolution or an explicit supplied
unknown/insufficient-evidence option. Gates alone do not determine which input is
true. Do not force equal gates or train sensitivity to all swaps: irrelevant swaps
should preserve answers; relevant truth-changing swaps should alter them.

Freeze budget, sampler, loss, seed17, selection by validation NLL and all evaluation
IDs across arms before learning. Choose the budget from measured throughput and
record full curves, best/final, unique assets and repeat counts; do not infer a
ceiling from a short smoke. Retain four-modality results, joint-task NLL/Brier/ECE,
counterfactual pair correctness, appropriate cluster intervals, latency and memory.
No adoption without useful joint gain and non-joint retention. No threshold or
training schedule is selected by this implementation-only stage.

## This stage's acceptance evidence

CPU tests: baseline parity at initialization and with missing streams, both
directions' gradient reach, candidate permutation, masked-padding invariance,
conditional gates, same-policy exact reload and cross-policy rejection, invalid
inputs, and total instantiated parameters below200M including the existing encoder
ledger195,365,529. Controlled finite random tensors test mechanics only.
No quality or robustness claim follows from these tests. Record common-corpus AV
availability from metadata only, without loading media or final evaluation data.

## Research context

[GMU](https://arxiv.org/abs/1702.01992) studies learned multiplicative modality
gates. [Attention Bottlenecks](https://arxiv.org/abs/2107.00135) studies compact
cross-modal exchange. These motivate hypotheses; this residual module is not a
reproduction of either paper and their benchmark results do not predict ours.
