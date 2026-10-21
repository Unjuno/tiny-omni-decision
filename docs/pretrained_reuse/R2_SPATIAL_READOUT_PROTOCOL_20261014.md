# Spatial context readout comparison — fixed before QA fitting

## Intent

Follow the completed role-supervised QA comparison with the missing later
appearance/position input. Preserve the full Omni goal and all existing artifacts.
This isolated new readout does not alter the active spatial extraction's code,
configuration, weights or sampling. CPU implementation while extraction is live;
no second GPU task. User autonomous strategy authorization permits execution.

## Architecture A extension

Frozen point tokens remain256×512 and text candidates1536-d. Additional spatial
tokens are32×256×(384appearance+3normalized x/y/source-time), with every source
cell represented by the fixed pooling. Project387→64, cross-attend16 learned
64-d latent queries with4 heads, LayerNorm/residual and a64→128→64 feedforward.
The existing cover point/context encoder remains512→64. Point-to-latent and
candidate-to-latent4-head attention enrich point and candidate representations;
candidate-conditioned point attention adds **predicted** cover logits and yields
three supplied-option logits. No source box, track ID, event or label enters
forward. All encoders/projectors from released models remain frozen; only this
small readout learns. No new large world model or runtime teacher.

Expose encode_state then score_candidates so multiple questions reuse one encoded
state. This is transient reuse under fixed weights; persistent cache/version
invalidation and raw-media runtime are not claimed complete. Padding must be inert,
point/candidate permutation equivariant, finite gradients and exact weight/state
reload proven. Input types/shapes/all-masked/nonfinite inputs fail closed.

## Controlled experiment

Two arms use identical architecture, initialization, seed17 sampler, original
99train/257development IDs and198/514 questions, source cuts,32 times, positions,
point-motion features, options, role-label availability98/253, auxiliary weight1,
AdamW0.001/weight_decay0.01, batch8videos,50epochs and earliest minimum original
dev-question stable log-softmax NLL. No last-checkpoint selection or seed sweep.

Only the appearance values differ: `initial_broadcast` repeats frame0 appearance
over all32 time/position slots; `temporal_appearance` uses actual32 appearances.
Keep positional tags and point motion unchanged. Comparison to the old139,521-d
readout is contextual; it is not a one-factor architecture comparison.

Full corpus must be complete and independently audited before fitting. First
execute a separate-process two-update train-only GPU smoke with all encoders
absent from training, predeclared4GiB peak allocation, runtime measurements and
exact best/final replay. If practical, execute both fixed50-epoch runs and report
all attempts/curves/unique and repeated counts/best/final artifacts.

Execution limits fixed before full fitting: total body at most3600seconds,
peak allocated CUDA memory at most4GiB, at least12GiB free CUDA memory and8GiB
disk at preflight. Check limits inside fit/inference batches and before PASS;
breaches preserve an incomplete attempt. Windows peak working set is recorded.
The registered readout count is232,385; no encoder is loaded during fitting.
Train-only smoke completed two updates per arm and exact best/final replay;
smoke timing includes concurrent CPU test contention and CUDA startup.

Primary original QA Accuracy/NLL/Brier/ECE15, both-condition and127 changed-target
pair correctness over all514 questions. Report train/dev gaps and existing
supplemental cover metrics with their own253-label denominator. Selected fixed
controls: zero state/context, cyclic other-video context, broadcast initial
appearance and time-reversed appearance while position tags remain fixed;
controls never choose checkpoints.10,000 paired video resamples/seed17/percentile95%
intervals; same observed checkpoint-selection dev, not blind/actor-disjoint evidence.
Do not claim helpful Video merely because aggregate metrics rise or features vary.

## Cost and exit

Track actual registered/trainable parameters, readout bytes, peak GPU/process
memory and synchronized time; include all resident frozen encoders in component
and eventual full Omni accounting. No quantization, cloud, product promotion or
old artifact overwrite. Close this comparison if gain is absent or conditional
use remains unproven, and report bottlenecks without claiming a family ceiling.
