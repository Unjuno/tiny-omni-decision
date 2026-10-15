# R2: matched condition-relevance fits — completed, no promotion

2026-10-16 JST. Both main fits and exact TRAIN replay completed. The fixed
development-transfer evaluation was running at initial publication and has now
[completed with an independent output audit](R2_CONDITION_RELEVANCE_TRANSFER_RESULTS_20261016.md).
The companion fit JSON preserves its initial pre-completion snapshot.
This is a structured-state **text diagnosis**, with zero native
media assets; these numbers are not four-modality product quality.

## Decision

The predeclared auxiliary-supervision gate fails. Both endpoints reach 100% in
each of the new 16 validation slices, giving a zero primary Accuracy difference.
Every trained scheduled checkpoint fails the unchanged 47-slice retention
selector. Both selected checkpoints are step 0. Preserve endpoints as research
artifacts; do not promote them or alter selection/calibration rules.

Auxiliary supervision improves endpoint macro NLL/Brier/ECE relative to CE,
but lowers macro Accuracy. This is a measured trade-off. It does not establish
that the model lacks capacity, nor that more training can resolve retention.
The frozen transfer comparison is now complete; close this intervention.

## Fixed experiment and executed scope

[Prospective protocol](R2_CONDITION_RELEVANCE_FIT_PROTOCOL_20261015.md) and
`configs/pretrained_reuse/condition_relevance_fit_v1.json` were unchanged.
Training source: `6fb617f233752f582af7daf0dd131ba7cc281a5c`.
Config SHA256: `942e86187fa587689165269b92e424d70fe5d615fbc3301342af48204fe1b5db`.
Corpus manifest SHA256: `7593e36769996d07e3654694013a989a8cdcd8a4b58442a29bde4209feb26964`.
Base revision: `262ae02f29173eec1c250f90804dc7edc677dcff`.

- Two fresh pretrained initializations, seed 17; no optimizer or weight resume.
- Same initial base/auxiliary weights, ordered sampler, retention logits,
  token preflight, validation IDs/order, LR timeline and original source pins.
- Both arms: 1,024 optimizer calls, 1,023 with positive LR; peak 2e-5,
  31-update warmup, cosine to zero at 1,024. No performance early stopping.
- Each update: eight new relevance, eight numeric and two legacy questions.
- Each run: 18,432 unique questions / 2,626 unique states / zero repeated
  questions / zero native assets. The two arms use the same data, so these are
  not 36,864 independent questions or 5,252 independent states.
- Per run: 8,192 condition-relevance, 8,192 programmatic numeric,
  1,024 curriculum and 1,024 original questions.
- Core: 70,830,722 parameters; 21,639,554 trainable core parameters plus the
  identical 385-parameter training-only auxiliary head in both arms.
  Optimization scope: 21,639,939. Input embeddings stay frozen.
- Only intervention: relevance loss coefficient 0 versus 0.1. Inference uses
  the native entailment score, without using relevance predictions as a mask.
- Full run consumption is 18,432; selected checkpoint consumption is **zero**.
  Optimizer/RNG state is not saved: this is not an exact-resume artifact.

## Validation learning curves

These are equal-slice macro metrics over all 47 authored/legacy development
slices, not equal-modality metrics. Validation has 10,732 questions. The minimum
Accuracy across all 47 slices remains zero at every scheduled point.

| Arm | Step | Consumed | Macro Accuracy | NLL | Brier | ECE | Retention-failing slices |
|---|---:|---:|---:|---:|---:|---:|---:|
| Both initial | 0 | 0 | 0.367780 | 2.079142 | 0.820687 | 0.245029 | 0 |
| CE | 256 | 4,608 | 0.785443 | 0.684120 | 0.323214 | 0.178977 | 22 |
| CE | 512 | 9,216 | 0.823970 | 0.550593 | 0.268229 | 0.154484 | 20 |
| CE | 768 | 13,824 | 0.842930 | 0.562593 | 0.266246 | 0.176031 | 21 |
| CE | 1,024 | 18,432 | 0.845240 | 0.585113 | 0.272114 | 0.187770 | 22 |
| Relevance | 256 | 4,608 | 0.813926 | 0.589668 | 0.278853 | 0.135918 | 22 |
| Relevance | 512 | 9,216 | 0.811378 | 0.556455 | 0.260488 | 0.114741 | 18 |
| Relevance | 768 | 13,824 | 0.825702 | 0.573696 | 0.264928 | 0.144931 | 20 |
| Relevance | 1,024 | 18,432 | 0.827608 | 0.579873 | 0.264538 | 0.145100 | 20 |

Both arms' macro NLL rises after step 512 while macro Accuracy increases.
This is an observed calibration/generalization warning; do not equate it with
a fixed-distribution train/validation gap. Full per-slice endpoint and selected
metrics, failure reasons, resources and hashes are in the companion JSON;
all scheduled points are in the CSV.

Endpoint retention violations overlap: CE has Accuracy/NLL/Brier/ECE violations
in 2/2/2/22 slices; relevance has 3/1/3/20. Both regress on Accuracy in
`binding:5:color` and `binding:5:count`; relevance also regresses in
`numeric:extremum:time:maximum`. Calibration alone cannot repair those rankings.

The new 16-slice minimum is 100% for both endpoints. The prespecified paired
whole-state bootstrap uses all 128 validation states, all 16 questions per
state, 10,000 replicates, seed 17, linear percentile 95% interval, recomputing
the minimum each time: difference 0, interval [0, 0]. This degenerate result
reflects both arms being correct on every observed new question. It is not a
zero-width uncertainty statement about arbitrary unseen tasks. Keep the
original >=3pp positive-interval requirement; do not relax it after seeing
the CE ceiling. The trained-checkpoint retention requirement also fails.

## Physical preservation, source audit and replay

Both main processes exited normally. Each selected model was reloaded with
exact saved validation and fixed-TRAIN scores. Independent post-fit checks
rehashed every registered input, all 953 original code/config/protocol files,
and all output artifacts; replayed the 1,024 sampler/LR/accounting updates;
and recomputed the unchanged selector at every scheduled point.

Post-fit code was integrated only after the main GPU process exited. Comparison
source: `f2dc24872a22dad92344c2c3009c68f9feb5d44c`. It binds actual executing
entrypoints, imported module origins, namespace locations and pinned file bytes.
Copied runners, stale bare imports and redirected paths are rejected.

One comparison-preparation attempt failed before output/model/transfer access:
the same legitimate namespace location appeared twice through working-directory
and PYTHONPATH entries. A failing regression reproduced it; the correction
accepts repeated identical locations while retaining rejection of other paths.
The failed-attempt record is preserved separately. No model fit was restarted.

After freezing all five roles, a separate GPU process replayed all 536 original
fixed-TRAIN questions for all three distinct weight hashes. Every score array
was exactly equal to the saved reference, with no optimizer/backward. It took
74.431 seconds and peaked at 1,826,286,592 allocated CUDA bytes. Initial,
CE-selected and relevance-selected roles remain explicit even though their
byte-identical weights share one inference pass.

## Frozen identities and local artifacts

All local directories are under `C:/CodexArtifacts/pretrained-reuse-r2/`.

| Artifact | Directory / SHA256 |
|---|---|
| CE fit | `condition-relevance-fit-ce-01` |
| CE terminal receipt | `539d88b1166332404a61a27efc1fc9416b6590d74a339baff41a9be54ce2320a` |
| Relevance fit | `condition-relevance-fit-aux-01` |
| Relevance terminal receipt | `c70b89d4b04c936ab213eef163bfa3342b1b8a98b3c87558b0f321dae4175e35` |
| Comparison | `condition-relevance-postfit-comparison-02` |
| Comparison receipt | `b7541d4f7ec7b3736e78bd48d364f9fd007ffd985d9730e496ceb02ee5fa583e` |
| Frozen roles | `c39d0fc447ea8d233672368a7a1152ffb96ce473fe6cdfa61a4c83f76d7ca7c8` |
| Initial / both selected weights | `ea43cc5dfedff0966fb2d99fb88e05cfef5196c47ca0ac71091397b8374e97a2` |
| CE endpoint weights | `a7fa5ff60b81889a167c232c30709beb6193593395d4f0ac4e8abfd4004a80d3` |
| Relevance endpoint weights | `4f2286b0582cbe15cbd4be770014401c3457f37acf1f1b07119be965269c0eca` |
| TRAIN replay | `condition-relevance-postfit-train-replay-01` |
| TRAIN replay receipt | `ce12d2c33b7266702bb708938db67ea5c7ff91672a7eaf1f85d3952285b5c338` |
| Preserved preparation failure | `condition-relevance-postfit-integration-01/prepare-attempt-01.json` |

Each fit directory retains effective config, input/code hashes, sample order,
per-update logs, history, every scheduled delta, separate best/final deltas,
per-example probabilities, metrics and reload evidence. Commands are preserved
in terminal receipts and the companion JSON. No old run, corpus or weight was
overwritten; no large weights/media/checkpoints are added to Git.

## Hardware and cost

Local RTX 3080 Laptop 16 GB; Python 3.11.9, PyTorch 2.6.0+cu124,
CUDA 12.4, Transformers 5.6.2, NumPy 2.4.6, Tokenizers 0.22.2,
Safetensors 0.8.0; FP32/eager, deterministic algorithms, TF32 disabled.
No environment upgrade or paid compute.

| Arm | Process wall seconds | Training seconds | Evaluation seconds | Peak CUDA allocated | Peak host bytes |
|---|---:|---:|---:|---:|---:|
| CE | 4,680.924 | 2,429.462 | 2,157.396 | 3,420,603,904 | 2,492,239,872 |
| Relevance | 4,798.283 | 2,455.211 | 2,248.608 | 3,420,603,904 | 2,464,546,816 |

Combined main-process wall time is 2.633 hours; this is not a GPU-kernel-time
measurement. Separate replay/transfer costs are additional. Cloud cost: zero.

## Transfer and remaining gates

The fixed factorial completed in `condition-relevance-postfit-transfer-01`
(PID11760 exited 2026-10-16 JST): 256 previously untouched source states,16,384 questions,
40 supplied candidates, four seen/new query × seen/new candidate cells of
4,096 questions each. Source/state separation and complete groups pass; maximum
pair length is 70 tokens with no truncation. Only the fixed three unique weights
are scored, retaining all five role identities. No calibration or checkpoint,
seed, wording or candidate reselection occurred. Full terminal evidence and
metrics are in the [transfer report](R2_CONDITION_RELEVANCE_TRANSFER_RESULTS_20261016.md).
Endpoint Accuracy is93.70% /94.17%; the weakest condition/field is50.00% /50.88%,
and an actual query/form slice is0% for both. The adoption gate remains failed.

Sealed audit and reserved calibration remain unopened. This is authored
development transfer, not an estimate of general native-media performance.
The [updated strategy](NEXT_STRATEGY_20261016.md) requires a common candidate
across native text/image/audio/video, unknown conditions, temporal and genuinely
joint tasks. Full integrated <=200M measurement, package/latency and each required
task's 95–98% quality gates remain unmet. No promotion or compression follows.

Verification: 1,858 local CPU tests passed, one skipped before the namespace
regression; 32 targeted tests passed after the correction. Ruff and all 25
manifest commands pass. GitHub Actions PR run #1789 is green for exact evaluation
commit `f2dc24872a22dad92344c2c3009c68f9feb5d44c`.
