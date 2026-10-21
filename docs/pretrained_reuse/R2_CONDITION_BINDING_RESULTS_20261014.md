# Train-only question/state binding diagnosis — executed

## Outcome and scope

Fixed existing temporal-appearance best6 and final50 were evaluated on all99train
videos/198questions, with no fitting, optimizer, new checkpoint selection or
development scoring. Original epoch QA/cover/logp bytes match exactly. Execution
PASS; **question information/routing loss alone does not explain the quality
failure**. The final checkpoint uses question and scene information on its
training examples, yet its existing development result is poor. This can be
memorization rather than correct object reasoning; do not promote it.

[Fixed protocol and documented corrections](R2_CONDITION_BINDING_PROTOCOL_20261014.md),
[all metrics/perturbations/hashes](R2_CONDITION_BINDING_RESULTS_20261014.json),
[original full learning curves and validation](R2_SPATIAL_READOUT_RESULTS_20261014.md).
The broad four-modality95–98% Omni goal, useful native fusion and OOD quality remain
unmet. No sealed/audit set was opened. The common loader validates already observed
development caches but evaluates **only training** in this diagnosis.

## Input and interventions

Byte-pinned text features preserve two distinct question384-d vectors: norms
3.284560/3.323704, cosine .993017, L2difference .392420. Question sharing within
each condition and the1536-d question/option/abs-difference/product factorization
are exact. Cosine alone does not establish a quality ceiling or information loss.
The two exact question strings differ at beginning versus end of the game.

Four conditions retain all original IDs/options/targets. Neutral question averages
the two question vectors and recomputes the interaction fields; options stay
unchanged. Swapped question exchanges question vectors and preserves options/state.
Mismatched state cyclically replaces both point motion/initial appearance and
spatial context with another training video's state. It is intentionally different
from the previous context-only mismatch. None of these perturbed tasks is a new
deployment benchmark or a checkpoint-selection signal.

## Actual train-only results

| Weights/control | Accuracy | NLL | Brier | ECE15 | Changed-target pairs both correct |
|---|---:|---:|---:|---:|---:|
| best6/normal | .449495 | 1.062387 | .642435 | .044682 | 0/53 |
| best6/neutral question | .449495 | 1.062568 | .642542 | .044456 | 0/53 |
| best6/swapped question | .449495 | 1.062850 | .642721 | .044682 | 0/53 |
| best6/mismatched state | .449495 | 1.066968 | .645329 | .044542 | 0/53 |
| final50/normal | .959596 | .139763 | .073836 | .047210 | 46/53 |
| final50/neutral question | .722222 | .677735 | .407739 | .164349 | 0/53 |
| final50/swapped question | .494949 | 1.552123 | .787606 | .417915 | 0/53 |
| final50/mismatched state | .404040 | 3.367678 | 1.013344 | .481858 | 3/53 |

Best6 changes no answers under any intervention. Its probabilities still depend
weakly on input: maximum change .001699/.002796/.078789 for neutral/swap/mismatch.
Final50 predicts differently across the two conditions on47/99videos; neutralizing
the question removes all such differences. Swapping questions changes94/198answers,
neutralizing49/198, mismatching state119/198. Maximum probability changes are
.976485/.939707/.999846, respectively. Swapped question logits/probabilities obey
paired condition reversal at fixed rtol1e-5/atol1e-6. Raw and centered-logit changes
are saved; common logit offsets alone are not interpreted as useful decisions.

This rules out complete removal of condition information in these cached inputs
and shows the readout can fit conditional training decisions. It does not show
understanding of new scenes, actor-relative geometry or correct hidden/cover
association. Final50's previously saved development Accuracy35.80%, NLL2.6764
contrasts with95.96%/.1398 training; this repeats the existing overfitting evidence,
not a new development evaluation or selected final checkpoint.

## Failed attempts and numeric execution correction

All attempts are preserved; failures were not reported as successful diagnosis.

1. `condition-binding-diagnosis-01` stopped at preflight before loading a checkpoint
   or running forward (24.658674seconds). Two of99 paired option vectors differ
   by at most2.0861626e-7 despite identical option strings. Question sharing and
   feature factorization are exact. This is consistent with batched encoder
   arithmetic; the original last-bit cause was not independently proven.
   The corrected diagnostic preserves every original option value and accepts
   paired consistency at the existing candidate-equivariance tolerance. Exact
   source/archive hashes and original-epoch byte replay remain mandatory.
2. `condition-binding-diagnosis-02` completed the fixed best6 controls but failed
   original byte replay (30.295991seconds). QA/cover/logp maximum differences
   were1.1920929e-7/1.7881393e-7/4.7683716e-7. Pending predictions and metrics,
   source/config/error are retained; they were not used for model selection.
3. `condition-runtime-parity-01` tested the first8train videos in four fixed
   combinations of requires_grad flags and no_grad/inference_mode
   (25.339882seconds). Reference flagsTrue reproduced original QA/cover/logp bytes
   under both contexts; flagsFalse changed last bits under both. Every weight
   remained identical and no backward was called. Specific internal CUDA kernels
   were not profiled; this is evidence for the flag-dependent numerical path here.
4. `condition-binding-diagnosis-03` keeps reference parameter flags and uses
   no_grad. Both complete best/final training outputs pass strict original replay.
   Actual232,385 requires_grad-flagged parameters are explicitly distinguished
   from **zero optimizer updates, zero gradients and no recorded output graph**.
   Every weight/state hash and flag remains unchanged; CPU/CUDA RNG is unchanged.
   No encoder/projector is loaded or unfrozen. Numeric parity correction is not
   an explanation or improvement of the original quality failure.

Failed attempts include source snapshots/provenance. No existing cache, corpus,
teacher, adapter or run artifact was overwritten. The diagnosis added0parameters
and produced no new trained model. Independent auditing recomputed all8 metric
groups from saved probabilities/logp/raw logits and original targets/ordered IDs,
and rechecked both original epoch bytes/hashes. No CPU/GPU tests use sealed data.

## Reproduction, cost and artifacts

- Source commit `c9d76523edc35850ffa00d1aac55c2f2c5b5c7d1`.
- Command: `python -m scripts.diagnose_condition_binding --output C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-diagnosis-03`.
- Effective config SHA `0ad93365ce64f05dca20dcaec8c059331262398ace908359d569b792d8c3b9c4`.
- Result/publication SHA `57d3fc19fb9132fc283ed3239463461e8bdebb1faaa56850f6dad9eb82853960`.
- Original full run SHA `cdb3b521519af36d3394fcd6e5a620f1dd0312da193223d71371075704ab019b`.
- RTX3080Laptop16GB, Torch2.6.0+cu124, FP32/TF32off, deterministic algorithms,
  CPU4threads, existing environment. Successful body40.227535seconds includes
  source/cache integrity loading and final hashing, not just GPU forward time.
  Peak allocated318,501,376bytes; Windows peak working set2,183,024,640bytes.
- Four sequential attempts/probe bodies total120.522083seconds; no training,
  cloud cost0. Forward-only GPU busy time was not separately measured. Manifest
  verification partly overlapped attempt01; these are not isolated speed benchmarks.
- Outputs: `C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-diagnosis-03/`;
  best/final prediction NPZs include logits, probabilities, logp and cover outputs.
  Config preserves all reference input hashes, checkpoint SHA, options/IDs/targets,
  controls and numerical gates. Old runs are read only.
- Independent audit `C:/CodexArtifacts/pretrained-reuse-r2/condition-binding-independent-audit-03.json`;
  mode probe `C:/CodexArtifacts/pretrained-reuse-r2/condition-runtime-parity-01.json`.
- CPU tests1056passed/1skipped15.81seconds; Ruff passes. All manifests, four catalogs
  and policy audit pass. Code review passed after the provenance/logit fixes and
  the runtime corrections. Exact publication-head CI is checked after push.

## Next direction

Close condition-loss/routing-bug as the sole explanation. Do not patch an answer
rule, inject source roles into forward, enlarge capacity without evidence, select
final50 for its training Accuracy or extend these same99video repetitions.
The next useful primitive is input-only object/cover and temporal association with
training-source supervision, plus a verified mapping from visual relationships
to supplied conditions. Distinguish visible-cue availability from feature/readout
generalization before another fit. Public licensed source expansion and genuinely
complementary native AV/sensor tasks remain part of the full goal; no product,
ternary, Recovery LoRA, paid compute or main merge follows this diagnostic.
