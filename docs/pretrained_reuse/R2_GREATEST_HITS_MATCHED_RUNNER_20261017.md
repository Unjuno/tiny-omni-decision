# Matched AV learning runner: implementation and admission evidence

**Full comparison training remains unstarted.** This stage implements the fixed
[prospective protocol](R2_GREATEST_HITS_MATCHED_LEARNING_PROTOCOL_20261017.md) and
executes CPU plumbing tests plus a real audit of the existing rehearsal banks.
It does not supply a new model-quality result or a full GPU runtime estimate.

Implementation commits: `8488a31`, `5356d56`, review fix
`07e958c63af85de371921d925ee931f014133e2f`, branch `codex/r2-av-transport`.

## Implemented execution path

`python -m scripts.train_greatest_hits_matched` has four explicit phases:

| Phase | Required evidence | Output / scope |
|---|---|---|
| `audit` | Terminal complete video, independently revalidated audio, frozen corpora and preflight | Full bank hashes/shapes/finiteness, all video clocks,21 smoke matches, full binding |
| `smoke` | Passing pinned audit; idle local GPU | Per arm32 exact-plan updates, complete evaluation and checkpoint reload, projected full cost |
| `train` | Same-source/arm/config/input passing smoke; resource allowance at least1.5x projection |32768 updates, ten scheduled evaluations, retention-selected best and separate final |
| `replay` | Passing pinned training receipt with all artifacts | Re-evaluate best and final against every saved score/gate matrix |

Each invocation requires a new `--output` and explicit `--max-seconds`. GPU phases
require `--arm`, `--audit` and `--audit-sha256`; train/replay also require
`--reference` and `--reference-sha256`. There is no resume or scientific-budget
override. Resource guards stop as incomplete, never as a measured capacity ceiling.
The smoke projection includes ten full evaluations and two endpoint reloads.
The32-update smoke cost is a projection with limited timing coverage; actual
training/evaluation time and resource usage remain recorded separately.

The code keeps the original frozen preflight sampling and fitted initial tensors.
Eight losses are averaged per optimizer call; candidate permutation and remapped
targets agree for both joint and common inputs. All readout parameters are
optimizer-eligible; no encoder is loaded for training. Fresh AdamW state is used.
The inherited62-step warmup/32768-step cosine is unchanged.

Evaluation always includes the full planned consumed TRAIN union:23707 joint and
84480 common questions, fixed even at step0. Actual cumulative exposure is stored
separately. Validation remains8875 joint IDs in joint/audio-only/video-only views
and24346 original common IDs, all in their original order. Missing-input views
are ablations, not proven answerable subsets. Each point saves raw logits aligned
to ID/options/target metadata, metrics, descriptive common-task gaps, and candidate
gate values where active. Gate values describe the added residual, not the total
importance of a modality or calibrated input reliability.

The selector requires each existing modality to retain Accuracy within2pp and
existing macro NLL within1e-5 of its initial value. Lowest eligible joint task-macro
NLL wins, min_delta1e-5 and earliest tie. Every scheduled checkpoint is kept.
An ineligible endpoint cannot replace best. `best.pt` and `final.pt` are reloaded
and compared against their complete original prediction matrices before a training
PASS. Optimizer, sampler position, selector reference and RNG snapshots are saved;
this does not claim a tested interrupted-training resume implementation.

## Executed verification

- 10 runtime tests: correct mean-of-eight gradient, candidate alignment, fixed probe,
  real conditional-readout gates/missing bypass, and checkpoint policy preservation.
- 20 runner/admission tests, including a synthetic CPU test that executes the CLI's
  smoke, train and checkpoint-replay branches with a tiny fixture and real readout.
  This test uses one pytest process with CPU transport/device stubs. It is not
  full-corpus training, a GPU smoke, or fresh-process equivalence evidence.
- Full suite:2376 passed,1 skipped,67.53seconds. Ruff and25 manifest commands pass.
- Fresh read-only review found one admission defect: a complete-looking video
  receipt could omit a bank from its hash ledger and fail only when that bank was
  later accessed. Five regression cases failed before the fix. Admission now
  requires the exact complete video/positions/pts ledger before hashing any inputs;
  video plans are also explicitly pinned. All30 targeted tests pass. The reviewer
  confirmed the fix and found no additional material issue in the focused diff.

Logs: `C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-matched-runner-verification-01/`.
Both initial Ruff formatting failures and the review RED evidence remain saved.
The final full checks are in `checks-02/`; exact pushed-commit CI is recorded
separately in `ci.json` after publication.

## Real rehearsal-bank audit

The independent CPU audit read the unchanged common corpus and checked all466
registered banks (6223056384 tensor bytes, about6.22GB), including query features.
Every file hash, stored shape, FP32 dtype and every feature's finiteness passed.
The100580 TRAIN and24346 validation rows have no overlapping observation IDs or
source-qualified groups. Their feature observation split ownership also agrees.
This is integrity evidence, not proof of sufficient labels or actor/site independence.

Wall time32.4790seconds, optimizer calls0, no quality inference, no CUDA work,
no test/calibration/sealed payloads, no cloud cost. Output:
`C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-matched-rehearsal-audit-01/`.

| Evidence | SHA-256 |
|---|---|
| Audit script | `b58aeda1ad4eabc9c1044e17c0a6c0e6b8968ec6f6df95bdefd66424ca24b05a` |
| Audit result | `5b7aa5bc3d64e04d72746d0123386c7b5fce9b77ebd440e1bb55cd7bcbc98f3c` |

The real original video producer remained running, untouched. At05:42JST it had
saved6532/11151 events across236/409 parents. The new runner refuses partial cache
admission. Existing teacher data/config/checkpoints, original worktrees and user
uncommitted changes were not edited. No new Teacher artifact was produced.

## Remaining work

Finish video extraction; execute the complete independent AV audit; measure the
full run path per arm; record practical local allowances; then execute the five
fixed long curves sequentially and compare fixed steps plus selected checkpoints.
No job is claimed scheduled merely because its runner exists.

C0 remains196254874 total parameters and factorial arms197097949, prequantization
and encoder-inclusive. Sparse Video rehearsal and zero bell TRAIN positives remain
documented limitations. The new runner does not repair those data limitations.
Input swaps and defensible complementary/conflict labels still need their own
frozen diagnostic inventory. Joint quality improvement,95–98% task goals and
product promotion remain unproven.
