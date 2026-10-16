# Matched AV comparison: actual sampler and fitted initialization preflight

**Preflight PASS; full training has not started.** Source
`4f90bbd46f9aa2765531072a68f861ad1fee67db`, branch `codex/r2-av-transport`.
The [prospective protocol](R2_GREATEST_HITS_MATCHED_LEARNING_PROTOCOL_20261017.md)
fixes five arms,32,768 updates and joint plus four-modality rehearsal before any
new validation result. No existing experiment was resumed or changed.

## Actually executed

- Read pinned complete TRAIN and validation metadata from the existing common
  corpus and Greatest Hits development corpus. No validation examples were
  forwarded, no test/calibration/sealed payload was opened.
- Run the actual new sampler for all32,768 batches of8 examples. Reverse input
  row order and regenerate; complete sequence and exhaustion traces agree.
- Generate/check complete candidate permutations and aligned targets for all
  262,144 planned presentations. Sampling uses IDs/source/task, never labels.
- Create five initial states from the exact selected step8192 control model
  (`native-long-budget-fit-01/best.pt`), preserving zero-output residual fixtures.
  The control optimizer is not adopted; future fits use fresh optimizers.
- On the existing21 TRAIN events /63 questions in joint/audio/video views, all
  five models give the same189 logits records each.945 initial outputs agree;
  frozen encoders are cached and no optimizer call is made.
- Independently recheck every stream cycle, prefix counts, class support, LR
  horizon, copied baseline/residual tensors and output parity. All9,695 older
  protected files retain their hashes;12 new output files are verified.

Runtime78.6229s, CPU only, peak process working set1,767,084,032bytes. Local
Python3.11.9/PyTorch2.6.0+cu124,1 CPU thread; no dependency changes or cloud cost.
The real video-cache process continued concurrently and was not interrupted.

## Planned exposure, not completed model training

| Stream | Available questions | Planned presentations | Unique planned questions | Repeat presentations |
|---|---:|---:|---:|---:|
| Joint AV | 23,707 | 131,072 | 23,707 | 107,365 |
| Text rehearsal | 18,432 | 32,768 | 18,432 | 14,336 |
| Image rehearsal | 33,696 | 32,768 | 32,768 | 0 |
| Audio rehearsal | 47,940 | 32,768 | 32,768 | 0 |
| Video rehearsal | 512 | 32,768 | 512 | 32,256 |
| Total | 124,287 | 262,144 | 108,187 | 153,957 |

There are25,318 planned unique observations and12,992 media assets by the existing
source asset identities; this is not proof of independence by actor/site or near
duplicate content. Within each stream, every complete cycle includes each question
once. No small bucket restarts before the whole stream is exhausted. Exhaustion
times and changing task mix are retained in `sampling.json` and `coverage.json`.
Joint material/action/reaction presentations are43,819 /43,909 /43,344.

All1,800 **observed source/task/target groups** have at least one planned exposure.
This excludes classes that have no TRAIN positive at all, and is not adequate
class support by itself. In particular:

| Existing STARSS task | Available positive questions | Planned unique positive questions |
|---|---:|---:|
| bell | 0 | 0 |
| clapping | 9 | 9 |
| knock | 10 | 8 |
| instrument | 73 | 68 |
| telephone | 27 | 27 |

The bell task has1,955 available negative questions and1,821 planned negatives.
Video's512 questions still come from the existing small source corpus;64 cycles
do not create new scenes. Neither deficiency is fixed by the gate or this sampler.
Preserve the frozen sources, report these limitations, and do not infer a model
capacity ceiling or a95% product result from this comparison.

## Fixed comparison conditions

C0 has196,254,874 encoder-inclusive parameters; each of the four factorial arms
has197,097,949. Same fitted baseline and added initial tensors, same data/order,
same32,768-step warmup/cosine/AdamW schedule and candidate permutations. All readout
parameters are eligible. No encoder/projector training or quantization.

The selector first requires each existing modality's hierarchical macro Accuracy
to remain within2pp of its shared initial value, and existing macro NLL not to
increase beyond1e-5. Among eligible checkpoints it takes lowest joint task-macro
NLL, min_delta1e-5, earliest tie, initial0 eligible. These are research comparison
rules; all task95–98%, complementarity and release requirements remain unchanged.
Keep ineligible checkpoints/results and fixed endpoints. No performance early stop.

The warm start and larger TRAIN coverage differ from historical experiments, so
historical values are references. Causal exchange/gate comparisons are within the
new matched arms. Added capacity is controlled by E0G0, not by C0 alone.

## Verification and reproducibility

20 sampler/selector tests failed before implementation, then passed.11 fitted
initialization tests likewise passed after observed RED. Full suite2,346 passed,
1 skipped; Ruff and25 existing manifest commands pass. An independent read-only
whole-change review of `fe79994..4f90bbd` found no actionable material issue.

The first local audit attempt used Windows' defaultcp932 to read UTF-8 metadata
and failed before producing an audit result. Its source/log are retained as
`audit-attempt-01.py` and `audit.log`; only the audit reader's explicit encoding was
corrected. The successful audit is `audit.py` /`audit-02.log`; no experiment input,
generated plan, model state or producer code was altered to bypass this error.

Run from `C:/Users/junny/.codex/worktrees/r2-av-transport/tiny-omni-decision`:

```powershell
$env:PYTHONPATH='C:/CodexArtifacts/pretrained-reuse-r1/python-deps;C:/CodexArtifacts/pretrained-reuse-r2/native-speech-deps-01;C:/Users/junny/.codex/worktrees/r2-av-transport/tiny-omni-decision/src;C:/Users/junny/.codex/worktrees/r2-av-transport/tiny-omni-decision'
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
& 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe' -m scripts.preflight_greatest_hits_matched --output C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-matched-preflight-01
```

Exit0. The output already exists and must not be reused. The run's source/config,
input files and exact command are pinned in its receipt. Full sampling, all
scheduled-prefix counts, label support, LR vector, validation ID order, five new
initial state files and initial predictions stay outside Git under that path.

| Evidence | SHA-256 |
|---|---|
| Execution `result.json` | `d0dc5e585b00a45cfdc17773b42cd40a0e4b9b08015127af014ab8ce5e7b5ac1` |
| Canonical sampled ID order | `7c4fdba786336afa6fe0eb4a855b5c04da340b04d54cb404ea71755312ed9b4b` |
| Candidate permutation sequence | `9c1e752fa2b68e0a51953881106c39c0de462f5918c9adf308e73306a32208aa` |
| Independent audit source | `8a36a83692cee8808fe58109944403e59708f079b260732aa996a00984850e97` |

Verification root:
`C:/CodexArtifacts/pretrained-reuse-r2/greatest-hits-matched-preflight-verification-01/`.
Audit status `PASS_MATCHED_AV_PREFLIGHT_AUDIT`. Repository record:
`manifests/pretrained_reuse/greatest-hits-matched-preflight-20261017.json`.
Exact pushed-commit CI is saved separately in verification `ci.json`.

## Remaining execution

At publication preparation, the original live CPU video cache reported5,151 of
11,151 events,184 of409 parents. Complete video/hash/clock binding with the
independently revalidated audio is still required. Do not train on a partial cache.
Implement the full matched trainer using this saved plan and initialization, then
measure full-path smoke and evaluation cost before admitting sequential local fits.
Raw metrics, meaningful input-intervention correctness, four-modality retention,
best/final reload and the main quality comparison remain unperformed. No training
reservation, product promotion or completion of the overall goal is claimed.
