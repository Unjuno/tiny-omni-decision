# Video Teacher v2 E long-budget attempt

## Status

**Execution blocked at optimizer step 128 of 2,048 by a Windows checkpoint
serialization path-length error.** This is an infrastructure failure, not a
negative learning-quality result. The requested fixed-step comparisons at 512,
1,024, 1,536, 1,920, and 2,048 were not reached, so the learning-budget
hypothesis remains **uncertain**. No Teacher artifact was selected for product
use.

The failure exposed and fixed a checkpoint staging-path bug. Exact resumption
of this run is not possible: the checkpoint write failed before the optimizer,
scheduler, sampler position, selector state, and RNG state were persisted. The
saved step-128 adapter is validation-selected weights only; it is not a
training-resume checkpoint. Per the run protocol, training was not restarted
from guessed state.

## Scope and immutable inputs

- Repository: `Unjuno/tiny-omni-decision`
- Branch: `codex/video-teacher-v2-long-budget`
- Fetched branch baseline: `6b15fcbb2546ac153c0313af0d63be86f41ff885`
- Run source commit: `ebf1ae9f37312dd77a09bb2b5532480dd90a1268`
- Code fix commit: `57bdccc12292d6a9ee1bcaca4127ace3c04c58fb`
- Base/processor: `google/gemma-4-E2B-it-qat-q4_0-unquantized` at revision `6befbaca7398925921802abd1f277b495b78b738`
- Run-time trainer/training source hashes: `e0bf53fe6403beb33793ca3f058d8cb63a0f8c1415fdfae291b5c40089b99b0a` / `ac31f680a48c977f18dee9e39ea88fb484e0b0cd2810ff78f51b505d8f0a4bc1`
- Runner SHA-256: `859f3370455a0a9c9ef41a02d658003601ddc96a0f3911ff3c09f5ebc1683157`
- Experiment ID: `20261005T013219Z-7f7e0a8a`
- Run start/end: `2026-10-05T01:32:19.965Z` / `2026-10-05T02:07:22.430Z`
- Run command: `$env:PYTHONPATH='src'; python -u 'artifacts/tiny-omni-decision-teacher-v2/candidate-e-long-budget-seed17-2048/run_e_long_budget.py' 1> 'artifacts/tiny-omni-decision-teacher-v2/candidate-e-long-budget-seed17-2048/run.log' 2>&1`
- v1 comparison command: `$env:PYTHONPATH='src'; python -u 'artifacts/tiny-omni-decision-teacher-v2/candidate-e-long-budget-seed17-2048/evaluate_v1_reference.py' 1> 'artifacts/tiny-omni-decision-teacher-v2/candidate-e-long-budget-seed17-2048/v1-reference-eval-deterministic.log' 2>&1`
- Effective config payload SHA-256: `e8d547a888796f0b1888142f377d245a90abc64433d20513f0bce11a5729161f`
- Config file SHA-256: `4c669bdaf9e52f1d365d26da6c5ad0362b08898dd1488bf711e487f638a253aa`
- Effective config file SHA-256: `e2a3e33fae1fef6f9e9495e16cce5fb5c40abc5ff639b37c8d340db8eb275518`
- Frozen train corpus SHA-256: `55e5622f90163a4d6ce421872e37e9825aba459cf8ff92f285056527f2eceb96`
- Frozen validation corpus SHA-256: `32fd9166bf37774fab08ae7b8ce99b38a7630c5ee500ff167fdb1f8b5b45197c`
- Frozen corpus manifest SHA-256: `ef1239db99747a9eec3d1f5c8d474515a4ae12a80b8e6dbe15d568b67851830c`
- Preflight report SHA-256: `778b27722ee368409eaa34ff3047e0e13016d1d05bfcc662d1410690e057e291`
- Effective E validation IDs/order SHA-256: `a3497d17679b5a6c6778317d12c74df9e1bcaa1daca04cddef9ee68a1ab18df4` (472 IDs)
- First 8,192 sampler IDs/order SHA-256: `fa934630a2c75a2da2be5d344c3ccc05d476105fa21e72433812590c8681cfdc`
- Step-128 consumed-ID order SHA-256: `aa96e97b1c94083480f64dbf01b8afff76d36ebdb1540fafced0042b5c0b583f`

The effective config matched Candidate E except for the new run/output identity,
2,048 maximum updates, patience 17, and latest-only resume snapshot retention.
It used seed 17, 8 video frames, decoder-all-linear LoRA rank 16 / alpha 32 /
dropout 0.05, no rsLoRA, LR `5e-5`, cosine scheduling over the full 2,048-step
horizon, warmup ratio 0.03, accumulation 4, sequence length 1,024, and frozen
modality encoders/projector. Loss, option shuffle, modality/source/video-task
weights, corpus, and validation selector were unchanged. CLI overrides were
empty. Performance early stopping remained outside the schedule: patience 17
exceeds the 16 scheduled validation intervals, covered by a test.

The E-512 invocation used the same pinned base revision, seed, corpus hashes,
and a fresh LoRA. It passed Teacher v1's selected adapter only as a validation
reference and had no `resume_from`. E-long likewise used a new LoRA over the
pinned base, did not load E-512 weights, and used v1 only for comparison. The
v1 selected adapter hash remained `4e664a5500a3a109a48de7ec7c807d2b09b61cf386fd124ca8368e197802f951`.
No v1 or prior v2 config, corpus, checkpoint, or artifact was changed. The
sealed audit was not opened, loaded, evaluated, or hashed by this run. No
ternary conversion or Recovery training was performed.

The 2,048-step cosine trajectory is not the 512-step trajectory extended by
1,536 updates: its warmup horizon is 61 steps rather than 15. At E-long step
128, LR was `4.9867963e-5`; at step 512 it would still be `4.3929010e-5` and
only reaches zero at 2,048. The run did not reach step 512.

## Prior run reconciliation

| Run | Configured cap / completed step | Selected step | Run examples | Selected-step examples |
|---|---:|---:|---:|---:|
| Teacher v1 selected run | 2,048 / 2,048 | 1,920 | 8,192 | 7,680 |
| Candidate E-512 | 512 / 512 | 512 | 2,048 | 2,048 |
| E-long attempt | 2,048 / 128 | 128 (only observed eval) | 512 | 512 |

Teacher v1's 2,048-step cap is documented in its frozen config; its trainer
state records 2,048 completed steps, selected step 1,920, and sample indices
8,192 at run end versus 7,680 at selection. Its selected adapter has 2,678,784
trainable parameters. Candidate E's stored run metadata records 512 completed
updates, selected step 512, 2,048 examples at run end and selection, 205 loaded
decoder target modules, 24,158,208 trainable parameters, and a 96,693,360-byte
adapter. E-long retained the same E rank/target policy and the saved step-128
adapter has the same 96,693,360-byte size. Its saved adapter config and tensor
shapes confirm 205 unique target modules, all under `model.language_model.*`,
and 24,158,208 LoRA parameters; projector parameters remain frozen. E-long's
final run metadata was not written.
Candidate E-512 recorded 3,031.515 s of optimizer time, 2,211.610 s of
validation evaluation, a 5.921 s mean update, and peak allocated VRAM of
12,006,743,040 bytes. Its actual base dtype and loaded attention backend are
not recorded in the old run metadata.

The E-512 runner passed `artifacts/tiny-omni-decision-teacher-v1/selected` as
`reference_adapter_path`; the E-long invocation records that same path and
`resume_from: null`. In both runs, trainer code loaded the reference only to
evaluate it, unloaded the reference adapter, and then created a new LoRA over
the base. Neither run warm-started from the reference adapter nor resumed an
older training state. The 512-step and 2,048-step cosine horizons have
different warmup/LR histories, so their comparison changes more than the
number of optimizer updates.

## Preflight and actual consumption

The frozen sampler dry-run confirmed the first 8,192 examples were unique with
zero repeats, covering 3,731 underlying assets, 318 video scenes, and 2,097
video parent questions. All media files existed and train/validation identity
and content gates passed. The predictive video bucket exhausted at example
4,873 and the CLEVR-4 image bucket at 5,087; the saved preflight report records
the resulting actual mixture changes. No bucket limits or data were changed.

The failed run completed exactly 128 optimizer updates at four one-example
microbatches per update:

| Measure at failure / selected step 128 | Observed |
|---|---:|
| Run-wide examples consumed | 512 |
| Selected-step examples consumed | 512 |
| Unique examples / repeats | 512 / 0 |
| Unique underlying assets | 459 |
| Unique video scenes / parent questions | 139 / 171 |
| Peak allocated VRAM | 11,885,037,568 bytes |
| Sum of optimizer-step times | 899.736 s (15m 00s) |
| Mean step time | 7.029 s |
| Step-128 validation time | 379.234 s |
| LR / gradient norm at step 128 | 4.9867963e-5 / 19.3836 |
| Fixed-selector score / selected step | 0.629342 / 128 |
| Selector warnings / overfit flags at this eval | none / none |

Consumed examples were Audio 85, Image 128, Text 128, and Video 171. By source:
Speech Commands 85, CLEVR-4 128, Open-Jev 64, Typed Decisions Synth 64, and
CLEVRER 171. Video tasks consumed were counterfactual 34 (19.9%), explanatory
51 (29.8%), predictive 52 (30.4%), and temporal descriptive 34 (19.9%). The
unique underlying assets by modality were Audio 81, Image 111, Text 128, and
Video 139; unique assets by source were Speech Commands 81, CLEVR-4 111,
Open-Jev 64, Typed Decisions Synth 64, and CLEVRER 139. The matched E-512
configuration had 205 decoder targets and 24,158,208 trainable parameters;
E-long used the same loaded target policy, while its run-side parameter count
was not persisted after the checkpoint failure. The selected local step-128
adapter is 96,693,360 bytes.
rolling training window at step 128 had CE 1.2501 and Accuracy 0.5508; this is
not a fixed-example generalization-gap measurement. No overfitting warning was
raised at this first evaluation only.

The complete preflight expected 8,192 examples over 2,048 updates, but only 512
were consumed. Run-wide and selected-step counts therefore happen to be equal
because the process stopped immediately after the first scheduled validation.

## Same-validation comparison

The v1 reference and Candidate E-512 persisted results, the new E-long step-128
result, and the standalone v1 comparison all use the same ordered 472 E
validation IDs and 8-frame E preprocessing. The selected Teacher v1 adapter
was evaluated once in a separate process with the trainer's deterministic
attention settings; its result matches the v1-reference metrics recorded in
the prior E run. An initial comparison using default attention settings gave
macro Accuracy 0.7225, minimum-modality Accuracy 0.5424, macro NLL 0.6373,
macro Brier 0.3525, macro ECE 0.0899, and Video Accuracy 0.5424. It is retained
locally as `validation-v1-reference-evaluation-default-backend.json` but
excluded from this table and from any selection because its attention settings
did not match the trainer.

### Macro metrics

| Candidate | Step | Macro Accuracy | Minimum modality Accuracy | Macro NLL | Macro Brier | Macro ECE | Mean confidence | Video Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher v1 reference | n/a | 0.7267 | 0.5508 | 0.6371 | 0.3531 | 0.0887 | 0.7863 | 0.5508 |
| Candidate E, selected | 512 | 0.7161 | 0.5339 | 0.7041 | 0.3677 | 0.0723 | 0.7497 | 0.5339 |
| E-long, only observed checkpoint | 128 | 0.6525 | 0.4661 | 0.8151 | 0.4168 | 0.1060 | 0.7196 | 0.4661 |
| E-long, fixed comparison point | 512 | not reached | not reached | not reached | not reached | not reached | not reached | not reached |
| E-long, fixed comparison point | 1,024 | not reached | not reached | not reached | not reached | not reached | not reached | not reached |
| E-long, fixed comparison point | 1,536 | not reached | not reached | not reached | not reached | not reached | not reached | not reached |
| E-long, fixed comparison point | 1,920 | not reached | not reached | not reached | not reached | not reached | not reached | not reached |
| E-long, final planned point | 2,048 | not reached | not reached | not reached | not reached | not reached | not reached | not reached |

### Per-modality metrics

Each cell is Accuracy / NLL / Brier / ECE. Counts are 118 per modality.

| Modality | Teacher v1 reference | Candidate E-512 | E-long step 128 |
|---|---|---|---|
| Audio | 0.9068 / 0.3756 / 0.1361 / 0.0522 | 0.8898 / 0.4271 / 0.1539 / 0.0446 | 0.8729 / 0.4311 / 0.1517 / 0.0597 |
| Image | 0.7119 / 0.7652 / 0.3807 / 0.1017 | 0.7203 / 0.9230 / 0.4150 / 0.1047 | 0.6525 / 1.1746 / 0.4991 / 0.1616 |
| Text | 0.7373 / 0.6327 / 0.3565 / 0.0537 | 0.7203 / 0.6844 / 0.3809 / 0.0885 | 0.6186 / 0.8505 / 0.4710 / 0.1242 |
| Video | 0.5508 / 0.7750 / 0.5392 / 0.1471 | 0.5339 / 0.7820 / 0.5211 / 0.0513 | 0.4661 / 0.8039 / 0.5453 / 0.0784 |

### E-long step-128 per-source metrics

| Source | Count | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| MIT-IBM/CLEVRER | 118 | 0.4661 | 0.8039 | 0.5453 | 0.0784 |
| TypeSafeAI/Open-Jev | 59 | 0.4237 | 1.1141 | 0.6456 | 0.2334 |
| google/speech_commands | 118 | 0.8729 | 0.4311 | 0.1517 | 0.0597 |
| n4ze3m/typed-decisions-synth | 59 | 0.8136 | 0.5870 | 0.2964 | 0.0875 |
| sgvaze/clevr4 | 118 | 0.6525 | 1.1746 | 0.4991 | 0.1616 |

### Video question-type metrics

Cells are count / Accuracy / NLL. E-long values are one validation point and
should not be read as a settled video-task result.

| CLEVRER type | Teacher v1 reference | Candidate E-512 | E-long step 128 |
|---|---:|---:|---:|
| Temporal descriptive | 24 / 0.6667 / 0.7839 | 24 / 0.5417 / 1.1209 | 24 / 0.3333 / 1.2111 |
| Explanatory | 35 / 0.4857 / 0.8065 | 35 / 0.5429 / 0.6920 | 35 / 0.3143 / 0.7276 |
| Predictive | 35 / 0.4857 / 0.7924 | 35 / 0.5429 / 0.6962 | 35 / 0.6286 / 0.6856 |
| Counterfactual | 24 / 0.6250 / 0.6950 | 24 / 0.5000 / 0.6995 | 24 / 0.5833 / 0.6807 |

At step 128 the fixed selector chose the observed checkpoint with score
0.629342 (lower is better); it is not a final selection. The early E-long
checkpoint is below E-512 on aggregate, all four individual accuracies, and
Video Accuracy. This is descriptive only: the long-budget scheduler horizon is
different, and the run did not reach the planned 512/2,048 comparison. It does
not support or reject the claim that more training helps. The video task-type
counts are small, and no scene-paired bootstrap was run because the predeclared
512-versus-2,048 pair does not exist.

None of the four modalities reached 90% at this partial checkpoint. This is
not a final or sealed-audit result and does not change the product target.

## Failure, fix, and restart boundary

At step 128, validation predictions and the selected adapter were written, then
`model.save_pretrained()` failed while writing the full resume snapshot. The
error was `SafetensorError: Error while serializing: I/O error: specified path
not found (os error 3)`. The latest-only staging path was 261 characters on
this Windows workspace; a direct small-tensor safetensors reproduction failed
at the same path length. The temporary checkpoint directory was cleaned up,
leaving the `checkpoints/` directory empty: no latest pointer, optimizer,
scheduler, trainer state, consumed-ID cursor, selector snapshot, or RNG state
was saved. Consequently, exact resume cannot be verified.

Commit `57bdccc` shortens only the temporary staging directory name and adds a
test that writes a small safetensors file under a similarly deep Windows path.
The test passes. The previously selected `best/` adapter and step-128
predictions remain available as evidence, but cannot seed this run without
changing its initialization. A new output directory and fresh initialization
would be required for another full attempt; that would be a new run, not a
resume of experiment `20261005T013219Z-7f7e0a8a`.

## Environment and compute

- OS / Python: Windows 10 build 26300 / Python 3.11.9
- GPU: NVIDIA GeForce RTX 3080 Laptop GPU, 16,384 MiB reported VRAM
- PyTorch / CUDA: 2.6.0+cu124 / CUDA available
- Transformers / PEFT / Accelerate: 5.6.2 / 0.21.2 / 1.15.0
- safetensors / torchvision / PyAV: 0.8.0 / 0.21.0+cu124 / 18.1.0
- Video loader: Transformers fell back to torchvision; torchcodec was absent.
- Windows base-weight safetensors loader: `pread` backend.
- Requested dtype: `auto`; actual loaded base dtype: UNKNOWN because the run
  failed before writing run metadata. Model `_attn_implementation`: UNKNOWN in
  persisted run metadata. Trainer configured flash SDPA off, memory-efficient
  SDPA off, and math SDPA on.
- E-long allocated-VRAM peak: 11,885,037,568 bytes. Live `nvidia-smi` samples
  during updates were about 14.2 GiB used, with about 2.1 GiB free.
- Preflight recorded 31.8 GB free disk. Later live free-space readings varied
  between about 79 and 99 GB; the cause of that discrepancy is UNKNOWN. Disk
  exhaustion did not cause this failure.
- E-long process wall time: 2,102.465 s (35m 02s), including model load,
  initial base/reference validation, 128 updates, and step-128 validation.
  Optimizer-step times sum to 899.736 s; step-128 validation took 379.234 s.
  The remainder was setup and initial evaluations and was not individually
  timed before the failure.
- A deterministic v1 reference evaluation on the same 472 validation IDs took
  319.391 s after model load. A prior 244.109 s comparison used default
  attention settings; it is saved separately, excluded from analysis, and not
  used for selection.
- Estimated local wall across the failed training process and both
  reference-comparison processes: about 45m 22s. Cloud GPU hours and cloud
  cost: zero. Electricity cost was not measured.

## Local evidence files

The following files are under the ignored local directory
`artifacts/tiny-omni-decision-teacher-v2/candidate-e-long-budget-seed17-2048/`;
model weights, media, and checkpoints were not added to Git.

| File | SHA-256 |
|---|---|
| `preflight-report.json` | `778b27722ee368409eaa34ff3047e0e13016d1d05bfcc662d1410690e057e291` |
| `effective-config.json` | `e2a3e33fae1fef6f9e9495e16cce5fb5c40abc5ff639b37c8d340db8eb275518` |
| `run-invocation.json` | `2792fef7b5fdf0ac80b6ac1dcf46092969208d59c2d874aa56c35f64e48130ac` |
| `training-history.jsonl` (128 updates) | `4e6db5b305e234a5efa4825dbe533b1d50bad589667008758b589fa80ec72c46` |
| `validation-step-128.json` | `01a331181c25373dba72ffd8731d7be5ceb17c2b01ad2c81c6cf9075329ca98c` |
| `validation-step-128-predictions.jsonl` | `32c7be1b2d86cd29cc699a7e2caf5ee8811dd92d23144e2f40ccc2400b45794a` |
| `validation-v1-reference-evaluation.json` | `672b3bd753558df451aba63410ccad26ea1055f58e43133e264121faedba8589` |
| `validation-v1-reference-predictions.jsonl` | `53a84f64a0a46ec64e9ea321fe367406a53ea2487746cfb8eb5ef84868302bed` |
| `evaluate_v1_reference.py` | `02d8810fb0a6647a1ab30dcc76ae13571cc5a499cb991f626a6d105fb0a14088` |
| `validation-v1-reference-evaluation-default-backend.json` (excluded) | `387208c870c48d84465ff39a1dc6817e63cfeb2e97dba3b922fe8ac39781b79e` |
| `validation-v1-reference-predictions-default-backend.jsonl` (excluded) | `cf457ae9678c13831b8f0bcf87480c5be9427492e803bc9166c3ea52eeb443e7` |
| `run.log` | `4df19f8de038b2dec585d8973912af20cc882399b801a96c39e942a1abe3ec05` |

`experiments.jsonl` records the run as failed at step 128. It has no final
`run-metadata.json`, `experiment-manifest.json`, full resume snapshot, final
checkpoint, or verified reload. The only model artifact saved by this attempt
is the validation-selected step-128 adapter (96,693,360 bytes; SHA-256
`968f40de479682acd91d9fa0640ad7ff7af566402cc01883bb3531daaa3af337`), which
remains local and is not a Teacher v1/v2 replacement.

## Validation and next boundary

- Ruff: pass.
- CPU tests: 126 passed, including the deep-path safetensors regression test.
- Manifest validation: all CI model/dataset/corpus-manifest commands passed.
- GitHub Actions: run #82 for `57bdccc` succeeded: [run #82](https://github.com/Unjuno/tiny-omni-decision/actions/runs/37254763372).
- This report commit `313f3c8` also passed GitHub Actions run #83: [run #83](https://github.com/Unjuno/tiny-omni-decision/actions/runs/37256516736).
- No final sealed evaluation, paired 512/2,048 bootstrap, full learning curve,
  2,048-step finish, best-checkpoint reload, product promotion, or PR was done.

After the serialization failure, training was not restarted because the saved
state is insufficient to prove exact resume. Continuing the quality study
requires a fresh, separately named seed-17 run in a new output directory; that
is a new initialization after an infrastructure failure, not a continuation of
this partial run. No result here establishes a training-capacity, forgetting,
or temporal-representation limit.

## Subsequent exact continuation: clean-dev-v2, seed 17, 2,048-step plan

This addendum records a later continuation and does not revise the failed
128-step attempt above. The original process failed at step 1,024 while
publishing a Windows checkpoint under the OneDrive workspace. A later attempt
in that output also failed at step 1,024. A third process then passed the
resume helper's exact-state preflight and resumed from the preserved step-896
snapshot; `continuation-manifest.json` records matching checkpoint hashes,
global step 896, sampler index 3,584, consumed-ID order hash, and
`new_seed_or_hyperparameter_run: false`. Its output directory is outside
OneDrive:
`C:\Users\junny\AppData\Local\CodexArtifacts\tiny-omni-decision-teacher-v2\clean-dev-v2-seed17-2048-resume-step896-local\`.

At the latest observation on 2026-10-06 18:30 JST, this exact continuation was
still running as process 13068 at step 1,420. It had not reached the planned
2,048 updates. The live checkout was branch
`codex/teacher-quality-clean-dev-v1` at source commit
`c3b7f187b90cdaaf9538214011a8e890294508b4`; the dedicated
`codex/video-teacher-v2-long-budget` branch remained at `b1067c8`. The
continuation used the saved 2,048-step effective configuration and existing
clean-dev-v2 frozen corpora. It did not load sealed audit data.

- Effective config SHA-256: `d9d3ee1153fdc9ac7f454f509b68b5b5f4c5efc8814f017fe9d09545ee9a0b54`.
- Source config SHA-256: `4ce9af8d0aa83c5526d3d3828b77df3a5cd727351f912ed58cd9aca8c7e1773a`.
- Train corpus SHA-256: `2ebf1e75ed91b6e7021b5b202fce9ddd3cfa9feb3a9fec0cce17d3623cd4aaea`.
- Validation corpus SHA-256: `cad3fe4f87e64fcc4b908ae6ab59ae6973cbe32093551956bb3164a49c6e8edf`.
- Validation selector order SHA-256: `f3b2c57133100c5b9d8e3be35597a7925356f3a79e3f85f6b12b7ff7655d88eb`.
- Pinned base revision: `6befbaca7398925921802abd1f277b495b78b738`.

| Step | Macro Accuracy | Minimum modality Accuracy | Macro NLL | Macro Brier | Macro ECE | Video Accuracy | Best selector step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.7285 | 0.4453 | 0.8178 | 0.3362 | 0.0886 | 0.4453 | 384 |
| 1,024 | 0.7407 | 0.4434 | 0.7227 | 0.3263 | 0.0899 | 0.4434 | 1,024 |
| 1,152 | 0.7515 | 0.4434 | 0.6335 | 0.3030 | 0.0778 | 0.4434 | 1,152 |
| 1,280 | 0.7705 | 0.4570 | 0.6117 | 0.2898 | 0.0689 | 0.4570 | 1,280 |
| 1,408 | 0.7773 | 0.4707 | 0.5589 | 0.2733 | 0.0581 | 0.4707 | 1,408 |

At step 1,280, modality Accuracy / NLL / Brier / ECE was: Audio
1.0000 / 0.000085 / 0.0000004 / 0.000085; Image 0.8145 / 0.5790 / 0.2715 /
0.0729; Text 0.8105 / 0.9405 / 0.2985 / 0.1101; Video 0.4570 / 0.9272 /
0.5894 / 0.0924. The selected step-1,280 adapter hash was
`6f4bb8deb3a64791f35d7f6250a70499e7f99aa8626f41947ffac404c69d8168`.
Validation took 1,386.2 seconds and peak allocated VRAM was 11.93 GB. By that
step the run had consumed 5,120 unique examples, zero repeats, 3,511 unique
underlying assets, 319 video scenes, and 1,413 parent questions. These rolling
train metrics are not fixed-example generalization-gap measurements.

The step-1,280 validation metrics file hash is
`807f5c9921235ddbfcfbcc7eb1c1e75fc847ab6b0164d7669fa175d7918aa770`;
its predictions file hash is
`3bac5d39f32e6946e7610a13e8624f296acb77170b0d1a26925e13cd3669e260`.

At step 1,408, validation Accuracy / NLL / Brier / ECE was Audio
1.0000 / 0.000215 / 0.000003 / 0.000214; Image 0.8184 / 0.4851 / 0.2425 /
0.0617; Text 0.8203 / 0.9043 / 0.2934 / 0.1067; Video 0.4707 / 0.8461 /
0.5572 / 0.0636. Macro Accuracy was 0.7773 and minimum modality Accuracy was
0.4707. The step-1,408 adapter SHA-256 is
`e52513c43ac042095ca11ab25c0aac5fe4409b739accfbe83118634468fa2816`.
Validation took 1,363.9 seconds; peak allocated VRAM remained 11.93 GB. The
metrics and predictions SHA-256 values are respectively
`7e29c5b6c9e7e9d0b03b78c653609ee8c9c1e0869954d0099adf5b7a0693c0e4` and
`4eff0402e18b4e921228b70a51ea58bb5d63d606382878b7bb87ce8ae7473b48`.

The 512, 1,024, 1,152, and 1,280 observations use the same selector order in
this continuation. Video improves by 1.17 percentage points from step 512 to
1,280, while non-video modalities also improve. This is an interim validation
trend only; no learning-budget conclusion is made until the predeclared 1,536,
1,920, and 2,048 points and completion checks are observed. The following
remain pending: final-step evaluation, selected-checkpoint reload equivalence,
paired scene bootstrap, v1 same-selector comparison, completed compute log,
commit/push on the dedicated long-budget branch, and CI. No product promotion or
sealed-audit evaluation has occurred.

## Exact-resume audit after an unrecorded process exit

The step-1,536 validation did not produce a result file. At 2026-10-06 19:01
JST, process 13068 was absent and GPU memory was released. The latest rolling
training-history row was step 1,536, but the last complete validation,
selected adapter, trainer state, and full resume snapshot remained at step
1,408. No terminal process output was available from the expired shell session;
recent Windows Application/System logs contained no matching Python or GPU
failure event. The exit cause is therefore **UNKNOWN**. No metrics are
attributed to the incomplete step-1,536 evaluation.

Before continuing, the step-1,408 snapshot was audited without modifying it:

- `global_step=1408`, `sample_index=5632`, matching four microbatches per update.
- Replayed consumed-ID order SHA-256 matched the saved hash
  `b71b99aabfd0bfe58e0e05f956816947baf11a85526b461f943f7d62d5177030`.
- All four RNG states were present (Python, NumPy, torch CPU, and one CUDA
  stream); optimizer state had 410 entries, all at step 1,408.
- Scheduler `last_epoch=1408`, `_step_count=1409`, and its last LR was
  `1.1754254303406659e-05`, preserving the original 2,048-step cosine schedule.
- Selected adapter and its metadata/predictions hashes matched. Snapshot
  hashes for adapter, optimizer, scheduler, state, selected adapter, and
  selector evidence are recorded in the new continuation manifest.
- Config, train corpus, validation corpus, validation selector order, base
  revision, processor revision, and trainer/training source hashes passed the
  existing exact-resume checks. No audit data was loaded.

An exact continuation was started at 2026-10-06 19:09 JST in a new output
directory
`C:\Users\junny\AppData\Local\CodexArtifacts\tiny-omni-decision-teacher-v2\clean-dev-v2-seed17-2048-resume-step1408-exact\`.
Its manifest records the earlier process exit as unknown, the unvalidated
step-1,536 tail as discarded, and retains the verified parent evidence through
step 1,408. At 19:16 JST the process was still performing the same-selector
base/reference setup evaluations; training had not yet advanced beyond the
exact step-1,408 resume point. This continuation keeps the same run identity,
seed, config, corpus, scheduler state, and model initialization trajectory.

By 2026-10-06 19:52 JST, base and Teacher-v1 reference setup evaluations had
completed and training had advanced to step 1,422. The retry will produce its
own step-1,536 validation from the reproducible step-1,408 state; the previous
unpersisted step-1,536 attempt remains excluded.

## Post-run recovery: completed 2,048 updates, result finalization failed

The exact continuation subsequently reached optimizer update 2,048. The
trainer then raised `FileNotFoundError` while constructing the final result
because the frozen clean-dev-v2 corpus directory has no
`corpus-manifest.json`. The run ledger correctly retains this as a failed
process termination. No manifest was added to, or reconstructed inside, the
frozen corpus. This is an infrastructure/post-run metadata-finalization
failure, not a training-quality failure. The failure ledger timestamp is
2026-10-06 14:12:01 UTC (23:12:01 JST).

The selected-checkpoint in-memory reload/prediction equality check is before
the failing result-construction path in `trainer.py`; it completed before the
traceback. This conclusion is based on the observed control-flow location, not
on an independently persisted post-reload prediction file. Therefore the run
is **complete through training and validation, but not a clean PASS under the
requested reload-artifact gate**. The best and final adapters remain separate:
selected step 1,536 is under `best/`, and the terminal update is preserved at
`checkpoints/final-step-002048/`.

The recovered local post-run summary and paired bootstrap are stored beside
the run in AppData Local, outside Git. They do not alter the training ledger,
frozen corpus, Teacher v1, Candidate E, or prior v2 outputs. The run command
used the existing `resume_from` snapshot at step 1,408 for full optimizer,
scheduler, RNG, sampler, and selector continuation. `reference_adapter_path`
loaded Teacher v1 for a comparison evaluation only. The earlier E-512 adapter
was not used for initialization; the verified lineage was a fresh LoRA over
the pinned base, then exact resume. Run arguments and invocation metadata
confirm this distinction. The exact-resume preflight helper is retained as
`scripts/resume_teacher_quality_clean_dev_v2.py`.

The raw shell argv was not persisted, so the exact command string is UNKNOWN.
The durable invocation record identifies the config path/hash, source commit,
reference adapter, exact resume snapshot, run options, and environment; these
verified arguments are reported above rather than reconstructing a command.

### Final observed curve

All rows below use the same 2,048 validation IDs, order, preprocessing and
selector. Each optimizer update consumed four microbatches; terminal accepted
state consumed 8,192 unique training examples with zero repeated IDs. The
step-1,536 selected checkpoint represents 6,144 unique examples consumed up to
that point. Run-wide consumption and selected-checkpoint consumption are not
interchangeable.

| Step | Macro Acc. | Min modality Acc. | Macro NLL | Macro Brier | Macro ECE | Video Acc. | Selector best step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.7285 | 0.4453 | 0.8178 | 0.3362 | 0.0886 | 0.4453 | 384 |
| 1,024 | 0.7407 | 0.4434 | 0.7227 | 0.3263 | 0.0899 | 0.4434 | 1,024 |
| 1,536 | 0.7798 | 0.4766 | 0.5471 | 0.2697 | 0.0555 | 0.4766 | 1,536 |
| 1,920 | 0.7822 | 0.4824 | 0.5687 | 0.2723 | 0.0573 | 0.4824 | 1,536 |
| 2,048 | 0.7832 | 0.4824 | 0.5668 | 0.2717 | 0.0572 | 0.4824 | 1,536 |

The primary fixed-step comparison is same-run step 512 versus 2,048: Video
Accuracy increased 3.71 percentage points (0.4453 to 0.4824), while macro
Accuracy increased 5.47 points. The predeclared supporting points show that
most Video improvement occurred by step 1,536 and largely plateaued by 1,920.
The selected step is 1,536 because its fixed selector score (0.29253) was
better than the terminal score (0.31048); final-step weights did not overwrite
the selected checkpoint.

A paired scene-cluster bootstrap over the same 512 Video questions grouped
within 100 CLEVRER video scenes used 10,000 resamples and seed 17. The 2,048
minus 512 Video Accuracy delta was +0.0371, percentile 95% interval
[+0.0019, +0.0739]. This is a descriptive development interval on data used
for checkpoint selection, not an independent test or a correction for
selection bias.

| Modality | Step 1,536 Accuracy / NLL / Brier / ECE | Step 2,048 Accuracy / NLL / Brier / ECE |
|---|---|---|
| Audio | 1.0000 / 0.000458 / 0.000007 / 0.000456 | 1.0000 / 0.000194 / 0.000003 / 0.000193 |
| Image | 0.8320 / 0.483394 / 0.234894 / 0.061967 | 0.8359 / 0.479888 / 0.228043 / 0.066108 |
| Text | 0.8105 / 0.884679 / 0.295690 / 0.090317 | 0.8145 / 0.940826 / 0.301663 / 0.107476 |
| Video | 0.4766 / 0.819963 / 0.548361 / 0.069274 | 0.4824 / 0.846372 / 0.557214 / 0.055072 |

Video validation Accuracy / NLL by question type was:

| Question type (n) | Step 512 | Step 1,536 | Step 2,048 |
|---|---:|---:|---:|
| Counterfactual (100) | 0.4000 / 0.695842 | 0.4100 / 0.693147 | 0.4100 / 0.693147 |
| Explanatory (174) | 0.4655 / 0.698366 | 0.5287 / 0.695089 | 0.5345 / 0.698746 |
| Predictive (138) | 0.5000 / 0.693147 | 0.5000 / 0.693147 | 0.5000 / 0.693147 |
| Temporal descriptive (100) | 0.3800 / 1.392644 | 0.4200 / 1.339065 | 0.4400 / 1.467917 |

Predictive scores remained at binary chance with near-uniform NLL at each point.
Temporal descriptive improved in Accuracy but its NLL worsened after the best
checkpoint; these task slices are small and share scenes, so they are
diagnostic rather than independent evidence.

At step 1,536, training had consumed 6,144 unique examples, 4,106 unique
assets, 319 video scenes and 1,608 parent questions. Source counts were
CLEVRER 2,048; Clevr-4 1,536; and 512 each from Typed Decisions Synth,
MASSIVE, Open-Jev, LibriSpeech and Speech Commands. Video task counts were
422 counterfactual, 715 explanatory, 488 predictive and 423 temporal
descriptive. At step 2,048 the source counts were CLEVRER 2,731; Clevr-4
2,048; Typed Decisions Synth 683; MASSIVE 683; LibriSpeech 683; Open-Jev
682; and Speech Commands 682. Unique assets by modality were Audio 1,235,
Image 1,708, Text 2,001 and Video 319 (5,263 total). Video task counts were
617 counterfactual, 1,008 explanatory, 488 predictive and 618 temporal
descriptive. All these counts are consumed-example counts, not validation
denominators.

### Same-selector reference comparison and interpretation

The Teacher v1 reference adapter was separately evaluated by the same
clean-dev-v2 evaluator on these same IDs. It scored Audio 0.9434, Image
0.1777, Text 0.1211 and Video 0.4414. E-long's selected checkpoint scored
1.0000, 0.8320, 0.8105 and 0.4766 respectively. This is a same-selector
reference run under the current preprocessing/readout path; it must not be
confused with Teacher v1's historical validation or sealed-audit metrics.

The result supports a **modest learning-budget contribution within this
single seed and development selector**: step 2,048 beats its own step-512
Video Accuracy and selector score, with the curve flattening late. It does not
identify learning amount as the only cause. The terminal checkpoint has a
validation NLL/selector regression relative to the best step; Text ECE and
Video NLL/Brier also worsen from step 1,536 to 2,048. Video remains far below
90% and the revised roadmap's 95% shipping gate. The run does not establish a
model-family ceiling, product readiness, or permission to use sealed audit.

The run ledger records `global_step=2048`, `sample_index=8192`, best step
1,536, and effective config SHA-256
`d9d3ee1153fdc9ac7f454f509b68b5b5f4c5efc8814f017fe9d09545ee9a0b54`.
Train and validation hashes remain respectively
`2ebf1e75ed91b6e7021b5b202fce9ddd3cfa9feb3a9fec0cce17d3623cd4aaea` and
`cad3fe4f87e64fcc4b908ae6ab59ae6973cbe32093551956bb3164a49c6e8edf`;
validation order hash is `f3b2c57133100c5b9d8e3be35597a7925356f3a79e3f85f6b12b7ff7655d88eb`.
The pinned base and processor revision is
`6befbaca7398925921802abd1f277b495b78b738`. The selected adapter SHA-256 is
`0497a25742451daeb2e08c7879dc221d174a17bf451e4ab2de0f3af6223b5009`; the
terminal adapter SHA-256 is
`d17534df7b5141390c338a3f74e1c79aa4bf75be925d239036715372a4cefd62`.
Step-1,536 and step-2,048 metric/prediction hashes are retained in the local
recovery summary and the corresponding files; the terminal metric hash is
`69e5cc5a8f7402b36409983fc3b06b896c20427f580c7348cd6201df3c1143cb` and
prediction hash is
`f19109a0421e6fbf381cfd4f80c0a7194cd4d39ad7f28922f5a1bcf119adbdec`.

The active worktree in which training ran was `codex/teacher-quality-clean-dev-v1`
at source commit `c3b7f187b90cdaaf9538214011a8e890294508b4`; result packaging is
being recorded separately on `codex/video-teacher-v2-long-budget`. No model
weights, corpus media, or large checkpoints are added to Git. No sealed audit,
ternary conversion, Recovery training, product promotion or extra seed was
run.

### Hardware, environment and elapsed compute

- Local GPU only: NVIDIA GeForce RTX 3080 Laptop GPU, 16,384 MiB, driver
  616.92; no cloud compute or purchase. Electricity cost was not measured.
- Windows 10 build 26300, Python 3.11.9, PyTorch 2.6.0+cu124,
  torchvision 0.21.0+cu124, Transformers 5.6.2, PEFT 0.21.2, Accelerate
  1.15.0, and PyAV 18.1.0. torchcodec was absent; the video path used the
  recorded Transformers/torchvision loader fallback. BF16 was configured.
- Attention backend: PyTorch math SDP; flash and memory-efficient SDP were
  disabled. The bounded-memory Windows safetensors loader patch was active.
- Peak allocated VRAM recorded at the final validation: 11,929,597,952 bytes.
- Logged optimizer-step time sums to 11,255.342 seconds over 2,048 updates
  (5.50 seconds/update on average). Training-history validation calls sum to
  22,023.655 seconds; setup/reference evaluations add 7,282.484 seconds.
  Trainer cumulative active elapsed time is 40,685.907 seconds (11.30 hours).
  This is accumulated work across attempts and exact resume, not uninterrupted
  calendar duration. The recorded experiment began 2026-10-05 23:38 UTC; the
  final post-run failure was logged 2026-10-06 14:12 UTC. A validation call
  around step 2,048 took 1,408.469 seconds. The calendar interval includes
  retries and idle gaps and must not be read as GPU-hours.
- Cloud cost: $0. No 4,096-step extension or extra seed was run.

## Follow-up: manifest-name repair and independent selected-checkpoint reload

The post-run exception said `corpus-manifest.json` was missing. The frozen
corpus itself was not missing provenance: its builder wrote `manifest.json`,
whose SHA-256 is `f5efe676063d90596eecd8fb88c1fdf43c039c11ddc96330f6f16bf0c0e867d1`,
exactly matching the hash recorded in the run invocation. The trainer assumed
only the other filename. A resolver now accepts either supported name and
rejects conflicting manifests if both exist. The frozen corpus and original
failed run ledger were left untouched.

The selected step-1,536 adapter was independently reloaded in a fresh local
process and evaluated on the same ordered 2,048 validation IDs. All prediction
records matched the saved step-1,536 predictions exactly, and macro,
per-modality, per-source, and Video question-type metrics matched exactly.
The adapter hash remains
`0497a25742451daeb2e08c7879dc221d174a17bf451e4ab2de0f3af6223b5009`.
Evaluation took 1,319.093 seconds and peaked at 11,426,224,128 allocated VRAM
bytes. The sealed audit was not loaded. This verifies the selected checkpoint
reload gate; it does not rewrite the original process's finalization-failure
event or claim product readiness.

A separate research-only freeze record is stored outside Git at
`C:\Users\junny\AppData\Local\CodexArtifacts\teacher-quality-reloads\clean-dev-v2-step1536-20261007\precompression-master-freeze.json`
(SHA-256
`b1eb22eb47b7d03487df17b62fe952e6fd89893e5371cf1f8dad30f86b2b5d83`). It
points to the existing selected adapter without copying or modifying the
checkpoint. Text/Image/Audio/Video Accuracy on this development selector is
0.8105 / 0.8320 / 1.0000 / 0.4766; the frozen Teacher-quality goal remains
unmet. This checkpoint may serve as an engineering-research pre-compression
Master under the updated roadmap, not as a product-qualified Teacher.
