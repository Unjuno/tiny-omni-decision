# R2 full-extent × unique-coverage long comparison

## Question and scope

The existing 32,768-update unique-coverage run used center-crop visual
features. The existing full-extent run used the repeated 6,656-question pool.
This experiment adds the fourth cell of that fixed crop × coverage comparison:
full-extent visual features with the existing fresh-coverage sampler. It changes
only the sampler relative to the full-extent repeat arm. It does not test new
Video examples: the pinned fresh pool adds 2,048 unique Text, Image, and Audio
questions and zero Video questions.

The run uses the same seed17 initialization, 32,768-update cosine schedule,
8 microbatches per update, frozen encoders, 889,345 trainable readout parameters,
and 24,346 ordered validation IDs as the completed comparators. The fixed
checkpoint train probe is the same 6,656 original-pool questions for every arm;
the full 12,800-row parent-plus-added TRAIN union is scored only at the endpoint.
The existing `native-full-extent-fit-01` and
`native-coverage-long-fresh-fit-01` artifacts remain read-only.

## Implementation

`train_native_full_extent.py` now accepts a separate config path and a `fresh`
policy while retaining the original default config and `repeat` behavior. The
fresh policy is rejected unless its regenerated full-run sample order exactly
matches the pinned completed center-crop fresh arm. The shared training probe
uses the original 6,656-row parent set in corpus order, so per-step train-probe
metrics remain comparable across repeat and fresh conditions. The fixed
12,800-row endpoint union continues to report seen versus unseen rows.

The CPU-only `prepare_native_full_extent_unique_coverage.py` checks pinned input
hashes and regenerates the pool, complete 32,768-step plan, LR curve,
validation IDs, and common train probe before any model/GPU load. It writes only
to a new output directory. The new config is
`configs/pretrained_reuse/native_full_extent_unique_coverage_long_v1.json`.

## Fixed conditions

- Fresh pool and occurrence plan: `native-coverage-plan-02/fresh-pool.json` and
  `fresh-plan.json`, SHA-pinned in the config.
- Full-extent feature cache: existing immutable
  `native-full-extent-cache-01`; all 8,936 Image and 456 Video feature banks.
- Fresh center-crop comparator: `native-coverage-long-fresh-fit-01`, pinned
  result, sample order, LR plan, validation IDs, selected/final checkpoints,
  and step 8,192/32,768 predictions.
- Full-extent repeat comparator: `native-full-extent-fit-01`; validation IDs,
  common train probe, and LR curve remain identical.
- Training: 32,768 optimizer updates; scheduled validation at
  0/512/1,024/2,048/4,096/8,192/12,288/16,384/24,576/32,768; no performance
  early stop; best checkpoint remains separate from final.
- Evaluation: same fixed development validation only. No sealed audit, model
  promotion, extra seed, quantization, or encoder update.

## Preflight status at record time (historical)

The test-first implementation passed its targeted tests (28 passed) and Ruff.
The CPU-only pinned preflight passed at
`C:\CodexArtifacts\pretrained-reuse-r2\native-full-extent-fresh-long-preflight-02`.
Its result SHA-256 is
`346a99beb7e6bde990c38212d57fe3abe39c1b541074a2bbda71d5a6f9f73847`; source
commit `eecb2c076016f63c2f27e8f68c05eba612d4b3df`; effective config SHA-256
`7758008edcfcedee819a6ed15b1f546f865568b2bfb56cc57a42ac2202138cb5`. All five
preflight artifact hashes were rechecked against the saved files with zero
mismatches. The independent CPU sampler check regenerated 16,384 pool
occurrences with 12,800 unique questions, including 2,048 additional unique
rows each for Text, Image, and Audio, no new Video rows, and 2,048 repeated
occurrences within the pool. Two full pool cycles yield 262,144 example
presentations. The sample-order
SHA-256 matches the completed center-crop fresh run exactly. The LR SHA-256 is
`14234584342acb62df7d3987fdc835ec7beec3b04f5070ae864a9eac0c6becc9`; validation
IDs SHA-256 is
`db7d0f1d09319a48a7725ed36ccb515cdcc85a409f019400cf0062be38768f30`; the common
parent-probe SHA-256 is
`d6293594fc44a9cb6d1bf251d08b1815550d555388214927cf2937a30fc76c96`.

Preflight attempt-01 at
`C:\CodexArtifacts\pretrained-reuse-r2\native-full-extent-fresh-long-preflight-01`
is preserved but invalid: a Windows text-writer newline conversion caused the
receipt's predicted hashes to differ from the files. No model or GPU was loaded
in that attempt. The writer now writes the exact pre-hashed bytes, with a
regression test; attempt-02 is the valid receipt above. The CPU preflight pins
the full-extent cache receipt but does not inspect its bank metadata or
stream/hash all 3.3GB of visual feature arrays. During the separate GPU smoke,
before model load, the runner checks each bank's path, SHA, shape, dtype, and
finite values.

At the time of this preflight record, no GPU fit had started: unrelated live
process PID26920 was responsive on the RTX 3080 Laptop and occupied
16,122/16,384 MiB; it was left untouched. The planned smoke/fit/replay command
sequence was later run in fresh attempt-04 directories, recorded below.

```powershell
python -u -m scripts.train_native_full_extent --phase smoke --policy fresh --config configs/pretrained_reuse/native_full_extent_unique_coverage_long_v1.json --output C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-fresh-long-smoke-01
python -u -m scripts.train_native_full_extent --phase train --policy fresh --config configs/pretrained_reuse/native_full_extent_unique_coverage_long_v1.json --output C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-fresh-long-fit-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-fresh-long-smoke-01 --reference-sha256 <smoke-result-sha256>
python -u -m scripts.train_native_full_extent --phase replay --policy fresh --config configs/pretrained_reuse/native_full_extent_unique_coverage_long_v1.json --output C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-fresh-long-replay-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/native-full-extent-fresh-long-fit-01 --reference-sha256 <fit-result-sha256>
```

The runner itself fails closed while the GPU is occupied. The later run outcome
and fixed-validation comparison are recorded below; they do not establish a
blind result or justify model promotion.

## Runner attempt and provenance repair — 2026-10-25

The first HotGPU smoke attempt is preserved at
`C:\CodexArtifacts\pretrained-reuse-r2\native-full-extent-fresh-long-smoke-01`.
It ended `INCOMPLETE` at step 0, before model loading or optimizer updates,
because a pinned source file in the visual-cache receipt was compared to a newer
checkout file. The receipt pins source commit
`4ca45b517328d77f2be644c625920f8cdd22624f`; the expected hash for
`scripts/extract_frozen_videoprism_features.py` is
`0318c43e94337dbbb395d400d6c6027c2d389b8789b97b973ac37d1086129c65`. Its
current checkout hash is different, but the original commit's Git blob matches
the expected SHA-256. An audit of all 1,050 `source_pins` found six checkout
mismatches, and all six matched the exact committed blobs. No file in that
external worktree was changed.

The runner now verifies changed external source files against the visual-cache
receipt's pinned commit and records each Git-blob fallback. Its local HotGPU
guard permits only the guard process and the verified Cursor GPU renderer;
unknown GPU processes and active utilization still block launch. The measured
free memory must remain below 14,000 MiB and at least 12,288 MiB. The guard
reserves enough memory to keep Orbitune's 14,000 MiB restart threshold blocked
for the whole smoke/train/replay sequence and keeps that reservation for all
three phases. The live Orbitune
process remains untouched. Tests cover the UI-process exception and rejection
of any unrecognized auxiliary PID.

Protected sequence attempt-02 is preserved in
`native-full-extent-fresh-long-guard-02.stdout.log`, its stderr log, and
`native-full-extent-fresh-long-sequence-02.json`. The guard reserved 2,418 MiB
and kept free memory at 13,744 MiB, but the child gate rejected the guard's
`-m scripts.run_with_hotgpu_guard` command form because it only recognized the
direct `.py` spelling. This happened before an output directory was created,
before model loading, and at step 0. No training or quality measurement ran.
The gate now recognizes both verified invocation forms. Attempt-03 was later
run after CI and a matching-source CPU preflight pass; its termination and the
successful attempt-04 result are recorded below.

## GPU run and matched factorial comparison — 2026-10-25

### Execution outcome

The approved experiment design completed as a single successful attempt-04
sequence on the existing local RTX 3080 Laptop GPU. It was launched under the
HotGPU guard, which reserved enough memory to keep the restart threshold
blocked through smoke, fit, and replay. The Orbitune supervisor was not
modified. No cloud compute was used; recorded cloud cost is $0. No sealed-audit
data was loaded, no quantization or encoder update occurred, and none of the
older v1/v2, corpus, or checkpoint artifacts were overwritten. The new model
weights and media remain outside Git under `C:/CodexArtifacts`.

| Phase | UTC interval | Outcome | Duration |
|---|---|---|---:|
| Smoke, `...smoke-04` | 12:26:11–12:28:00 | PASS, 32 updates, reload max abs 0 | 108.7 s |
| Fit, `...fit-04` | 12:28:03–13:19:28 | PASS, 32,768 updates, best step 4,096 | 3,084.6 s end-to-end |
| Replay, `...replay-04` | 13:19:31–13:24:19 | PASS, best/final replay max abs 0 | 288.4 s |

The run used Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, NumPy 2.4.6,
float32, TF32 disabled, and `NVIDIA GeForce RTX 3080 Laptop GPU` (driver
616.92, 16 GiB). The process reported 199.5 MiB peak CUDA allocation / 312 MiB
reserved and 3.50 GiB peak host working set. During the guarded sequence,
`nvidia-smi` showed about 3.2 GiB total GPU use including the guard's
reservation; that figure is not the model's peak allocation. The model has
196,254,874 total resident parameters and 889,345 trainable readout parameters.

The effective config is
`configs/pretrained_reuse/native_full_extent_unique_coverage_long_v1.json`,
SHA-256 `7758008edcfcedee819a6ed15b1f546f865568b2bfb56cc57a42ac2202138cb5`.
It pins seed 17, 32,768 updates, 8 microbatches per update, AdamW, LR 0.001,
62-update warmup, cosine scheduler, no performance early stopping, full-extent
features, and fresh sampling. The ordered validation IDs SHA-256 is
`db7d0f1d09319a48a7725ed36ccb515cdcc85a409f019400cf0062be38768f30`; the
planned LR schedule SHA-256 is
`14234584342acb62df7d3987fdc835ec7beec3b04f5070ae864a9eac0c6becc9`. The
CPU preflight result SHA-256 is
`DA0732F1C658C15134211EFEB169CEC461DA88D22561C6615BE88CC99028F961`.

Run accounting distinguishes optimizer updates from presented examples:
32,768 updates × 8 microbatches = 262,144 example presentations over the full
run. The sampler consumed 12,800 unique questions (8,417 observations, 5,385
unique media assets, 5,461 groups) and 249,344 repeated question
presentations. At the selected checkpoint, 4,096 updates had consumed 32,768
presentations; all 12,800 unique questions had been seen, with 19,968 repeats.
The selected checkpoint was step 4,096 by the unchanged hierarchical macro-NLL
selector. The final step was retained separately and did not overwrite best.
Per-source full-run presentations were: CLEVR-4 65,536, CLEVRER 65,536,
Lombard 32,768, STARSS 32,768, condition relevance 21,856,
programmatic decisions 21,824, curriculum 15,904, and original text 5,952.
The fresh pool contributes 2,048 additional unique examples each for Text,
Image, and Audio, and none for Video.

### Interrupted attempt and retry

Protected attempt-03 reached step 8,320 and preserved checkpoints through
8,192, but ended `INCOMPLETE` when Windows returned `PermissionError` while
replacing `result.json.tmp` with `result.json`. The traceback is saved in its
result file; its result SHA-256 is recorded above. It consumed 1,096.1 seconds
before failure. Including attempt-03 and the successful smoke/fit/replay phases,
local compute time was about 4,578 seconds (1 hour 16 minutes); cloud cost was
zero. The failure coincided with reading the live result file during
monitoring; that is a plausible cause, not proven. I stopped reading result
files while attempt-04 was active. Because there is no exact resume path for
this runner, attempt-04 restarted from the pinned original seed-17 readout and
the full fixed schedule. It is not a continuation of attempt-03. The two
attempts' early validation curves match through step 8,192, which is a useful
sampler/seed consistency check, not a substitute for the completed run.

### Learning curve (same 24,346 development-validation rows)

| Updates | Accuracy | NLL | Brier | ECE | Selector best so far |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1604 | 2.2226 | 0.7997 | 0.0802 | 0 |
| 512 | 0.2963 | 1.7301 | 0.7485 | 0.0720 | 512 |
| 1,024 | 0.3060 | 1.7184 | 0.7485 | 0.0980 | 1,024 |
| 2,048 | 0.3214 | 1.7164 | 0.7440 | 0.0987 | 2,048 |
| 4,096 | 0.3379 | **1.6425** | 0.7210 | 0.0877 | **4,096** |
| 8,192 | 0.3669 | 1.6580 | 0.7254 | 0.1340 | 4,096 |
| 12,288 | 0.3921 | 1.7501 | 0.7257 | 0.1555 | 4,096 |
| 16,384 | 0.4035 | 1.9741 | 0.7448 | 0.1884 | 4,096 |
| 24,576 | 0.4516 | 2.7742 | 0.7538 | 0.2332 | 4,096 |
| 32,768 | 0.4723 | 3.4129 | 0.7523 | 0.2471 | 4,096 |

The main within-run comparison (512 → 32,768) shows aggregate accuracy rising
by 17.60 points while macro NLL worsens by 1.683. Video accuracy falls from
34.75% to 33.00%; Video NLL rises from 1.354 to 9.102 and ECE from 0.048 to
0.594. Thus the predeclared joint criterion—Video accuracy and selector both
improve—is **not supported within this budget**. Accuracy-only gains are
accompanied by severe calibration deterioration, especially for Video. This
does not establish an architecture-wide quality ceiling.

Per-modality validation metrics at the fixed 512-step point, selected checkpoint,
and final checkpoint:

| Modality / rows | Step | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Audio / 11,166 | 512 | 0.5136 | 1.2157 | 0.5612 | 0.1030 |
|  | 4,096 best | 0.5270 | 1.1729 | 0.5302 | 0.0802 |
|  | 32,768 | 0.7082 | 0.9355 | 0.3669 | 0.0833 |
| Image / 2,048 | 512 | 0.1357 | 2.2750 | 0.8936 | 0.0686 |
|  | 4,096 best | 0.2974 | 1.9418 | 0.7924 | 0.0750 |
|  | 32,768 | 0.5972 | 1.2368 | 0.5434 | 0.1225 |
| Text / 10,732 | 512 | 0.1883 | 2.0758 | 0.8395 | 0.0679 |
|  | 4,096 best | 0.1895 | 2.0887 | 0.8391 | 0.0684 |
|  | 32,768 | 0.2539 | 2.3775 | 0.8732 | 0.1892 |
| Video / 400 | 512 | 0.3475 | 1.3538 | 0.6997 | 0.0484 |
|  | 4,096 best | 0.3375 | 1.3668 | 0.7223 | 0.1272 |
|  | 32,768 | 0.3300 | 9.1019 | 1.2257 | 0.5936 |

The train-probe arrays use a fixed 6,656-row parent subset and the endpoint
union uses 12,800 rows. Rolling one-batch train CE in `training-history.jsonl`
is not a fixed-set train metric and is not interpreted as a train/validation
generalization gap. The selected best checkpoint's fixed parent-train probe
macro Accuracy/NLL/Brier/ECE was 37.38% / 1.537 / 0.679 / 0.025; the final
checkpoint scored 64.11% / 0.922 / 0.428 / 0.017 on that same probe. In
contrast, validation NLL/calibration deteriorated after step 4,096. This is a
train/validation divergence signal even while validation Accuracy rises. The
selector was evaluated repeatedly on the same development set, so these are
selection-affected development metrics, not a blind estimate.

### Four-cell fixed-endpoint comparison

The CPU factorial analyzer first failed at CLI entry because its `main()` call
preceded the definitions used on the fully populated path; its tests had only
exercised imported functions. This was fixed by moving the entrypoint after the
definitions, adding an explicit full-fresh directory option, and adding a
hash-pinned alternate-directory test. The analyzer compares step 32,768 across
all four arms; it does not choose checkpoints. All cells passed exact
validation-ID order, result, and prediction hash checks.

| Feature extent × sampler | Best step (context only) | Macro Accuracy | Macro NLL | Macro Brier | Macro ECE | Audio Acc. | Image Acc. | Text Acc. | Video Acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Center × repeat | 8,192 | 0.4570 | 4.8481 | 0.8155 | 0.3116 | 0.6798 | 0.5469 | 0.2287 | 0.3725 |
| Center × fresh | 8,192 | 0.4877 | 4.1994 | 0.7310 | 0.2458 | 0.7219 | 0.6055 | 0.2383 | 0.3850 |
| Full × repeat | 12,288 | 0.4168 | 3.2060 | 0.8364 | 0.3048 | 0.6099 | 0.4878 | 0.2321 | 0.3375 |
| Full × fresh | 4,096 | 0.4723 | 3.4129 | 0.7523 | 0.2471 | 0.7082 | 0.5972 | 0.2539 | 0.3300 |

Fresh coverage improved macro Accuracy in both extent conditions: +3.07 points
for center crop (paired group-bootstrap 95% CI +1.51 to +4.78) and +5.35
points for full extent (+3.17 to +8.76). Macro Brier improved in both. The
full-extent fresh-vs-repeat NLL contrast was +0.181 (95% CI −0.102 to +0.459),
so its direction is uncertain, and the Video accuracy contrast was −0.75
points. The extent×coverage interaction for macro Accuracy was +2.27 points
(95% CI −0.36 to +5.91); for macro NLL it was +0.845 (95% CI +0.273 to
+1.419). This suggests the coverage effect is not a uniform improvement across
metrics, particularly for NLL. Bootstrap: 10,000 paired resamples, seed 17,
groups resampled within modality; ECE is point-estimated only. All four arms
use one seed and the same exposed validation, so uncertainty intervals do not
remove training-seed or checkpoint-selection effects.

### Artifacts, hashes, and verification

Attempt-03 result SHA-256:
`B439D8078153C2CB73DA61003235798D3BF9AF3665C14FDD1BDF2BDD0F70486C`.
Successful attempt-04 hashes:

- Smoke result: `66A01799B92A3D89C1CFF188542DBBB8BFDF28496F0319F3513FE9A5621979AE`.
- Fit result: `DCC2190DA3167D5FDAA2F30BB166A999D3AE0104D5A08E55CC1181F855C0DF54`.
- Training history: `71E3A4D4EC59A77C08F02420F24D09590CC456D5265916C237E4AAFD310B45BB`.
- Learning curve: `F3D90BBBDD0931EABF57FA8CDD7ABA0AB9A12B969F1407126A711AC1C412C81B`.
- Replay result: `F87DCFB129BE67BFF7D178BA49A470F809F6CA70A67CE446F5C195A623D53AE1`.
- Guarded sequence: `63F2B86540A59B49470CC50BA2A9EDB727FA7DA48720491E72F7EE02D0A7A8BD`.
- Factorial comparison: `0C61240D3E46AECF0EE41902F0097B281DD8E4A139CF52C38AE8FC15BD59EAC2`.

Outputs are preserved at `C:/CodexArtifacts/pretrained-reuse-r2/` under
`native-full-extent-fresh-long-{smoke,fit,replay}-04` and
`native-coverage-factorial-comparison-03`. The fit includes checkpoints at all
scheduled validation steps, separate best and final weights, fixed-probe
predictions, endpoint accounting, and per-example validation predictions.
After moving the CLI entrypoint and adding the alternate fit-directory option,
`python -m pytest tests/test_native_coverage_factorial.py -q` passed (10 tests)
and Ruff passed. Full verification also passed: `python -m pytest -q` (2,460
passed, 11 skipped), `ruff check .`, all CI manifest/catalog validation and
dataset-manifest audit commands. The direct `pytest.exe` launcher was blocked
by Windows Application Control; invoking the same installed pytest via Python
module succeeded. GitHub Actions status is recorded with the pushed commit.
The `ci` workflow passed for commit `eb416be934a79507883f85cc849197a70f057992`
in [run 40198130727](https://github.com/Unjuno/tiny-omni-decision/actions/runs/40198130727).

### Decision

**Execution: PASS.** The full 32,768-update budget completed, best and final
checkpoints were retained separately, and replay matched saved best/final
predictions exactly. **Learning-amount hypothesis: NOT SUPPORTED WITHIN THIS
BUDGET for Video.** Video accuracy did not improve from step 512 to step
32,768, and the fixed selector peaked at 4,096; later Video NLL/calibration
deteriorated sharply. Overall accuracy rose, but it does not override those
Video and calibration results. Fresh coverage did improve several endpoint
metrics for Image/Audio and modestly for Text, while Video did not gain. This
single-seed exposed-validation comparison neither proves that more training
cannot help nor supports product promotion. Do not launch an extra seed or
extend the training budget without a separately reviewed design; first
diagnose Video miscalibration and the train-probe/validation divergence.
