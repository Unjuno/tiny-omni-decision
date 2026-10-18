# Small decoder learning and fixed transfer results

Execution PASS; quality objective unmet. No product promotion or sealed audit.
Fresh pinned SmolLM2-135M-Instruct; 134,515,008 resident parameters before
quantization. Only last four decoder blocks and final norm trained:
14,160,960 parameters. FP32/SDPA, seed17, AdamW constant2e-5, batch8,
2,048 updates. Full prompts, no truncation. See the frozen learning,
training-fit and transfer protocols for exact inputs and selection rules.

Source commit: `9646d669899d142a6159e092c4cd6060f7839c87`.
Start: 2026-10-11T19:36:50.458241+00:00; end: 2026-10-11T20:02:24.372605+00:00.
Normal scheduled completion. Best is earliest minimum mean validation NLL
across original and curriculum; best and final both step2,048, separate files.
Their serialization hashes differ; best reload predictions agree exactly.
Frozen tensor hash matches before/after. Weights-only, not exact resume.

## Learning curve

| Step | Original Acc | Original NLL | Curriculum Acc | Curriculum NLL | Selector |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.309055 | 1.313408 | 0.231934 | 1.489197 | 1.401302 |
| 256 | 0.422047 | 1.131342 | 0.258301 | 1.487274 | 1.309308 |
| 512 | 0.461811 | 1.067263 | 0.253906 | 1.446916 | 1.257089 |
| 768 | 0.477559 | 1.053701 | 0.230957 | 1.474482 | 1.264091 |
| 1024 | 0.480315 | 1.050843 | 0.246094 | 1.454693 | 1.252768 |
| 1280 | 0.489370 | 1.034379 | 0.260742 | 1.447201 | 1.240790 |
| 1536 | 0.484646 | 1.032095 | 0.232422 | 1.473183 | 1.252639 |
| 1792 | 0.492126 | 1.038008 | 0.227539 | 1.459908 | 1.248958 |
| 2048 | 0.499213 | 1.027002 | 0.242188 | 1.449104 | 1.238053 |

Whole run and selected checkpoint each consumed16,384 unique questions,
8,192 per source,6,056 underlying states; zero repeated question IDs.
Original validation2,540 and curriculum2,048 questions. Per-task metrics,
Brier/ECE and all scheduled predictions are preserved in local artifacts.
Original accuracy improves30.91%→49.92%; curriculum remains24.22%.
Best selector is at the endpoint: no demonstrated global plateau.
Curriculum fluctuates near chance, not evidence that general reasoning was learned.
Rolling training CE is not a fixed-example generalization-gap estimate.

## Fixed seen-training diagnostic

First1,024 consumed questions,512/source, selected weights only. Original
accuracy51.17%, curriculum22.27%. Low curriculum fit also occurs on seen
examples. This supports a fitting/optimization/representation bottleneck
within this run, not a unique capacity diagnosis or impossibility claim.
Training curriculum ranges/wording differ from narrow validation, so their
difference is not an identically distributed generalization gap.

## Fixed new-state transfer

128 new states,1,024 base questions, six paired arms;6,144 rows and5,632
distinct contents. Known templates; development diagnostics, not blind tests.
BERT has different pretraining and inherited training; not equal compute.

| Model | Arm | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| base | base | 0.220703 | 1.504639 | 0.791355 | 0.119994 |
| base | option_reverse | 0.246094 | 1.493679 | 0.785022 | 0.093313 |
| base | state_reverse | 0.213867 | 1.505064 | 0.791657 | 0.123833 |
| base | expanded | 0.223633 | 1.498962 | 0.788571 | 0.110984 |
| base | domain | 0.239258 | 1.476047 | 0.778417 | 0.074579 |
| base | expanded_domain | 0.238281 | 1.474639 | 0.777813 | 0.078389 |
| decoder | base | 0.234375 | 1.448113 | 0.765222 | 0.034558 |
| decoder | option_reverse | 0.243164 | 1.452589 | 0.767113 | 0.025675 |
| decoder | state_reverse | 0.248047 | 1.448218 | 0.765192 | 0.020503 |
| decoder | expanded | 0.236328 | 1.448583 | 0.765427 | 0.036184 |
| decoder | domain | 0.244141 | 1.463934 | 0.771662 | 0.053034 |
| decoder | expanded_domain | 0.250000 | 1.460690 | 0.770264 | 0.044400 |
| bert | base | 0.810547 | 0.469910 | 0.262276 | 0.032092 |
| bert | option_reverse | 0.810547 | 0.469910 | 0.262276 | 0.032092 |
| bert | state_reverse | 0.810547 | 0.462289 | 0.257849 | 0.045888 |
| bert | expanded | 0.713867 | 0.700460 | 0.386378 | 0.053414 |
| bert | domain | 0.381836 | 1.322590 | 0.716242 | 0.085333 |
| bert | expanded_domain | 0.318359 | 1.751607 | 0.872355 | 0.275722 |

JSON includes every task, opposite-condition pair success, candidate/state
order identity consistency and paired128-state bootstrap versus base
(10,000 draws, seed17, percentile95%, no simultaneous correction).
No checkpoints or temperatures selected from transfer outcomes.
Adapted decoder base-arm accuracy23.44% versus pretrained22.07%:
difference+1.37pp, paired95% interval[-2.25,+4.88]pp. Every decoder
arm's interval versus pretrained includes zero. The BERT reference gets
81.05% on the base arm but38.18% on domain terms and31.84% when expanded.
Candidate reversal preserves selected identity only12.60% for adapted
decoder, versus100% for independent-candidate BERT. This is a major
ordering weakness in this decoder pipeline, not general decision quality.
Keep BERT as the stronger measured text reference, without claiming95%.

## Resources and preservation

- `small-decoder-learning-01`: 1533.892s, peak allocated VRAM 2,517,472,256 bytes.
- `small-decoder-train-fit-01`: 33.265s, peak allocated VRAM 1,003,153,920 bytes.
- `decoder-transfer-eval-01`: 172.394s, peak allocated VRAM 664,519,168 bytes.
- Scheduled validation total: 848.734s; exact reload adds
  another full validation within total run time. Remaining time includes
  training, saving and hashing, not a pure training-only measurement.
- RTX3080Laptop16GB; local compute, cloud cost$0. Environment pinned by
  feasibility record; no package upgrade during this run.
- Prior checkpoints/corpora preserved. No sensor, image/audio/video training
  in this experiment; no quantization, no final sealed evaluation.
- All files live under `C:/CodexArtifacts/pretrained-reuse-r2/` plus the
  three run names above. JSON records config/history/prediction hashes;
  result records best/final tensor-file hashes. Large weights stay outside Git.
- Ruff passes;374 CPU tests pass,1 skipped; model/dataset/catalog
  manifest validation passes. GitHub status must be checked on the pushed commit.

Four-modality95–98% remains unmet. Do not replace the smaller BERT reference
based on parameter budget alone. Further architecture changes require a
separate predeclared diagnostic; this run does not justify an automatic
larger budget or repeated evaluation of the same transfer probe.
