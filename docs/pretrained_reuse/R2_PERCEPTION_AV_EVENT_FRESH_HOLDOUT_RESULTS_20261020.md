# Perception AV event fresh holdout — fixed-checkpoint transfer results

## Result and scope

The preregistered fixed-checkpoint comparison completed on 200 previously
unused official Perception Test TRAIN videos, covering 504 annotated
audio-event-to-object decisions. It compares the already selected joint
audio+visual readout against the capacity-matched video-only readout on the
same event IDs, candidate tracks, and deterministic option order. This is a
narrow transfer check for conditional source correspondence. It is **not** a
blind validation or sealed audit: source annotations were parsed earlier for
feasibility, participant/source-recording independence is unavailable, and
these records belong to the public TRAIN split.

No model was trained, no optimizer was run, no checkpoint was selected, and no
official VALID/TEST or sealed-audit data were opened. No model or checkpoint
was modified. The fixed-readout result does not qualify the complete Omni
product or establish that both modalities are causally necessary.

## Fixed inputs and artifacts

- Plan: SHA-256
  `887a1eea4b5a7b82d441025842465a09d0ffb33c97787073042ad1a1680f8eb3`.
- Source: official Perception Test revision
  `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`, TRAIN media archive generation
  `1685446268139925`; source README states software Apache-2.0 and other
  materials CC-BY-4.0. The public annotation metadata does not identify
  participant/source-recording groups.
- Acquisition receipt: `PASS_ACQUIRED_UNVERIFIED_DECODE`, 200/200 planned
  videos, 2,917,471,792 bytes, 200 unique media hashes, no overlap with earlier
  media IDs or exact media hashes. Manifest SHA-256
  `8591dbb0c173c70780c6d6066f18b2d7e3f29ba43c215ae0b2a251a07e2ac118`.
- Complete CPU PTS receipt: `PASS_COMPLETE_FRESH_HOLDOUT_PTS_AUDIT`, 200 videos
  / 504 events, 0 incomplete audio intervals and 0 right-censored events.
  Receipt SHA-256
  `bdc269632ce31d8dd3070effea96ee70d0993d63cf1e445301a45f679e3f5374`;
  it records 1,791 unique raw media hashes across prior artifacts and this
  holdout.
- Feature receipt: `PASS_FROZEN_FRESH_HOLDOUT_FEATURES`, 200 videos / 504
  events. Feature manifest SHA-256
  `5a4fed0a0419fe1091364b5b240b371e47c64ffff8516a2fe179eaac998fd1a4`;
  result SHA-256
  `d4eb4e57b1f357f32a6cc0e039a5f2e34f9460eae7fc1d540dcf5fc33a2288bc`.
  The external feature directory contains 1,011 files / 2,979,652,492 bytes.
- Evaluation receipt: `PASS_FRESH_FIXED_CHECKPOINT_EVALUATION`, 504 events /
  200 videos; result SHA-256
  `19fcae15cfe7ac5d385cf2d94cde4fe80468f61b7b1b90ec7b6ba5b8bda96fab`.
  Joint prediction SHA-256
  `ce1132d8cd9420931c94aeed9fd97ec6e0fe9e75da496071b1bd0af2fe2f8642`;
  video-only prediction SHA-256
  `4acac4d853302a369aca1aabbe3368bf923a0a0ee41fcf50a4a8a74f3a745a71`.
- The joint checkpoint is unchanged at SHA-256
  `be9a09f1c57c679b176d69c9a14825d4d38d053bee4b2ea5818ee108b7da8a36`,
  selected epoch 2, 246,658 active readout parameters. The matched video-only
  checkpoint is unchanged at SHA-256
  `208b283eff144f300c9d726c67eb5a063947eac87f8d4b6295764042e991cabb`,
  selected epoch 3, 246,115 active readout parameters.
- Frozen encoders: DINOv2-small revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`, 22,056,576 parameters;
  CLAP audio branch revision
  `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`, 28,190,872 parameters.
  These counts describe this task's frozen feature components, not the full
  resident Omni model or a deployable package.
- Feature extraction code ran from `codex/r2-perception-fresh-holdout` at
  source commit `4c75d001d73c26df2cbb3e4ff39cb87de665c1af`; extractor SHA-256
  `b7c5d839bcd5b35ed9345dd8821412d760032f50d4a4d035a7ff9f1d870b1b99`, shared
  feature pipeline SHA-256
  `d824e3f00611e34fc9ffa17bed980ce72b9135f4194ab6d6986371129275c734`, and
  evaluator SHA-256
  `da7fa3498bf6b84eece76787314b7cf945ab6630cb7fbdd97eeee97268646811`.
  The pinned HotUGPU
  guard was read-only from sibling branch `codex/r2-holoassist-results-20261020`,
  commit `5c0bae773f9d4ec983ff3ac58fc13b1dd24bc47a`, script SHA-256
  `ae3ca95654a2f57e6652455f9cf3cf6a9639ec0f456d4f24318ce146130ecde8`.
  The guard waited for the existing Orbitune process to yield; it sent no
  signal and did not terminate the process. It reserved 3,200 MiB and reported
  13,015 MiB free when the child started.
- Environment: Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2,
  PyAV 18.1.0, RTX 3080 Laptop GPU. Feature extraction took 442.10 seconds;
  CUDA peak allocated memory was 340,768,256 bytes. The Windows allocator
  emitted `expandable_segments not supported on this platform`; extraction
  completed successfully. Scoring was CPU-only. Paid/cloud compute cost: $0.

## Metrics

| Fixed condition | Accuracy | NLL | Brier | ECE | Events |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint audio + visual | 0.3948 | 1.8466 | 0.7622 | 0.0742 | 504 |
| Video only, matched | 0.3532 | 2.0045 | 0.8111 | 0.1415 | 504 |
| Uniform over supplied candidates | 0.0833 | 2.6373 | 0.9224 | 0.0232 | 504 |

Paired bootstrap clustered by video ID, 10,000 draws, seed 17:

| Contrast (joint − video-only) | Point difference | 95% percentile interval |
| --- | ---: | ---: |
| Accuracy | +0.0417 | [−0.0078, +0.0903] |
| NLL | −0.1579 | [−0.2478, −0.0670] |

The joint checkpoint's Accuracy is 4.17 percentage points higher, but this
interval includes zero. Its NLL is lower with the paired interval excluding
zero. Brier and ECE are descriptively lower; no uncertainty interval was
registered for those two summaries. Candidate counts range from 2 to 48 and
were not truncated; metrics by exact candidate count and every event's
prediction/probabilities are retained in the external result directory.

An independent CPU integrity replay joined both prediction files back to all
504 unique feature-manifest event IDs and checked video ID, sound ID, target,
candidate count, candidate-order hash, probability-vector length, finiteness,
normalization and target bounds. It independently recomputed Accuracy, NLL,
Brier and the evaluator's fixed 10-bin ECE for both checkpoints; all values
matched the receipt within `1e-10`. Both prediction files cover exactly 200
unique videos, with no duplicate event IDs. This replay confirmed result
integrity; it adds no new evaluation sample or quality claim.

## Interpretation

On this previously unused TRAIN subset, the fixed joint readout transfers with
better probability ranking under NLL than the fixed video-only readout. The
Accuracy contrast is uncertain. The difference is not a blind estimate of
generalization, and source annotation exposure plus unknown participant groups
limit the claim. The uniform baseline is not an audio-only learned readout;
the task requires choosing among visual track candidates, so an audio-only
candidate scorer without visual evidence is not a meaningful same-task
comparator.

This result supports retaining the small joint readout as a research reference
for this task. It does not justify another gate fit, shared-Lens adoption,
model merging, product promotion, or an R2 M-2 admission claim. Stronger
evidence requires a rights-cleared, independently grouped task with fresh
development data, a truth-bearing both-stream-required design, and predeclared
missing/shuffle/replacement controls. Do not reuse this holdout for tuning.

## Reproduction

Feature extraction was launched only through the cooperative HotUGPU guard:

```powershell
$env:PYTHONPATH='src'
& 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe' `
  'C:/CodexWorktrees/r2-holoassist-results-20261020/scripts/run_with_hotgpu_guard.py' `
  --max-wait-seconds 7200 --reserve-mib 3200 `
  --block-if-free-mib 14000 --min-child-free-mib 8000 -- `
  'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe' `
  -m scripts.extract_perception_av_event_fresh_holdout `
  --plan C:/CodexArtifacts/pretrained-reuse-r2/perception-av-event-fresh-holdout-plan-01/plan.json `
  --plan-sha256 887a1eea4b5a7b82d441025842465a09d0ffb33c97787073042ad1a1680f8eb3 `
  --media C:/CodexArtifacts/pretrained-reuse-r2/perception-av-event-fresh-holdout-media-01 `
  --pts C:/CodexArtifacts/pretrained-reuse-r2/perception-av-event-fresh-holdout-pts-full-01/result.json `
  --output C:/CodexArtifacts/pretrained-reuse-r2/perception-av-event-fresh-holdout-features-01 `
  --device cuda --image-batch-size 16
```

Fixed-checkpoint CPU evaluation:

```powershell
$env:PYTHONPATH='src'
& 'C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe' `
  -m scripts.evaluate_perception_av_event_fresh_holdout `
  --features C:/CodexArtifacts/pretrained-reuse-r2/perception-av-event-fresh-holdout-features-01/features.jsonl `
  --features-sha256 5a4fed0a0419fe1091364b5b240b371e47c64ffff8516a2fe179eaac998fd1a4 `
  --output C:/CodexArtifacts/pretrained-reuse-r2/perception-av-event-fresh-holdout-eval-01
```

All media, feature arrays, predictions and receipts remain in
`C:\CodexArtifacts\pretrained-reuse-r2\`; none were added to Git. Full R0/R2
product gates and all conditional R3–R8 work remain open.
