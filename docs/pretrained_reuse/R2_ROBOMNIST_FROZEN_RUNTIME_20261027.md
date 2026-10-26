# RoboMNIST: frozen CPU input runtime and task dependence

Execution of the input path passed on **one fixed training question**. This
does not admit a training corpus, measure Decision quality, establish both-input
necessity, or close R2 M-2. The full four-modality 95–98% objective is unmet.

## Actual execution

The first question of the hash-pinned attempt-05 training manifest selects
`Ar1_Vhigh_Ac0_Rp01_UNC062` video and two Rx1 audio candidates from repetitions
01 and 02 of the same archive/class. Selection preceded inference; no validation
or sealed-audit media was opened. Only runtime source references are passed to
the loader, with no target, class name, group identifier or question metadata
provided to either encoder.

Three HTTPS ZIP members were fetched into RAM with exact Content-Range/length,
bounded expansion and CRC32 checks. No raw media was written to disk. Their
content hashes are in the execution receipt. The fetched video decoded to 450
frames with PTS 0–14.9667 s. Requested centers were eight equally spaced points
from 1 to 13.8 s; actual selected PTS were 1, 2.8333, 4.6667, 6.5, 8.3,
10.1333, 11.9667 and 13.8 s. Each horizontally concatenated stereo frame is
split into two whole lenses, letterboxed to 224 square, and normalized with
the pinned processor. Resizing/cropping by the processor is disabled.

The audio streams contained 658,432 and 661,504 mono PCM16 samples at 44.1 kHz
(14.93043 and 15.00009 s). Complete two-second windows at the same requested
centers were resampled to 32 kHz without padding, gain normalization or wrapping.
Shared centers are an input convention, **not evidence of a common capture
clock**. The [previous collection-code audit](R2_ROBOMNIST_CLOCK_SOURCE_20261015.md)
still leaves WAV start/stop handling and camera capture offsets unknown.

| Measurement | Actual result |
|---|---:|
| Video features | `[8, 2, 512]`, finite and nonzero |
| Audio features | `[2, 8, 384]`, finite and nonzero |
| Mean neighboring-time cosine distance, video | 0.00378847 |
| Mean neighboring-time cosine distance, audio | 0.09406332 |
| Resident parameters, including unused TinyCLIP text tower | 85,062,424 |
| Trainable parameters | 0 |
| Complete encoder state before/after | Both hashes exactly equal |
| Network bytes, successful attempt | 9,290,186 |
| Network / video decode / inference plus audio preprocessing | 20.839 / 12.806 / 1.085 s |
| Total successful-attempt wall time | 42.067 s |
| Saved feature file | 53,477 bytes |
| Local GPU / paid compute | 0 / 0 |

These distances are input-variation diagnostics, not measures of action
understanding or information sufficient to choose the corresponding audio.
No cross-modal similarity score is computed between incompatible embedding
spaces. No probabilities, Accuracy/NLL/Brier/ECE, training or checkpoint
selection are claimed.

CPU, four threads, float32, SDPA, Python 3.11.9; PyTorch/Torchaudio 2.6.0+cu124,
Transformers 5.6.2, PyAV 18.1.0, NumPy 2.4.6. CUDA was not initialized and
the active Orbitune GPU training process was left running. RAM peak was not
instrumented for this diagnostic. Versions were not upgraded.

## Pins and retained failures

- TinyCLIP: `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M`, revision
  `07a4b0bc751cb64fecd2b661c048c1dd98d69444`, weights SHA-256
  `112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587`.
  MIT checkpoint; 83,101,185 resident parameters.
- EfficientAT DyMN-0.4: pristine source commit
  `a425fdce92572e602a1d5634799bd9f1f2efa806`, checkpoint SHA-256
  `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`.
  MIT source with explicit upstream checkpoint-use permission recorded in R1;
  1,961,239 parameters. CAV-MAE remains REVIEW and was not loaded.
- Full two-checkpoint size is 340,530,121 bytes; this runtime is under 200M
  parameters but exceeds the provisional 200 MB package objective.
- Successful run began `2026-10-26T15:12:12.778967+00:00` and ended
  `2026-10-26T15:12:54.845494+00:00` (October 27 JST). The receipt records
  its exact start timestamp, script/dependency hashes and base Git commit
  `753438df`; runtime code was uncommitted, so the script hashes bind execution.
- Attempt-01 failed after data fetch because Transformers 5.6.2
  `get_image_features()` returns a `BaseModelOutputWithPooling`, not a tensor.
  The corrected path uses `visual_projection(vision_model(...).pooler_output)`,
  as the existing native-container frontend does. The successful attempt also
  reuses its stereo-letterbox helper. The failed output is preserved.
- Total wall time across these two runtime attempts was 81.861 s; total network
  transfer was 18,580,372 bytes. No partial output was represented as success.

Outside-Git successful receipt:
`C:/CodexArtifacts/pretrained-reuse-r2/robomnist-frozen-features-cpu-smoke-20261027-attempt-02/run.json`,
SHA-256 `a617055830f5d17c8f597b8a727a8b0fbb368531eaa7e7c073fb93a1e56e89cd`.
Feature SHA-256:
`1bf4364ef34b87e5df4e2335a0433b397f27368de5429c662fdd1139815ebabb`.
The small [execution receipt](R2_ROBOMNIST_FROZEN_RUNTIME_20261027.json)
and [dependency receipt](R2_ROBOMNIST_TASK_DEPENDENCIES_20261027.json) are
retained in Git; media, model weights and feature arrays remain outside Git.

## Sampling dependence and remaining admission

The task contains reversed-order companion questions and cyclic reuse of
candidate audio from the same robot/velocity/action block. A union-find audit
joins questions that share any underlying repetition, including transitive
connections. It reads **only train and validation**; their pinned hashes match
the [manifest report](R2_ROBOMNIST_REPETITION_AUDIT_20261026.md).

| Split | Questions | Distinct repetitions | Shared-asset connected components |
|---|---:|---:|---:|
| Train | 3,100 | 1,550 | 60 (24–27 repetitions each) |
| Development validation | 480 | 240 | 60 (4 repetitions each) |

The 480 questions and 240 repetitions are not independent evaluation units.
The 60 components are a lower bound on dependency clusters, **not 60 verified
independent collection sessions**; common sessions/lab conditions could create
further dependence. A later uncertainty analysis must preserve shared-asset
components and any additional verified source groups. This does not establish
the roadmap's provisional 200-independent-group screen where applicable.

Exact option/asset-role balancing is a construction safeguard. It does not prove
absence of audio-only pair shortcuts, visual collection cues, common counters,
or perceptually ambiguous correct labels. Before fitting, establish the actual
session scope and episode-level task validity, then preregister matched
audio-only/video-only, missing-input, candidate-order and pair-shuffle controls.
Use timing/stale/offset controls only with an explicit supported time contract.
No extra model capacity or full-corpus fitting is justified by this smoke alone.

## Reproduction and checks

```powershell
python -m scripts.smoke_robomnist_frozen_features --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-frozen-features-cpu-smoke-NEW
python -m scripts.audit_robomnist_task_dependencies --output C:/CodexArtifacts/pretrained-reuse-r2/robomnist-task-dependencies-NEW.json
```

Both commands reject existing outputs and pin development inputs. The feature
runtime pins models/source/metadata and never accepts an audit path. Unit tests
cover actual-PTS selection, insufficient media coverage, no-padding audio time
windows, target-free runtime selection and transitive shared-asset dependencies.
The existing stereo-frontend tests check both complete lenses and mean fill.
Local verification passed: Ruff, 2,589 CPU tests (11 skipped), 20 dataset
manifests, four dataset catalogs and the example model/rights audit.
