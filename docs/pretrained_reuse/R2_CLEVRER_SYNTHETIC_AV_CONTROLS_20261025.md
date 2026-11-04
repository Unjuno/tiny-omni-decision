# CLEVRER synthetic AV fixed-checkpoint input controls (2026-10-25)

## Scope and interpretation

This is a post-selection sensitivity diagnostic using the already observed
1,110-row / 240-scene development validation from the synthetic CLEVRER AV v2
run. It uses the existing selected `joint_lstm` checkpoint and frozen feature
cache. No fitting, calibration, checkpoint selection, or sealed-audit access
occurred. These exposed development rows are not blind evidence, and the
synthetic tones plus silent CLEVRER video do not establish natural AV
understanding or general multimodal decision quality.

The question was whether the selected model's saved decisions respond to
missing, mismatched, reversed, delayed, and contradictory inputs. Each
perturbation was evaluated against a CPU replay of the same checkpoint and
features. The five perturbations are within-model interventions, not new model
candidates.

## Results

| Input condition | Accuracy | NLL | Brier | ECE-15 | Both pair variants correct | Prediction flip rate |
|---|---:|---:|---:|---:|---:|---:|
| Baseline replay (CPU) | 83.51% | 0.3534 | 0.2243 | 0.0369 | 68.47% | 69.91% |
| Video missing / masked | 65.77% | 0.6566 | 0.4445 | 0.1788 | 31.53% | 31.53% |
| Video from next validation scene | 69.55% | 0.7266 | 0.4402 | 0.1646 | 45.77% | 52.43% |
| Video time-reversed | 71.62% | 0.8302 | 0.4538 | 0.1979 | 43.42% | 43.60% |
| Audio shifted +4 of 16 bins | 45.41% | 1.9650 | 0.9069 | 0.4210 | 16.94% | 43.06% |
| Paired audio contradiction | 16.49% | 3.3789 | 1.4109 | 0.7027 | 1.44% | 69.91% |

The two paired rows per event share a scene and have opposing targets. The
`both pair variants correct` statistic is the fraction of the 555 pairs for
which both rows were correct. In the contradiction condition, audio was
swapped between paired opposite-target examples, so low Accuracy is the
expected consequence of preserving the original targets while changing the
input; it is not a standalone quality score.

For each perturbation, a paired scene-cluster percentile bootstrap compared
per-scene mean row outcomes against the CPU baseline (240 scene groups, 10,000
resamples, seed 17). Accuracy differences (perturbed minus baseline) and 95%
intervals were:

| Condition | Accuracy difference | 95% interval | NLL difference | 95% interval |
|---|---:|---:|---:|---:|
| Video missing / masked | -19.59 pp | [-22.49, -16.59] pp | +0.3362 | [+0.2839, +0.3889] |
| Video from next validation scene | -15.38 pp | [-18.38, -12.38] pp | +0.4129 | [+0.3429, +0.4871] |
| Video time-reversed | -11.47 pp | [-14.04, -8.96] pp | +0.4494 | [+0.3828, +0.5179] |
| Audio shifted +4 bins | -39.94 pp | [-44.17, -35.70] pp | +1.5760 | [+1.4269, +1.7331] |
| Paired audio contradiction | -69.22 pp | [-73.06, -65.26] pp | +3.1047 | [+2.9775, +3.2331] |

All five Brier difference intervals were above zero as well. Bootstrap
intervals are descriptive for this already observed validation and do not
account for checkpoint-selection effects. The contradiction is deliberately
label-inconsistent after intervention, so its interval should be read only as
a response check.

CPU replay matched the saved CUDA baseline's top-1 predictions on all 1,110
rows and reproduced Accuracy exactly (83.5135%). The maximum absolute
probability difference was 0.00062123; the diagnostics therefore report
cross-backend drift separately and compare interventions only to the within-CPU
baseline.

## Conclusion and limits

On this synthetic task, the fixed checkpoint is sensitive to video presence,
scene correspondence, and temporal order; it is even more sensitive to audio
timing and the paired contradictory-audio intervention. The results weaken a
simple claim that the model wholly ignores video, but they do not prove that
both modalities are semantically used correctly on natural AV tasks. Video
perturbations reduce Accuracy without bringing it to chance, so the baseline
does not establish video necessity. The strong audio-offset response is
consistent with the generated timing cue defining the task.

R2 M-2 remains open. A qualifying follow-up still needs a rights-cleared,
independently grouped task whose labels establish that multiple inputs are
needed, plus matched unimodal baselines and predeclared missing, shuffled,
stale, offset, and contradictory controls. Do not tune on these exposed IDs.

## Reproducibility and artifacts

- Source implementation commit: `33256eaf5521b3965a9b2174870a6bfdaf226547`.
- Selected checkpoint SHA-256: `edffaed3711c745db37205afc74ba19c42dcdadbcbf3b8c8dfc318351442195c`.
- Config SHA-256: `ceb8fc5c8f7fd1eaffad44b5eb31a5a04063a9d65b30f3987f526a40900f6406`.
- Frozen feature cache SHA-256: `0b146fa637049d84ad88f0a71638cf30f6e967bd66add13ead0ab062bcbebdff`.
- Feature report SHA-256: `d73eb35fa67c031102c20fa9f90718a220b1a9374bcf087468f67a30e0420645`.
- Validation manifest SHA-256: `7504eb1add69594d8d25df6b32b280e840b31018a5bd5f6e01baf0d01b3a4081`.
- Ordered validation ID SHA-256: `e21ec5bee231100f5949c38685890e452a140f6aade9f1bfd23efabaafd2c997`.
- Diagnostic / postprocessor SHA-256: `540290c7e2f0bbd0c703d807417c160dceff3db13f4c0871c892f0a3573ab53d` / `aaac08df8c095a471019e874a276eede46c16be0e9d4ce0473b0190795376ee1`.
- Artifact directory: `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-coverage-v2-controls-attempt-02`.
- `control-results.json` SHA-256: `d5702027101aca7a73dc00a6e2811eb226ed710336c2792bdc41e11b46239d48`.
- The six prediction JSONL files each contain 1,110 rows; their SHA-256 values and the results manifest are in `artifact-manifest.json` in the artifact directory.
- Attempt 01 was retained after its provisional CPU/CUDA probability-parity gate failed (top-1 parity held, but maximum probability drift was 0.00062123). Attempt 02 completed all inference and saved all predictions; its initial summary step failed on a feature-report key mismatch. The saved outputs were validated and summarized by a separate postprocessor without rerunning inference. Both attempt directories remain intact.
- Evaluation and postprocessing ran on CPU. No new GPU fit or GPU inference was started for this diagnostic; no cloud compute or cost was used.

