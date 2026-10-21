# CLEVRER synthetic audio-video timing result (2026-10-20)

## Result and scope

One matched seed-17 experiment completed all four predeclared 2,048-update readouts. On this *synthetic temporal-binding development task*, the joint LSTM outperformed the audio-only LSTM on the same 390 examples (79-scene clustered validation): Accuracy 78.97% vs 72.31%, NLL 0.4566 vs 0.5051, and Brier 0.3031 vs 0.3515. This is evidence that the video stream added useful collision-time context to the generated audio cue in this task. It is not evidence of natural audio-video understanding, broad Omni Decision quality, or the 95–98% target.

The text-only paired control is exactly chance by construction (Accuracy 50%, NLL 0.6931, Brier 0.5000, ECE 0). The task uses silent CLEVRER videos and generated 880 Hz, 80 ms audio bursts placed 16 source frames before or after a uniquely described collision. The two variants in each pair share the exact same video and question and have opposite targets. A decision needs both the visual collision time and audio-burst time. Scene-level time shifts reduce absolute-timestamp shortcuts. The video encoder is frozen VideoPrism-B with 16 framewise pooled features; the audio input is normalized waveform energy in 16 bins, not a pretrained audio representation. Text uses frozen MiniLM features.

## Matched results

| Candidate | Best step | Accuracy | NLL | Brier | ECE-15 | Both pair variants correct |
|---|---:|---:|---:|---:|---:|---:|
| Joint gated LSTM | 512 | 78.97% | 0.4566 | 0.3031 | 0.0670 | 58.46% |
| Audio-only LSTM | 256 | 72.31% | 0.5051 | 0.3515 | 0.0868 | 44.62% |
| Video-only LSTM | 768 | 50.00% | 0.6931 | 0.5000 | 0.0001 | 0.00% |
| Joint mean-pooled control | 1,408 | 50.00% | 0.6931 | 0.5000 | 0.0001 | 0.00% |

The validation-only paired bootstrap resampled 79 scenes 5,000 times (seed 17). Joint minus audio-only Accuracy difference was +6.91 pp, 95% percentile interval [+3.43, +10.65] pp. Audio-only minus joint NLL improvement was 0.0497, interval [+0.0006, +0.0973]; Brier improvement was 0.0490, interval [+0.0141, +0.0841]. These intervals reuse the development validation that selected checkpoints, so they do not remove selection bias. Each pair is dependent by construction and is kept within a single scene split.

The joint LSTM prediction changed on 58.97% of rows when its paired audio was swapped; after the swap, 78.97% matched the donor target and 21.03% retained the original target. This is intervention evidence of audio sensitivity, not proof that the model uses the intended audio event consistently. The learned mean gate values at the selected step were 0.598 audio / 0.402 video.

The final joint checkpoint reached 98.06% training Accuracy but 78.46% validation Accuracy; final validation NLL/Brier/ECE worsened to 0.9432/0.3704/0.1761. The best selector chose step 512. The large train/validation difference and late calibration degradation are overfitting warnings. Do not extend the budget on this same small sample set.

## Data, implementation and compute

### Source-pool limit

The prior 24-root exposure scan found 9,589 of the 10,000 official TRAIN scene IDs already present in metadata and 6,045 with cached media; only 405 scenes met that scan's previously-unseen candidate rule. The AV pilot fetched and checked all 405 candidates, and 401 contained eligible rows after event-ambiguity filters. No additional scenes remain under that same metadata/filename scan rule. This is not proof that all 9,589 metadata-listed scenes were used for model supervision, and the earlier scan did not cryptographically check all cached media content. A larger follow-up must either reuse official-train media already present in the scan roots with a clearly task-specific split, or screen another source. Any new validation from the former must be described as task-specific, not media-unseen.


- Frozen source: `MIT-IBM/CLEVRER` train, revision `98b842082ba4f7c18b6b9e3f39145871782a65ef`, CC0-1.0. The task and seed-17 scene split were preregistered before feature/model evaluation. The official CLEVRER test split and sealed audit were not accessed.
- Train: 798 event pairs / 1,596 rows / 322 scenes. Development validation: 195 pairs / 390 rows / 79 scenes. Scene overlap and media-hash overlap are both zero in the saved run receipt. Source metadata/filename screening did not cryptographically certify all earlier cached content, which remains a limitation.
- The saved split manifests reproduce their recorded SHA-256 values: TRAIN `f2ac5c836ac387661367e9caee150dec58684aa93f7df4e150aa574efd294e03`; development validation `b3436be649dc02ec825f488a710dac78d99f9dee36783b71d2ace74bb13a4b1a`. Ordered sample-ID hashes are TRAIN `df0b7525568d4ad8b9d22f536c241314cc42f218db65b4e0c0cf339c8c2532d7` and validation `572bf3aeb96b6b7ca102cc7ae2fa5efcc22a792f2aff97b50a840f4ebb0dbd40`. The four candidate metrics were independently recomputed from saved predictions; Accuracy/NLL/Brier/ECE match the run JSON to <1e-7.
- Every candidate used seed 17, 2,048 optimizer updates, batch size 64, learning rate 0.001 and weight decay 0.0001. No early stopping. Each candidate saw 130,748 row presentations but only 1,596 unique rows (about 81.9 presentations per unique row), another reason to prioritize unique data before changing capacity.
- Selected joint readout has 48,868 trainable parameters. The inference components represented here total 131,803,876 parameters: VideoPrism-B 114,365,184 + MiniLM-L3 17,389,824 + readout 48,868. This is below the provisional 200M parameter ceiling, but it is not a complete natural-audio Omni system. The two frozen checkpoint files alone are about 527.0 MB; the provisional 200 MB package target is not met.
- The fit ran on the local RTX 3080 Laptop GPU. Four-arm training took about 64.6 seconds in aggregate; the run occupied about 70.2 seconds wall time. Peak allocated GPU memory was 315,372,032 bytes. HotUGPU metadata confirms a cooperative 3,200 MiB reservation and no cloud cost. Separate framewise VideoPrism feature extraction used JAX CPU and took 1,073.43 seconds with about 4.19 GB peak process working set; it is not included in the fit wall time.
- Source code commit: `82f260b0525f1f95c943dd57009821ec46b7eda1`. Effective config SHA-256: `6edbd88442f29de8917377b734bda933c8ac0a7e51af88700825be6b7cd60f4b`. Frozen protocol SHA-256: `82b76e813455061769883b6a46517dbfb3a7a54f7593c1093ab92d8cc261fab7`.
- External artifact directory: `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-synthetic-av-temporal-decision-v1-attempt-01`. `results.json` SHA-256: `f175a2f7ec9fa80c4780c9f3df9af867dcd18d05271c4bb5efa6d0aca1b49b18`; `run-metadata.json`: `a88cf24bd85a8ea6a8b26f5d1e790e3e4102e5a5db72971e89e762f220e4c319`; selected joint predictions: `3510491a5682ee60ff40165a25afad0c2ea9db7e95d14c0a9ee9c2dab4f0604f`.

## Decision and next gate

Keep this as a successful **controlled development pilot**, not as an admitted product benchmark and not as natural AV evidence. It shows a real, measurable joint-over-audio-only gain in one synthetic setup, but only 58.46% of paired events have both variants correct and the model is overfit. Before another fit, note that no fresh CLEVRER scenes remain under the previous exposure rule. A larger AV fit must use previously-seen official-train media with a clearly task-specific split, or screen another source; retain the same frozen task semantics and all audio-only, video-only, mean-pooled, text-only and audio-swap controls. First prove unique sample coverage and feature-extraction feasibility. Do not use the current validation as a blind result or alter the 95–98% product objective.

The run's model-selection validation and its bootstrap are the same development set. No sealed evaluation was run. No weights, media or large caches are committed to Git.
