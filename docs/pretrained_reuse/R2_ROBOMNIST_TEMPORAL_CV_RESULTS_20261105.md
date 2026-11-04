# RoboMNIST temporal readout results — 2026-11-05

Five-fold TRAIN-only development compares a compact ordered recurrent readout
with the same network trained on deterministic per-source, per-modality token
permutations. The pooled bilinear readout is included on the same OOF rows and
folds. Original validation and sealed audit were not opened.

## Results

|Updates|Ordered Acc/NLL/ECE|Shuffled Acc/NLL/ECE|Pooled Acc/NLL/ECE|
|---:|---:|---:|---:|
|1|0.501/2.997/0.482|0.499/1.358/0.330|0.499/1.665/0.393|
|25|0.496/0.711/0.034|0.483/1.366/0.348|0.504/0.727/0.101|
|50|0.500/0.706/0.020|0.493/1.370/0.337|0.502/0.716/0.083|
|100|0.505/0.714/0.019|0.505/1.363/0.327|0.517/0.812/0.163|
|200|0.503/0.741/0.038|0.500/1.348/0.331|0.529/1.207/0.299|
|400|0.489/0.794/0.063|0.503/1.367/0.330|0.532/1.245/0.301|

At400 updates, ordered LSTM reaches48.91% Accuracy / NLL0.794 / ECE0.063;
shuffled is50.26% /1.367 /0.330; pooled is53.21% /1.245 /0.301. At step50
ordered NLL is0.706 (uniform binary NLL0.693), versus1.370 shuffled and0.716
pooled. Pair-unit bootstrap over390 intact units (2,000 replicates, seed17)
gives ordered-minus-shuffled Accuracy intervals [-1.54,+3.08]pp at step50 and
[-3.98,+1.22]pp at400. Shuffled-minus-ordered NLL intervals are[0.591,0.737]
and[0.489,0.652]. Folds share fit data and have historical development
exposure; intervals describe saved OOF pair variation only, not training
variability or a blind estimate.

This task shows token order affects confidence/NLL in a narrow repeated-scene AV
matching problem, but shows no accuracy gain, semantic action understanding,
scene/device generalization or broad Omni quality. The ordered model briefly
improves NLL over pooling at step50 but remains near chance; continued updates
worsen NLL and Accuracy. The shuffled control disrupts VideoPrism's contextual
position information and independently sampled audio windows; it is not a raw
media reversal or proof of synchronized cross-modal timing. The recurrent model
is not adopted as a quality improvement. Temporal architecture and task ceiling
remain open hypotheses.

## Protocol and artifact

- [Protocol](R2_ROBOMNIST_TEMPORAL_CV_PROTOCOL_20261105.md); runner source commit
  `3e0d16277c905df5a1050590b2f436e58d1ca20c`.
- Manifest SHA-256 `3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f`; config
  SHA-256 `f1d74cee8e23dbb03828c8d01c77dca5d9a4ad5aaaefdd2f508facd9e47d0eb8`; prediction SHA-256 `2facf20151c0344edbf8e461cb804a971c4904a5f1edc2a955ddcf96fa4a2884`.
- Each fold:624 fit groups/312 pairs/1,248 rows;156 held groups/78 pairs/312
  rows. Normalizers fit on fit groups only.
- Two separate one-layer LSTMs, hidden16;76,032 trainable parameters (3.13%
  above the pooled73,728). Encoders plus readout total116,402,455 parameters.
- 60 saved states reload with exact prediction parity. Runtime
  500.76s; observed peak working set
  1,372,413,952 bytes. CPU only; no GPU/cloud cost.
- Command: `python scripts/fit_robomnist_temporal_cv.py --manifest C:/CodexArtifacts/pretrained-reuse-r2/robomnist-av-match-split-20261103-attempt-03/manifest.json --cache C:/CodexArtifacts/pretrained-reuse-r2/robomnist-av-match-feature-cache-20261103-attempt-01 --output C:\CodexArtifacts\pretrained-reuse-r2\robomnist-temporal-cv-20261105-attempt-01`.
- Artifacts: `C:\CodexArtifacts\pretrained-reuse-r2\robomnist-temporal-cv-20261105-attempt-01` and monitor `C:\CodexArtifacts\pretrained-reuse-r2\robomnist-temporal-cv-20261105-attempt-01.monitor.json`.
- OOF metrics: [JSON](R2_ROBOMNIST_TEMPORAL_CV_CURVES_20261105.json),
  [CSV](R2_ROBOMNIST_TEMPORAL_CV_RESULTS_20261105.csv).

This diagnostic does not close the full R0/R2 or multimodal product objective.
