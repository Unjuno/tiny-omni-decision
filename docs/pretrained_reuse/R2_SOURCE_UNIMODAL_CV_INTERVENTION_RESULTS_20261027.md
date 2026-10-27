# Source-candidate CV audio intervention results

The [fixed protocol](R2_SOURCE_UNIMODAL_CV_INTERVENTION_PROTOCOL_20261027.md)
reloaded the three joint out-of-fold checkpoints from the existing 320-video,
2,593-record train-only cross-validation. The original-input scores and labels
matched the saved predictions before interventions. No checkpoint was changed,
and no new training, old development labels, official validation, or sealed
audit was used.

| Audio input to the same joint readout | Accuracy | Set NLL | Visible-source accuracy | None-visible accuracy |
|---|---:|---:|---:|---:|
| Original | 0.5515 | 1.4441 | 0.5066 | 0.6749 |
| Zero | 0.4404 | 1.4984 | 0.3204 | 0.7702 |
| Different held-out video | 0.4894 | 1.7366 | 0.4603 | 0.5694 |

The fixed different-video pairing rotates lexically sorted videos within each
fold and uses the donor video's first record. It is deterministic and always
uses a different video ID, but it is not matched for sound class. Thus the
change can reflect sound category, distribution shift, other recording
properties or correspondence. Zeroing the audio is also out of distribution.
Neither intervention is a separately trained audio-only/video-only baseline.

Original minus mismatched-audio Accuracy is +6.21 percentage points, with a
95% paired video-group bootstrap interval of +4.28 to +8.23 points; its NLL
difference is −0.2926 (−0.3689 to −0.2195). Original minus zero-audio Accuracy
is +11.11 points (+8.77 to +13.49), while NLL difference is −0.0544 with an
interval crossing zero (−0.1251 to +0.0186). Bootstrap used 10,000 draws,
seed 17, over 320 video groups; overlapping training folds, one seed, oracle
tracks, possible shared participants and the nonblind source limit these
intervals. The observed sensitivity does not establish that an independently
trained multimodal model beats a video-only model: the prior matched comparison
found joint minus video Accuracy +0.69 points with an interval crossing zero.
The 95–98% product goal remains unmet.

The first attempt completed successfully; its result SHA-256 is
`afc9e3525b873bcdbba42c22d323c002ba9644ccdffb3e04da021ea51906e186`.
A later replay in a new output directory followed two
lint-only changes and produced the **same per-record prediction hash**
`1fded3500ce4169e120f86893c518e89c56e2f777851154e6fefe94d70b13d9f`
and same donor-map hash. The replay result is
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-unimodal-cv-interventions-20261027-attempt-02/result.json`,
SHA-256 `2129749d6bb07031e20d45f5aa40378cf5ec4fa6deba9e6df3c4e9784e056de2`.
The executed replay script SHA-256 is
`0bfe4456c6b640f0a768995506a2a758562d34ded35bee6d6fa47421d77fefad`.
Two donor-mapping tests and Ruff passed. Both attempts were CPU-only and CUDA
was not initialized; neither source predictions nor checkpoints were changed.
