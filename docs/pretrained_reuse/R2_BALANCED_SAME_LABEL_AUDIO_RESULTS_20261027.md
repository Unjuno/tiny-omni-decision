# Balanced set readout: same-label audio substitution

The [fixed diagnostic protocol](R2_BALANCED_SAME_LABEL_AUDIO_PROTOCOL_20261027.md)
reused the three balanced-loss joint checkpoints and the prior hash-verified
same-label donor mapping. Original audio was replaced by audio from a different
video with the same annotated label in the same held-out fold. Before each
comparison the original candidate probabilities, none probability and selected
set were replayed within 1e-6 of the saved predictions. Labels were used only
to choose donors. No training, checkpoint selection or sealed audit occurred.

Of 2,593 records, 2,581 matched and 12 were unmatched. Results below use the
same 2,581 records in both conditions; candidate NLL and Brier are per-record
means of marginal binary scores, not a joint distribution over sets.

| Measure | Original audio | Same-label audio from another video |
|---|---:|---:|
| Exact-set accuracy | 18.33% | 18.36% |
| Candidate NLL | 0.3678 | 0.3643 |
| Candidate Brier | 0.1217 | 0.1203 |
| None-event NLL | 0.4327 | 0.4158 |
| None-event Brier | 0.1287 | 0.1263 |

Original minus replacement exact accuracy was -0.04 percentage points, paired
320-video bootstrap interval -1.04 to +1.00 points. Multi-target exact accuracy
was 7.35% versus 6.44%, difference +0.91 points, interval -0.34 to +2.20.
Bootstrap uses 10,000 video-group draws, seed 17. The selected set changed in
972/2,581 cases even though aggregate exact accuracy did not clearly change.
This combination shows sensitivity to the audio features without establishing
that original audio/video correspondence improves task accuracy. It does not
prove the models ignore audio or that the inputs are equivalent.

This remains nonblind train-source OOF evidence with oracle visual tracks;
same-label recordings can differ in event, speaker, physical source or scene,
and different video IDs do not guarantee different people. Training folds
share data, and the interval does not account for training-seed uncertainty.
The candidate is below the four-modality objective and was not promoted.

CPU only; CUDA was not initialized. Result:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-balanced-same-label-intervention-20261027-attempt-01/result.json`,
SHA-256 `2ff662bd09445b791638f9ec327727cc30ec0cd66a02794a4e59eae71100d073`.
The result records exact donor-map and per-example prediction hashes.
