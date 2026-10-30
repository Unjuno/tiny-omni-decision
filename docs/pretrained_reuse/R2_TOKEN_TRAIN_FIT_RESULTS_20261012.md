# Training-only token readout fit diagnostic

Completed both predefined512-update conditions. **Training fit only, not quality
or generalization success.** No validation inference, checkpoint selection or
sealed audit access. Source `7bc4f49`; protocol fixed before execution.

The first64 training choices contain5 videos,32 explanatory/6 predictive/26
counterfactual choices and31 correct/33 wrong targets. Selection uses existing
order, not labels or performance. Each condition consumes32,768 encounters,
of which32,704 are repeats. Encoders are frozen; token means match the prior cache.

| Condition | Step | Train macro Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Attention | 0 | 0.500000 | 0.693428 | 0.500281 | 0.003254 |
| Attention | 32 | 0.881410 | 0.260499 | 0.176855 | 0.114000 |
| Attention | 64 | 1.000000 | 0.021773 | 0.008257 | 0.019122 |
| Attention | 512 | 1.000000 | 0.00002960 | 0.00000001206 | 0.00002962 |
| No video | 0 | 0.447917 | 0.693150 | 0.500003 | 0.052166 |
| No video | 128 | 0.829327 | 0.456176 | 0.283589 | 0.194548 |
| No video | 256 | 0.883814 | 0.300911 | 0.175013 | 0.137372 |
| No video | 512 | 0.976763 | 0.137682 | 0.066920 | 0.099927 |

Full per-type metrics, histories, IDs, probabilities and artifact hashes are in
the companion JSON and original output directory. Both final reloads are exact.
No intermediate checkpoint was selected or evaluated on validation.

## What this changes

The attention model fits all64 examples by64 updates and maintains that fit.
This weakens a completely broken training/gradient path explanation for the
larger experiment's poor results. It does not establish sufficient capacity or
optimization for the full3794-example corpus.

No-video reaches97.68% macro Accuracy, demonstrating that high accuracy on this
repeated subset can arise without visual reasoning. Attention's faster fit can
use scene identity and memorization; it is not evidence of transferable physics.
Neither percentage counts toward the product's95–98% per-modality requirement.

Final normalized attention entropy averages0.364900, range0.088152–0.577696.
Query/key/value/position weight gradients are finite; position gradient norm
is4.54e-6. Key bias gradient is near zero as expected: the shared additive key
bias shifts every token score equally and cancels through softmax. No-video
value.weight has zero gradient because its input is zero; key/position gradients
are absent by design. These controls agree with the implemented computation.

The full-corpus run's train57.57%/validation52.44% at its selected checkpoint is
therefore not explained by inability to fit any examples. Generalization,
optimization under diverse batches, and representation/task alignment remain
unresolved. Do not automatically enlarge the model or extend the same validation
search. Next compare the full-corpus objective/gradient behavior with this stable
full-batch fit before selecting another training schedule.

## Resources and retained artifacts

- Whole diagnostic37.507s; attention15.779s, no-video5.221s; local cost$0.
- Peak allocated GPU memory152,674,816bytes. RTX3080Laptop; CPU frozen video extraction.
- Artifact directory: `C:/CodexArtifacts/pretrained-reuse-r2/video-token-train-fit-01`.
- Attention final SHA256: `19cf3eb9dc93562d77f82a1cd5d8742e9635cf1bc79d98c6b77e5c0282291342`.
- No-video final SHA256: `709a533e1e5af1f835aaba7f13d48a93fad8060a7be4ba5341f5c229b7f83973`.
- Original corpora, weights and prior run artifacts retained unchanged.
- No product promotion, compression or new full validation search performed.
