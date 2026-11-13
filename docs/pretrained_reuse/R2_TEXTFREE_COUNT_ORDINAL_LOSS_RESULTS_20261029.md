# R2 train-only image-count loss comparison — 2026-10-29

## Decision

The cumulative ordinal loss candidate was **not selected**. With the fixed
validation-NLL selector, its best checkpoint occurred at epoch 5 and was much
worse than categorical cross-entropy. A count-only CE head finished almost
identical to the prior four-head readout on the same internal development
images. Do not evaluate the ordinal candidate on the already exposed official
validation split; retain these artifacts as a negative, task-specific result.

## Controlled experiment

Two count-only heads used the same frozen VideoPrism-B features, train-derived
asset split, seed-17 initialization, per-epoch shuffled ID order, 512-epoch
budget, 256-image batches, AdamW (`lr=1e-3`, `weight_decay=1e-4`), and
minimum categorical validation NLL checkpoint selector. Each arm trained only
the same 7,690-parameter linear count head. The sole configured difference was
the count training loss:

- categorical cross-entropy;
- cumulative ordinal NLL, summing binary log-losses over the nine ordered
  count thresholds.

Both started with identical count-head SHA-256
`2ec1ec073eb3d52a2f6d8df9ea62b16d14309e018b87c5ed6eee720dcbf26935` and used
the same 7,582 train / 842 internal-development image IDs from the original
official train partition. Asset overlap was zero; ID order hashes were checked
against the prior fit. Each arm completed 15,360 updates and 3,881,984 image
exposures. The previous four-head selected model was loaded only as a reference
on those same internal-development features; it was not an initialization for
either arm.

The ordinal objective sums nine threshold losses while CE uses one categorical
loss. Although this is the single deliberate algorithmic change, its loss scale
and interaction with AdamW/weight decay are part of that change; this result
does not isolate ordinal structure from every optimization-scale effect.

## Results on the train-derived internal development split

All rows have n=842. The split was used to select each arm's checkpoint, so
these results and intervals are development evidence, not independent testing.

| Model/checkpoint | Selected epoch | Accuracy | NLL | Brier | ECE (15 bins) | Within ±1 | Argmax MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing joint four-head CE reference | 502 (macro selector) | 58.19% | 0.9447 | 0.5062 | 0.0379 | 92.64% | 0.500 |
| New count-only CE | 512 | 58.43% | 0.9440 | 0.5063 | 0.0259 | 92.52% | 0.498 |
| New cumulative ordinal NLL | 5 | 34.80% | 1.8402 | 0.7798 | 0.0911 | 79.93% | 0.960 |

The count-only CE difference from the prior joint model is only +0.24 percentage
points in Accuracy and −0.0007 NLL; it does not establish a useful gain. Paired
bootstrap for the selected ordinal checkpoint minus selected CE checkpoint,
resampling the 842 image rows 10,000 times (seed 20261029), gives Accuracy
−23.63 pp (95% percentile interval −27.91 to −19.36), NLL +0.896
(+0.840 to +0.955), and argmax MAE +0.462 (+0.397 to +0.527). Checkpoint
selection used this same split, so the intervals do not remove selection bias.

The ordinal run's learning curve is an overfitting/calibration warning under
this fixed schedule: at epoch 5 the train/dev NLL was 1.858/1.840; at epoch 512
it was 2.482/2.513. Train Accuracy rose from 35.1% to 64.8%, while dev Accuracy
rose from 37.0% to 55.7%; the fixed NLL selector correctly retained epoch 5.
For context, CE validation NLL improved from 2.112 at epoch 1 to 0.944 at
epoch 512, with Accuracy 21.38%→58.43%. The later ordinal Accuracy is reported
only as a learning-curve observation; it was not the configured selection
metric and does not make the final checkpoint eligible.

## Integrity, resources, and artifacts

- Source commit: `6b6cc3def65a5dd5bf2984ecc02311e43c3d9605`.
- Effective config SHA-256: `937464b8bc40ed90951dfdae206d0342f3127705b83e1596eb2f08c84d41de92`.
- Frozen feature cache SHA-256: `f36b9cf17efed6a5137eb48ffcaf40ff960015fc07bc194fb532bf5562d88fab`.
- Train-label SHA-256: `00f03f27982e2785636e85a50f4ecf258c0ac3ee483d7242e297cee14b50ec2e`.
- Train-media-manifest SHA-256: `2c56815beb9d2e9951cd38686eafdbe2ecdee05e69e74b5acfad24fbefbb9010`.
- Report SHA-256: `1e511f497487eb6a7a310af25aa17521fc6bc1c10fce8806917077ab4c9518c6`.
- External output: `C:\CodexArtifacts\pretrained-reuse-r2\textfree-image-count-losses-seed17-512-01`.
- Selected checkpoint SHA-256: CE `DE66746039691991E22936509F09E03320EA02D50E9D43723028EE6A22DEA385`; ordinal `BE045D3AB5E4FF6B4DD2C608EDEBCD8B56AB960B8CAC3E2043EEFBBA77063DEB`.
- Each selected head is 32,200 bytes / 7,690 trainable parameters. Frozen VideoPrism-B plus one count head is 114,372,874 parameters; this is only a task readout path, not a complete Omni package.
- Both prediction files contain 842 unique image IDs. The selected weights were reloaded and produced exactly matching internal-development logits.
- Combined fit time was 108.6 seconds on CPU (CE 20.5 s; ordinal 88.1 s); no CUDA was used. The existing Orbitune GPU process remained active and untouched.
- Official validation arrays and sealed audit were not loaded. No audio/video, natural-image or joint-modality claim is made.

## Next action

Keep the existing CE four-taxonomy readout as the image reference. Stop this
ordinal-loss variant here; a changed learning-rate or loss-weight search would
be a new experiment, not a continuation of this controlled comparison. The
larger R2 gates—natural-image reasoning, natural synchronized multimodal
necessity, and product-level performance—remain open.
