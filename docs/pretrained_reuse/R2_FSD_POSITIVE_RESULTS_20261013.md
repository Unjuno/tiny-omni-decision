# Fixed128-positive coverage experiment: completed, not adopted

## Acquisition and integrity

The second acquisition completed all128 fixed clips from128 uploaders. Six
previously verified files were reused; new transferred bytes57,243,798. Every
file passed hash/CRC checks and the frozen selection gate. No replacements.
Frozen384-dimensional features were finite and bitwise repeatable for all128
clips,11.59s CPU excluding load. Feature SHA:
`50712dc9e2ee1fdbbde85f8c46c37c79550d15acec48ef46d153545778a88212`.

Full waveform screen completed128×78=9,984 pairs in349.54s CPU: no >=.99 or
undefined-correlation flags, maximum .805805, no identical resampled FSD clips.
This rules out high whole-template correlation under the fixed transform only;
edited, mixed or shorter reuse remains outside its guarantee. No sealed data.

## Executed learning

Fresh seed17, frozen EfficientAT features, same74,753-parameter readout and
original-train normalization. Original train/validation/calibration retained.
Added exactly128 whole-clip positive queries per epoch; all unknown events
omitted.24epochs, same optimizer and validation-NLL selector/calibration grid.
Each epoch has23,588 known queries versus23,460 previously. Updates increase
4,416→4,440; sample order changes with corpus size. Thus this is a fixed-epoch
data augmentation comparison, not a compute/order-matched causal experiment.

Selected epoch9; final epoch24 retained separately. Training9.28s CPU, no cloud
cost or GPU training. Validation IDs exactly match the previous comparison.
An independent process reloaded best weights/normalization/temperature and
reproduced all5,832 saved probabilities bitwise (maximum difference0).

| Calibrated validation | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Original pooled features | .899691 | .270390 | .151361 | .064576 |
| +128 positive clips | .893690 | .270229 | .159043 | .064134 |

Outcome is mixed and insufficient for adoption. Speech Accuracy .7572→.8148,
but music .8909→.7716 and domestic sounds .5514→.5329. Bell NLL .4058→.2085
while recall remains0. Clapping, telephone, instrument and knock also have0
recall at the fixed decision rule; water tap has no validation positives.
Do not describe the small aggregate NLL change as solving rare-event recognition.

| Epoch | Rolling train CE | Raw validation NLL | Accuracy |
| --- | ---: | ---: | ---: |
| 1 | .4462 | .4267 | .8338 |
| 6 | .1401 | .2800 | .8831 |
| 9 selected | .1006 | .2787 | .8937 |
| 12 | .0792 | .2879 | .8892 |
| 24 final | .0387 | .3360 | .8891 |

After selection, falling train CE with worsening validation NLL is an overfitting
signal. Rolling training CE is not a fixed-example generalization gap. No extra
epochs, seed search, threshold tuning or event removal followed the result.
Only two development rooms and repeated validation use limit inference. These
128 additional positives do not identify a universal data/capacity ceiling.

## Reproduction and retained outputs

`python -m scripts.train_starss_fsd_positive` from base commit d711da3 plus the
new runner whose exact hash is saved in protocol.json. Local paths under
`C:/CodexArtifacts/pretrained-reuse-r2/`:

- `fsd50k-training-128-02/`: complete licensed acquisition and positive queries
- `fsd50k-features-128-01/`: frozen features and input hashes
- `fsd-starss-waveform-screen-128-01/`: all pairwise waveform diagnostics
- `starss-fsd-positive-128-run-01/`: best/final, full per-event histories,
  predictions, normalization, protocol, validation IDs and independent reload

Compact reports retain final per-event results, macro learning curves and hashes.
Existing models/corpora/artifacts were not overwritten; no product promotion.
