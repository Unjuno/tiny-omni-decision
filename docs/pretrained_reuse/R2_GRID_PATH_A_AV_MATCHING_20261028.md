# GRID Path A synchronized AV matching probe — 2026-10-28

## Question and scope

Does a small readout over the existing frozen Whisper and temporal VideoPrism
features use synchronized audio and video to match an utterance to its video?
This is a narrow, speaker-disjoint, four-choice development probe for the R2-M
both-required task gate. It is not a general audiovisual decision benchmark.

For each of 768 training and 256 validation utterances, the audio query is
matched to one of four same-speaker GRID videos from its quartet; each video is
the target for exactly one query in that quartet. The other three videos are
distractors. Candidate order is independently randomized per training query
and fixed for validation. Train and validation speakers are disjoint (24 and
8 respectively), as are selected utterance IDs. The same quartet, examples,
optimizer, seed (17), 50-epoch budget, and architecture size were used for
Audio-only, Video-only and Joint arms. Checkpoint selection was by minimum
four-choice validation NLL. Each head has 115,457 trainable parameters.

This cohort's validation has already been used in earlier GRID development
readouts. It was reused here by design and is not blind. No sealed audit was
opened, no model/backbone was changed, and no GPU or cloud compute was used.

## Results

Four-choice chance accuracy is 25%. Metrics below use 256 validation queries;
Brier is the sum over four classes, and ECE uses ten equal-width confidence bins.

| Arm | Best epoch | Best Accuracy | Best NLL | Best Brier | Best ECE | Epoch 50 Accuracy | Epoch 50 NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Audio-only | 1 | 0.2305 | 1.3863 | 0.7500 | 0.0195 | 0.2305 | 1.3863 |
| Video-only | 20 | 0.2500 | 1.3863 | 0.7500 | 0.00004 | 0.2500 | 1.3863 |
| Joint | 2 | 0.3125 | 1.3702 | 0.7434 | 0.0678 | 0.3477 | 6.3498 |

Audio-only remains at uniform probabilities because every quartet has all four
targets exactly once. Video-only also remains effectively uniform. Joint
validation accuracy reaches 37.5% at epoch 20, but its NLL has already risen
from the selected 1.3702 at epoch 2 to 5.0213. At epoch 50, rolling train NLL
is 0.00005 while validation NLL is 6.3498 and mean confidence is 0.8957. The
late accuracy increase therefore comes with severe overconfidence and is not a
better checkpoint under the fixed selector. Rolling train NLL is not a fixed
train-set generalization-gap measurement.

### Joint best-checkpoint input controls

| Input condition | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Matched | 0.3125 | 1.3702 | 0.7434 | 0.0678 |
| Audio missing | 0.2500 | 1.3901 | 0.7519 | 0.0263 |
| Video missing | 0.2305 | 1.3863 | 0.7500 | 0.0195 |
| Both missing | 0.2305 | 1.3863 | 0.7500 | 0.0195 |
| Audio shuffled within quartet | 0.2070 | 1.4512 | 0.7816 | 0.1108 |
| Stale video from a different quartet, same speaker | 0.2422 | 1.4148 | 0.7620 | 0.0704 |
| Video temporal offset +4 frames | 0.3203 | 1.3737 | 0.7452 | 0.0783 |

The stale-video control was checked to contain no correct source utterance. A
10,000-replicate paired bootstrap resampling the eight validation speakers
(seed 1703) gave matched-minus-control speaker-macro Accuracy intervals of
0.0273–0.0977 for audio missing, 0.0273–0.1368 for video missing,
0.0273–0.1367 for both missing, 0.0234–0.1836 for shuffled audio,
0.0000–0.1250 for stale video and -0.0234–0.0078 for the temporal offset.
These intervals are descriptive only: there are eight speaker clusters, the
validation is repeatedly used development data, and the selected checkpoint
was chosen on the same validation set.

## Interpretation

The matched Joint head is modestly above chance at its selected checkpoint,
while either unimodal head is near chance as the task construction predicts.
This is weak exploratory evidence that the fused head can exploit the pairing
task, but not evidence of robust generalization: the selected validation score
is only 31.25%, and the training curve severely overfits. The matched condition
does not beat the +4-frame temporal offset, whose paired interval includes zero.
Thus this run does not establish useful fine-grained temporal synchronization.

The R2-M M-2 gate remains open. Do not promote this head, call this a quality
pass, or infer that frozen audiovisual representations in general cannot solve
matching. The small cohort, eight validation speakers, the GRID source/task,
the frozen representation, and the readout are all possible limits. Further
hyperparameter selection on this validation set would add little evidence; a
future quality claim needs a new rights-cleared, task-matched, speaker- or
source-disjoint evaluation set.

## Reproduction and integrity

Source commit: `68d0844d270ed13c150b751c064645660640bf9c`.

Command:

```powershell
python -m scripts.fit_grid_path_a_av_matching --output-dir C:\CodexArtifacts\pretrained-reuse-r2\grid-path-a-av-matching-20261028-attempt-01
python -m scripts.analyze_grid_path_a_av_matching --run-dir C:\CodexArtifacts\pretrained-reuse-r2\grid-path-a-av-matching-20261028-attempt-01
```

The fit took 79.14 seconds on CPU. Exact wall-clock start/end timestamps,
hardware inventory and peak memory were not serialized by this short run and
are UNKNOWN. No media was extracted during the fit. The external artifact
directory is `C:/CodexArtifacts/pretrained-reuse-r2/grid-path-a-av-matching-20261028-attempt-01/`.
It contains separate `*-best.pt` and `*-last.pt` files, histories, saved
validation predictions, candidate/control IDs, configuration and normalizer;
none of those binaries are in Git.

- Frozen cohort hash: `61b70af8cb826ded36986aaa53b55ef88f3e5973542d9237df1c951086bab752`.
- Frozen feature result hash: `40a06e5b4a3c69b6c965a48b59d4f08a32000f7b25c0a11cd1038f37af2e87e3`.
- Effective config hash: `d3448cad08244653de966dfdbe9bb2ffb66582f00b7bfdba7992b08c79aec716`.
- Run result hash: `22673a2967bc747182f6f12fb65a6a1940d65ddff90a2e2f958cd1217e723b95`.
- Independent replay/bootstrap analysis hash: `ca0436c6b2d2b500d2d81284b544a21e1ecb7639d12b076ef8417a3f1810525b`.
- Audio / Video / Joint selected checkpoint SHA-256: `d14499c625f53d8130707499a249760e7eb7f45577f246827c696423ea907a3c` / `f1ffbcf1c8edfc0e199abc7ca8066589d91c589e3c28b4e97333390a72836943` / `b1d261a54a08dd52817d4d9d447bfd5ebca7c7feb7a332f77e047402d7859a8b`.
- Independent reload reproduced all three saved validation probability arrays exactly (maximum absolute delta 0; zero argmax changes), and confirmed split IDs, order, targets and candidate identities.
- Fit and analysis scripts: `scripts/fit_grid_path_a_av_matching.py`, `scripts/analyze_grid_path_a_av_matching.py`.

This run did not change frozen features, prior readouts, teacher artifacts or
sealed-audit data.
