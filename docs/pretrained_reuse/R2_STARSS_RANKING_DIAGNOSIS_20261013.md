# Saved-prediction ranking diagnosis

No new fitting, calibration, threshold search, model selection or sealed audit.
Computed tie-aware AUROC and stepwise Average Precision from saved validation
probabilities for both fixed comparison arms. AP uses whole equal-score groups;
single-class AUROC and zero-positive AP are null. Three CPU tests cover perfect,
reversed, tied, single-class and nonfinite inputs. No dependency installation.

## Pooled-feature arm

| Event | AUROC | AP | Positive training windows / recordings / rooms |
| --- | ---: | ---: | ---: |
| speech | .9326 | .9626 | 1631 / 47 / 6 |
| clapping | .6538 | .0505 | 9 / 6 / 3 |
| telephone | .7732 | .1915 | 27 / 11 / 4 |
| laughter | .8229 | .4632 | 273 / 39 / 6 |
| domestic sounds | .8342 | .8696 | 249 / 10 / 4 |
| footsteps | .4132 | .0410 | 105 / 31 / 6 |
| door | .8188 | .1582 | 60 / 22 / 6 |
| music | .9723 | .9739 | 339 / 11 / 5 |
| instrument | .5135 | .0159 | 73 / 3 / 1 |
| water tap | undefined | undefined | 28 / 6 / 2 |
| bell | .6099 | .0782 | 0 / 0 / 0 |
| knock | .9471 | .3482 | 10 / 7 / 3 |

Speech/music rankings are stronger than several rare events. Footsteps and
instrument rankings do not suggest a simple 0.5-threshold fix. Bell has no
positive training evidence; instrument positives occupy only one training room.
These facts prioritize independent positive-source coverage and source shift
diagnosis over another threshold or head search. They do not prove data scarcity
is the sole cause. Footsteps already span six training rooms, so increasing
class counts alone is not a demonstrated remedy.

AP depends on prevalence; the companion JSON includes validation prevalence
for every event and both arms. Knock has only four validation positives, and
water tap none. Two validation rooms, checkpoint selection and repeated observed
development use prevent blind/general quality claims. High AUROC/AP is not
95–98% decision Accuracy or calibrated output. No per-event threshold was chosen.

Run: `python -m scripts.diagnose_starss_ranking`.
Local output: `C:/CodexArtifacts/pretrained-reuse-r2/starss-ranking-diagnosis-01/`.
Input prediction hashes are preserved. Existing model artifacts remain unchanged.
