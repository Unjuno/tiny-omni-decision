# Frozen video token readout comparison

Execution completed; previously observed development validation, not blind quality.
Source `364ad23213755c708b8b46df1b2c86fef1e1522b`. Seed17; 24epochs each; fixed split and order.
3794 unique training choices/256 scenes; 3014 validation choices/200 scenes.
Each condition sees 91,056 encounters (87,262 repeats). No sealed audit access.
456 full FP32 token tensors extracted transiently; every pooled mean matches prior cache.

| Mode | Type | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| mean | macro | 0.527184 | 0.693495 | 0.500300 | 0.040656 |
| mean | explanatory | 0.524311 | 0.692535 | 0.499323 | 0.018787 |
| mean | predictive | 0.541176 | 0.691630 | 0.498496 | 0.060669 |
| mean | counterfactual | 0.516066 | 0.696321 | 0.503080 | 0.042512 |
| attention | macro | 0.524362 | 0.692730 | 0.499559 | 0.020051 |
| attention | explanatory | 0.534846 | 0.690686 | 0.497537 | 0.011932 |
| attention | predictive | 0.537255 | 0.691699 | 0.498518 | 0.015176 |
| attention | counterfactual | 0.500984 | 0.695804 | 0.502622 | 0.033044 |
| no_video | macro | 0.515593 | 0.691880 | 0.498736 | 0.021709 |
| no_video | explanatory | 0.537277 | 0.690049 | 0.496904 | 0.020066 |
| no_video | predictive | 0.501961 | 0.691857 | 0.498715 | 0.029723 |
| no_video | counterfactual | 0.507541 | 0.693735 | 0.500587 | 0.015338 |

## Paired scene uncertainty

- attention_minus_mean: -0.282pp; 95% interval [-3.038, +2.476]pp.
- attention_minus_no_video: +0.877pp; 95% interval [-1.455, +3.281]pp.
- matched_minus_shuffled: -0.964pp; 95% interval [-2.796, +0.788]pp.

200-scene paired percentile bootstrap, 10,000 draws, seed17;
intervals do not remove validation selection bias or multiple comparisons.

## Learning curves

| Mode | Epoch | Train Acc | Val Acc | Train NLL | Val NLL |
|---|---:|---:|---:|---:|---:|
| mean | 1 | 0.503430 | 0.501804 | 0.695482 | 0.695969 |
| mean | 2 | 0.478183 | 0.483593 | 0.694778 | 0.694772 |
| mean | 3 | 0.496747 | 0.498196 | 0.696845 | 0.697504 |
| mean | 4 | 0.513389 | 0.486970 | 0.692797 | 0.696324 |
| mean | 5 | 0.498456 | 0.497926 | 0.698141 | 0.699249 |
| mean | 6 | 0.554340 | 0.491414 | 0.689521 | 0.694030 |
| mean | 7 | 0.513990 | 0.497978 | 0.690585 | 0.694411 |
| mean | 8 | 0.549465 | 0.509113 | 0.687643 | 0.694937 |
| mean | 9 | 0.505452 | 0.503473 | 0.707171 | 0.711116 |
| mean | 10 | 0.504658 | 0.501577 | 0.705085 | 0.715205 |
| mean | 11 | 0.563842 | 0.505385 | 0.682451 | 0.693956 |
| mean | 12 | 0.562657 | 0.527184 | 0.678051 | 0.693495 |
| mean | 13 | 0.561360 | 0.519096 | 0.677334 | 0.699485 |
| mean | 14 | 0.564787 | 0.517129 | 0.679250 | 0.696157 |
| mean | 15 | 0.569737 | 0.506713 | 0.677800 | 0.695617 |
| mean | 16 | 0.569146 | 0.521700 | 0.671002 | 0.698851 |
| mean | 17 | 0.568486 | 0.510157 | 0.677824 | 0.706536 |
| mean | 18 | 0.586464 | 0.520600 | 0.666263 | 0.698844 |
| mean | 19 | 0.582767 | 0.515421 | 0.667123 | 0.701410 |
| mean | 20 | 0.605876 | 0.522261 | 0.659081 | 0.696411 |
| mean | 21 | 0.576781 | 0.512771 | 0.667573 | 0.705188 |
| mean | 22 | 0.566513 | 0.513675 | 0.675416 | 0.718150 |
| mean | 23 | 0.596163 | 0.525241 | 0.655554 | 0.704495 |
| mean | 24 | 0.622235 | 0.518805 | 0.653143 | 0.705003 |
| attention | 1 | 0.503430 | 0.501804 | 0.697297 | 0.697917 |
| attention | 2 | 0.503430 | 0.501804 | 0.706728 | 0.706522 |
| attention | 3 | 0.496570 | 0.498196 | 0.716142 | 0.716798 |
| attention | 4 | 0.505327 | 0.483527 | 0.693538 | 0.695769 |
| attention | 5 | 0.489073 | 0.495144 | 0.707292 | 0.704420 |
| attention | 6 | 0.514362 | 0.501116 | 0.692500 | 0.693294 |
| attention | 7 | 0.523525 | 0.508231 | 0.690568 | 0.693705 |
| attention | 8 | 0.540254 | 0.505968 | 0.688346 | 0.693640 |
| attention | 9 | 0.509889 | 0.510341 | 0.693840 | 0.696316 |
| attention | 10 | 0.510145 | 0.500647 | 0.709841 | 0.718158 |
| attention | 11 | 0.503607 | 0.502408 | 0.700064 | 0.707746 |
| attention | 12 | 0.551999 | 0.519665 | 0.683043 | 0.692826 |
| attention | 13 | 0.552648 | 0.520475 | 0.682504 | 0.698644 |
| attention | 14 | 0.548488 | 0.519771 | 0.684808 | 0.699319 |
| attention | 15 | 0.545677 | 0.502710 | 0.686490 | 0.698588 |
| attention | 16 | 0.566206 | 0.506972 | 0.677035 | 0.700213 |
| attention | 17 | 0.560014 | 0.499882 | 0.677346 | 0.702791 |
| attention | 18 | 0.576785 | 0.503262 | 0.674378 | 0.701430 |
| attention | 19 | 0.540116 | 0.498786 | 0.691386 | 0.710899 |
| attention | 20 | 0.540482 | 0.499118 | 0.691544 | 0.706418 |
| attention | 21 | 0.575736 | 0.524362 | 0.678763 | 0.692730 |
| attention | 22 | 0.527060 | 0.505448 | 0.689444 | 0.702737 |
| attention | 23 | 0.558707 | 0.503007 | 0.680001 | 0.712704 |
| attention | 24 | 0.532537 | 0.497060 | 0.690546 | 0.712736 |
| no_video | 1 | 0.503430 | 0.501804 | 0.693053 | 0.693119 |
| no_video | 2 | 0.503430 | 0.501804 | 0.693021 | 0.693106 |
| no_video | 3 | 0.524687 | 0.505209 | 0.693025 | 0.693104 |
| no_video | 4 | 0.540101 | 0.514198 | 0.693017 | 0.693097 |
| no_video | 5 | 0.509759 | 0.501423 | 0.692909 | 0.693059 |
| no_video | 6 | 0.503430 | 0.501804 | 0.692742 | 0.693025 |
| no_video | 7 | 0.547694 | 0.509535 | 0.692774 | 0.692998 |
| no_video | 8 | 0.543406 | 0.512085 | 0.692410 | 0.692856 |
| no_video | 9 | 0.550841 | 0.512149 | 0.692104 | 0.692743 |
| no_video | 10 | 0.503253 | 0.501804 | 0.692134 | 0.692985 |
| no_video | 11 | 0.530503 | 0.514856 | 0.691330 | 0.692586 |
| no_video | 12 | 0.537525 | 0.515807 | 0.691652 | 0.692632 |
| no_video | 13 | 0.538214 | 0.518378 | 0.690827 | 0.692519 |
| no_video | 14 | 0.544158 | 0.511413 | 0.690413 | 0.692216 |
| no_video | 15 | 0.548113 | 0.514211 | 0.689450 | 0.692026 |
| no_video | 16 | 0.550655 | 0.516659 | 0.689484 | 0.692017 |
| no_video | 17 | 0.532465 | 0.518538 | 0.689127 | 0.692832 |
| no_video | 18 | 0.552262 | 0.515751 | 0.688224 | 0.691942 |
| no_video | 19 | 0.550357 | 0.515593 | 0.687778 | 0.691880 |
| no_video | 20 | 0.536286 | 0.517830 | 0.688946 | 0.693551 |
| no_video | 21 | 0.553435 | 0.511615 | 0.686484 | 0.691928 |
| no_video | 22 | 0.544932 | 0.514366 | 0.686477 | 0.692585 |
| no_video | 23 | 0.543577 | 0.516948 | 0.686224 | 0.692952 |
| no_video | 24 | 0.548616 | 0.519380 | 0.685595 | 0.692824 |

## Resources and artifacts

- mean: best epoch 12, selected encounters 45528, total parameters 393984, active parameters 295168, 40.296s.
- attention: best epoch 21, selected encounters 79674, total parameters 393984, active parameters 393984, 103.058s.
- no_video: best epoch 19, selected encounters 72086, total parameters 393984, active parameters 295168, 43.846s.
- Extraction 558.189s; total 748.778s; peak allocated VRAM 3063818752 bytes; local cost $0.
- Frozen encoders: VideoPrism 114,365,184 + MiniLM 17,389,824 parameters.
  Mean/zero-video controls carry unused key/position parameters; active capacity differs.
- Best/final weights separate; best reload exact. Original artifacts preserved.
- Artifacts `C:/CodexArtifacts/pretrained-reuse-r2/video-token-readout-01`; full probabilities, histories and hashes retained.
- smoke-01 stopped before training: one validation scene cannot be deranged.
  smoke-02 used first/last rows across task types, completed all three conditions.
- Four-modality95% and general decision quality remain unproven.

## Decision and interpretation

Do not adopt this attention readout: Accuracy is 0.282pp below its matched mean
control, and the paired interval spans zero. Its lower NLL/Brier/ECE versus mean
does not establish a reliable gain over no-video; no-video NLL is lower still.
Scene derangement does not hurt Accuracy reliably (matched-minus-shuffled -0.964pp).
This does not prove video information is absent in the pretrained tokens.

At selected attention epoch21, train Accuracy57.57% versus validation52.44%
and train NLL0.67876 versus validation0.69273 indicate limited train fit plus a gap.
At epoch24 both train and validation deteriorate; do not label this final change
pure overfitting while ignoring optimization instability. Best21 is retained.
The single attention readout does not explicitly model object-to-object relations.
These data do not isolate optimization, task/label complexity, text conditioning,
or representational limitations. No backbone capacity ceiling has been established.

Full video/text/attention parameter count is132,148,992 before quantization.
This excludes any audio encoder; it is not a complete omni-model size claim.
The logged active_parameters field counts parameters in the chosen forward branch.
For no-video, value.weight (98,304 parameters) receives zero input and has no
functional influence, despite being included in that field and optimizer group.
Thus functionally influential no-video parameters are196,864. No effective-capacity
equivalence claim is made. Mean and attention share identical query/value initialization
by resetting CPU RNG before construction; the extra keys and positions are explicit.

No automatic epoch extension. Before another full validation search, diagnose whether
the readout can fit a fixed training-only subset and whether its gradients/attention
use video. Such a diagnostic must retain original data and use a new output path.
