# Shared text features with matched video/no-video readouts

Execution completed; previously observed development validation, not blind quality.
Source `69089312618f737407c46e627235810a0aea35dc`. Seed17; 24epochs each; fixed split and order.
3794 unique training choices/256 scenes; 3014 validation choices/200 scenes.
Each condition sees 91,056 encounters (87,262 repeats). No sealed audit access.
456 full FP32 token tensors extracted transiently; every pooled mean matches prior cache.

| Mode | Type | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| minilm_attention | macro | 0.524362 | 0.692730 | 0.499559 | 0.020051 |
| minilm_attention | explanatory | 0.534846 | 0.690686 | 0.497537 | 0.011932 |
| minilm_attention | predictive | 0.537255 | 0.691699 | 0.498518 | 0.015176 |
| minilm_attention | counterfactual | 0.500984 | 0.695804 | 0.502622 | 0.033044 |
| minilm_no_video | macro | 0.515593 | 0.691880 | 0.498736 | 0.021709 |
| minilm_no_video | explanatory | 0.537277 | 0.690049 | 0.496904 | 0.020066 |
| minilm_no_video | predictive | 0.501961 | 0.691857 | 0.498715 | 0.029723 |
| minilm_no_video | counterfactual | 0.507541 | 0.693735 | 0.500587 | 0.015338 |
| shared_attention | macro | 0.497922 | 0.693170 | 0.500023 | 0.005731 |
| shared_attention | explanatory | 0.503241 | 0.693186 | 0.500039 | 0.000948 |
| shared_attention | predictive | 0.490196 | 0.693257 | 0.500110 | 0.014227 |
| shared_attention | counterfactual | 0.500328 | 0.693067 | 0.499920 | 0.002020 |
| shared_no_video | macro | 0.501804 | 0.693145 | 0.499998 | 0.001908 |
| shared_no_video | explanatory | 0.499190 | 0.693162 | 0.500015 | 0.002391 |
| shared_no_video | predictive | 0.501961 | 0.693142 | 0.499995 | 0.000511 |
| shared_no_video | counterfactual | 0.504262 | 0.693130 | 0.499983 | 0.002823 |

## Paired scene uncertainty

- shared_minus_minilm_video: -2.644pp; 95% interval [-5.593, +0.167]pp.
- shared_minus_minilm_no_video: -1.379pp; 95% interval [-4.204, +1.444]pp.
- minilm_video_minus_no_video: +0.877pp; 95% interval [-1.455, +3.281]pp.
- shared_video_minus_no_video: -0.388pp; 95% interval [-2.194, +1.428]pp.
- minilm_matched_minus_shuffled: -0.964pp; 95% interval [-2.796, +0.788]pp.
- shared_matched_minus_shuffled: -0.220pp; 95% interval [-1.442, +1.030]pp.

200-scene paired percentile bootstrap, 10,000 draws, seed17;
intervals do not remove validation selection bias or multiple comparisons.

## Learning curves

| Mode | Epoch | Train Acc | Val Acc | Train NLL | Val NLL |
|---|---:|---:|---:|---:|---:|
| minilm_attention | 1 | 0.503430 | 0.501804 | 0.697297 | 0.697917 |
| minilm_attention | 2 | 0.503430 | 0.501804 | 0.706728 | 0.706522 |
| minilm_attention | 3 | 0.496570 | 0.498196 | 0.716142 | 0.716798 |
| minilm_attention | 4 | 0.505327 | 0.483527 | 0.693538 | 0.695769 |
| minilm_attention | 5 | 0.489073 | 0.495144 | 0.707292 | 0.704420 |
| minilm_attention | 6 | 0.514362 | 0.501116 | 0.692500 | 0.693294 |
| minilm_attention | 7 | 0.523525 | 0.508231 | 0.690568 | 0.693705 |
| minilm_attention | 8 | 0.540254 | 0.505968 | 0.688346 | 0.693640 |
| minilm_attention | 9 | 0.509889 | 0.510341 | 0.693840 | 0.696316 |
| minilm_attention | 10 | 0.510145 | 0.500647 | 0.709841 | 0.718158 |
| minilm_attention | 11 | 0.503607 | 0.502408 | 0.700064 | 0.707746 |
| minilm_attention | 12 | 0.551999 | 0.519665 | 0.683043 | 0.692826 |
| minilm_attention | 13 | 0.552648 | 0.520475 | 0.682504 | 0.698644 |
| minilm_attention | 14 | 0.548488 | 0.519771 | 0.684808 | 0.699319 |
| minilm_attention | 15 | 0.545677 | 0.502710 | 0.686490 | 0.698588 |
| minilm_attention | 16 | 0.566206 | 0.506972 | 0.677035 | 0.700213 |
| minilm_attention | 17 | 0.560014 | 0.499882 | 0.677346 | 0.702791 |
| minilm_attention | 18 | 0.576785 | 0.503262 | 0.674378 | 0.701430 |
| minilm_attention | 19 | 0.540116 | 0.498786 | 0.691386 | 0.710899 |
| minilm_attention | 20 | 0.540482 | 0.499118 | 0.691544 | 0.706418 |
| minilm_attention | 21 | 0.575736 | 0.524362 | 0.678763 | 0.692730 |
| minilm_attention | 22 | 0.527060 | 0.505448 | 0.689444 | 0.702737 |
| minilm_attention | 23 | 0.558707 | 0.503007 | 0.680001 | 0.712704 |
| minilm_attention | 24 | 0.532537 | 0.497060 | 0.690546 | 0.712736 |
| minilm_no_video | 1 | 0.503430 | 0.501804 | 0.693053 | 0.693119 |
| minilm_no_video | 2 | 0.503430 | 0.501804 | 0.693021 | 0.693106 |
| minilm_no_video | 3 | 0.524687 | 0.505209 | 0.693025 | 0.693104 |
| minilm_no_video | 4 | 0.540101 | 0.514198 | 0.693017 | 0.693097 |
| minilm_no_video | 5 | 0.509759 | 0.501423 | 0.692909 | 0.693059 |
| minilm_no_video | 6 | 0.503430 | 0.501804 | 0.692742 | 0.693025 |
| minilm_no_video | 7 | 0.547694 | 0.509535 | 0.692774 | 0.692998 |
| minilm_no_video | 8 | 0.543406 | 0.512085 | 0.692410 | 0.692856 |
| minilm_no_video | 9 | 0.550841 | 0.512149 | 0.692104 | 0.692743 |
| minilm_no_video | 10 | 0.503253 | 0.501804 | 0.692134 | 0.692985 |
| minilm_no_video | 11 | 0.530503 | 0.514856 | 0.691330 | 0.692586 |
| minilm_no_video | 12 | 0.537525 | 0.515807 | 0.691652 | 0.692632 |
| minilm_no_video | 13 | 0.538214 | 0.518378 | 0.690827 | 0.692519 |
| minilm_no_video | 14 | 0.544158 | 0.511413 | 0.690413 | 0.692216 |
| minilm_no_video | 15 | 0.548113 | 0.514211 | 0.689450 | 0.692026 |
| minilm_no_video | 16 | 0.550655 | 0.516659 | 0.689484 | 0.692017 |
| minilm_no_video | 17 | 0.532465 | 0.518538 | 0.689127 | 0.692832 |
| minilm_no_video | 18 | 0.552262 | 0.515751 | 0.688224 | 0.691942 |
| minilm_no_video | 19 | 0.550357 | 0.515593 | 0.687778 | 0.691880 |
| minilm_no_video | 20 | 0.536286 | 0.517830 | 0.688946 | 0.693551 |
| minilm_no_video | 21 | 0.553435 | 0.511615 | 0.686484 | 0.691928 |
| minilm_no_video | 22 | 0.544932 | 0.514366 | 0.686477 | 0.692585 |
| minilm_no_video | 23 | 0.543577 | 0.516948 | 0.686224 | 0.692952 |
| minilm_no_video | 24 | 0.548616 | 0.519380 | 0.685595 | 0.692824 |
| shared_attention | 1 | 0.503430 | 0.501804 | 0.697707 | 0.699034 |
| shared_attention | 2 | 0.487870 | 0.499722 | 0.693920 | 0.694041 |
| shared_attention | 3 | 0.496570 | 0.498196 | 0.693263 | 0.693793 |
| shared_attention | 4 | 0.504341 | 0.501367 | 0.693071 | 0.694056 |
| shared_attention | 5 | 0.514642 | 0.497922 | 0.692803 | 0.693170 |
| shared_attention | 6 | 0.513142 | 0.490391 | 0.692766 | 0.693428 |
| shared_attention | 7 | 0.494596 | 0.509753 | 0.694182 | 0.693443 |
| shared_attention | 8 | 0.499715 | 0.493722 | 0.697953 | 0.696009 |
| shared_attention | 9 | 0.500361 | 0.497270 | 0.705025 | 0.738569 |
| shared_attention | 10 | 0.507392 | 0.492856 | 0.699563 | 0.707093 |
| shared_attention | 11 | 0.534806 | 0.497092 | 0.691041 | 0.699230 |
| shared_attention | 12 | 0.539517 | 0.488264 | 0.690014 | 0.701246 |
| shared_attention | 13 | 0.534088 | 0.505907 | 0.688003 | 0.696562 |
| shared_attention | 14 | 0.547606 | 0.500859 | 0.689287 | 0.697420 |
| shared_attention | 15 | 0.534748 | 0.500812 | 0.689058 | 0.701322 |
| shared_attention | 16 | 0.556293 | 0.503391 | 0.682697 | 0.702678 |
| shared_attention | 17 | 0.530831 | 0.491628 | 0.708548 | 0.731036 |
| shared_attention | 18 | 0.562970 | 0.504509 | 0.682573 | 0.705954 |
| shared_attention | 19 | 0.588230 | 0.500652 | 0.677358 | 0.707552 |
| shared_attention | 20 | 0.600015 | 0.514279 | 0.672957 | 0.700000 |
| shared_attention | 21 | 0.606220 | 0.497497 | 0.672621 | 0.709195 |
| shared_attention | 22 | 0.561291 | 0.490715 | 0.677628 | 0.712381 |
| shared_attention | 23 | 0.585916 | 0.507424 | 0.667092 | 0.706621 |
| shared_attention | 24 | 0.610453 | 0.494786 | 0.662739 | 0.708971 |
| shared_no_video | 1 | 0.503430 | 0.501804 | 0.693119 | 0.693145 |
| shared_no_video | 2 | 0.503430 | 0.501804 | 0.693103 | 0.693165 |
| shared_no_video | 3 | 0.503430 | 0.501804 | 0.693112 | 0.693159 |
| shared_no_video | 4 | 0.526033 | 0.482882 | 0.693127 | 0.693162 |
| shared_no_video | 5 | 0.504389 | 0.499400 | 0.693116 | 0.693166 |
| shared_no_video | 6 | 0.503430 | 0.501804 | 0.693098 | 0.693176 |
| shared_no_video | 7 | 0.508115 | 0.496130 | 0.693119 | 0.693165 |
| shared_no_video | 8 | 0.504566 | 0.500493 | 0.693096 | 0.693180 |
| shared_no_video | 9 | 0.503430 | 0.501804 | 0.693080 | 0.693188 |
| shared_no_video | 10 | 0.510186 | 0.495693 | 0.693096 | 0.693175 |
| shared_no_video | 11 | 0.528696 | 0.499601 | 0.693079 | 0.693188 |
| shared_no_video | 12 | 0.503962 | 0.501149 | 0.693099 | 0.693172 |
| shared_no_video | 13 | 0.518207 | 0.478142 | 0.693063 | 0.693202 |
| shared_no_video | 14 | 0.517169 | 0.504839 | 0.693065 | 0.693193 |
| shared_no_video | 15 | 0.526588 | 0.500038 | 0.693017 | 0.693224 |
| shared_no_video | 16 | 0.510956 | 0.495041 | 0.693017 | 0.693226 |
| shared_no_video | 17 | 0.523993 | 0.485651 | 0.692964 | 0.693254 |
| shared_no_video | 18 | 0.504175 | 0.496567 | 0.692928 | 0.693299 |
| shared_no_video | 19 | 0.511665 | 0.496344 | 0.692953 | 0.693261 |
| shared_no_video | 20 | 0.509708 | 0.494158 | 0.692916 | 0.693305 |
| shared_no_video | 21 | 0.530352 | 0.496340 | 0.692793 | 0.693362 |
| shared_no_video | 22 | 0.505122 | 0.497223 | 0.692803 | 0.693421 |
| shared_no_video | 23 | 0.500843 | 0.489822 | 0.692889 | 0.693508 |
| shared_no_video | 24 | 0.509769 | 0.493670 | 0.692764 | 0.693378 |

## Resources and artifacts

- minilm_attention: best epoch 21, selected encounters 79674, total parameters 393984, active parameters 393984, 104.864s.
- minilm_no_video: best epoch 19, selected encounters 72086, total parameters 393984, active parameters 295168, 41.458s.
- shared_attention: best epoch 5, selected encounters 18970, total parameters 393984, active parameters 393984, 110.216s.
- shared_no_video: best epoch 1, selected encounters 3794, total parameters 393984, active parameters 295168, 44.011s.
- Extraction 597.392s; total 907.170s; peak allocated VRAM 3148188160 bytes; cloud charges $0; electricity unmeasured.
- Frozen video:114,365,184; text: MiniLM17,389,824 or shared Cross-Encoder22,713,601.
  Mean/zero-video controls carry unused key/position parameters; active capacity differs.
- Best/final weights separate; best reload exact. Original artifacts preserved.
- Artifacts `C:/CodexArtifacts/pretrained-reuse-r2/shared-text-video-01`; full probabilities, histories and hashes retained.
- smoke-01 failed on worker imports; smoke-02 passed all four conditions.
- Four-modality95% and general decision quality remain unproven.

## Probability-score paired comparisons

- shared_minus_minilm_video, nll: +0.000441; 95% interval [-0.004074, +0.005010].
- shared_minus_minilm_video, brier: +0.000464; 95% interval [-0.003996, +0.004988].
- shared_minus_minilm_no_video, nll: +0.001264; 95% interval [-0.001376, +0.003914].
- shared_minus_minilm_no_video, brier: +0.001262; 95% interval [-0.001370, +0.003902].
- minilm_video_minus_no_video, nll: +0.000849; 95% interval [-0.002042, +0.003653].
- minilm_video_minus_no_video, brier: +0.000824; 95% interval [-0.002030, +0.003585].
- shared_video_minus_no_video, nll: +0.000025; 95% interval [-0.000231, +0.000280].
- shared_video_minus_no_video, brier: +0.000025; 95% interval [-0.000231, +0.000280].
- minilm_matched_minus_shuffled, nll: +0.000930; 95% interval [-0.001144, +0.003019].
- minilm_matched_minus_shuffled, brier: +0.000897; 95% interval [-0.001143, +0.002947].
- shared_matched_minus_shuffled, nll: +0.000069; 95% interval [-0.000109, +0.000246].
- shared_matched_minus_shuffled, brier: +0.000069; 95% interval [-0.000109, +0.000246].

## Measured decision

Neither text policy establishes useful video contribution: matched-minus-shuffled
and video-minus-no-video intervals include zero for Accuracy/NLL/Brier.
The shared policy selects epoch5 at49.79% macro Accuracy; its no-video model
selects epoch1 at50.18%. No sharing adoption or removal of existing MiniLM.
Both MiniLM24-epoch histories exactly reproduce the previous token-readout run.
Shared attention overfits after its selected epoch: train NLL falls from0.692803
to0.662739 while validation NLL rises from0.693170 to0.708971 by epoch24.
Low selected ECE accompanies near-chance predictions; it is not high quality.
Do not extend this same probe merely because the product target is unmet.
The next hypothesis needs different grounding evidence, not additional epochs
or a claim that the entire model family has reached its capacity ceiling.

## Interpretation limits

No automatic adoption or additional epochs. No-video has fewer functionally
influential parameters. These controls measure video contribution.
Macro weights question types equally, and bootstrap resamples whole scenes.
A weak-baseline match does not establish useful temporal reasoning or a capacity ceiling.
The combined parameter budget requires one actually shared loaded text tower;
this separate video experiment alone is not an assembled <=200M Omni runtime.
No sealed audit, new seed search or quantization was performed.
