# Pooled-video interaction comparison

Execution PASS for all three24-epoch runs; quality gate remains unmet.
Source `8321b272a43565a85c522d94a6b6977361666b8b`. Separate fresh seed17 runs, identical
sampling order, fixed weighted CE and validation macro-NLL selector.
Bilinear and zero-video controls have byte-identical initial states.
3794 unique choices/256 training scenes;3014 choices/200 validation scenes.
91,056 encounters per condition;87,262 repeats, not additional unique data.
Validation was previously observed and used for checkpoint selection.
No separate control reporting set or sealed audit was opened.

## Selected validation metrics

| Policy/input | Task | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| concat/matched | macro | 0.520209 | 0.691924 | 0.498779 | 0.017559 |
| concat/matched | explanatory | 0.538088 | 0.690653 | 0.497508 | 0.019616 |
| concat/matched | predictive | 0.525490 | 0.691796 | 0.498650 | 0.010648 |
| concat/matched | counterfactual | 0.497049 | 0.693324 | 0.500179 | 0.022414 |
| concat/shuffled | macro | 0.519450 | 0.691926 | 0.498781 | 0.015860 |
| concat/shuffled | explanatory | 0.536467 | 0.690682 | 0.497536 | 0.017988 |
| concat/shuffled | predictive | 0.525490 | 0.691795 | 0.498650 | 0.006536 |
| concat/shuffled | counterfactual | 0.496393 | 0.693301 | 0.500156 | 0.023056 |
| bilinear/matched | macro | 0.522413 | 0.693416 | 0.500216 | 0.036401 |
| bilinear/matched | explanatory | 0.514587 | 0.692388 | 0.499181 | 0.027788 |
| bilinear/matched | predictive | 0.541176 | 0.691387 | 0.498228 | 0.055489 |
| bilinear/matched | counterfactual | 0.511475 | 0.696473 | 0.503239 | 0.025925 |
| bilinear/shuffled | macro | 0.504118 | 0.693651 | 0.500480 | 0.038097 |
| bilinear/shuffled | explanatory | 0.528363 | 0.691630 | 0.498454 | 0.014826 |
| bilinear/shuffled | predictive | 0.482353 | 0.694664 | 0.501567 | 0.064438 |
| bilinear/shuffled | counterfactual | 0.501639 | 0.694658 | 0.501420 | 0.035025 |
| no_video/matched | macro | 0.519348 | 0.691908 | 0.498763 | 0.016337 |
| no_video/matched | explanatory | 0.538088 | 0.690026 | 0.496881 | 0.020791 |
| no_video/matched | predictive | 0.513726 | 0.691930 | 0.498788 | 0.011274 |
| no_video/matched | counterfactual | 0.506230 | 0.693768 | 0.500620 | 0.016945 |
| no_video/shuffled | macro | 0.519348 | 0.691908 | 0.498763 | 0.016337 |
| no_video/shuffled | explanatory | 0.538088 | 0.690026 | 0.496881 | 0.020791 |
| no_video/shuffled | predictive | 0.513726 | 0.691930 | 0.498788 | 0.011274 |
| no_video/shuffled | counterfactual | 0.506230 | 0.693768 | 0.500620 | 0.016945 |

## Paired macro-Accuracy uncertainty

- bilinear_minus_concat: +0.220pp; 95% interval[-1.831,+2.223]pp.
- bilinear_minus_no_video: +0.307pp; 95% interval[-2.151,+2.652]pp.
- matched_minus_shuffled: +1.829pp; 95% interval[-0.066,+3.720]pp.

200-scene paired bootstrap,10,000 draws,seed17,percentile intervals.
No simultaneous correction; these do not remove checkpoint-selection bias.
Zero-video matched/shuffled predictions are identical as required.

## Learning and checkpoints

| Policy | Epoch | Train macro Acc | Val macro Acc | Train macro NLL | Val macro NLL |
|---|---:|---:|---:|---:|---:|
| concat | 1 | 0.503430 | 0.501804 | 0.696606 | 0.696917 |
| concat | 2 | 0.496570 | 0.498196 | 0.693639 | 0.693564 |
| concat | 3 | 0.516477 | 0.495954 | 0.693038 | 0.693124 |
| concat | 4 | 0.535715 | 0.503617 | 0.692996 | 0.693095 |
| concat | 5 | 0.506741 | 0.505619 | 0.692917 | 0.693071 |
| concat | 6 | 0.503430 | 0.501804 | 0.692830 | 0.693060 |
| concat | 7 | 0.533670 | 0.512692 | 0.692859 | 0.693046 |
| concat | 8 | 0.542898 | 0.509340 | 0.692631 | 0.692950 |
| concat | 9 | 0.530924 | 0.513695 | 0.692524 | 0.692917 |
| concat | 10 | 0.505324 | 0.503986 | 0.692383 | 0.692917 |
| concat | 11 | 0.540680 | 0.513416 | 0.692094 | 0.692756 |
| concat | 12 | 0.533484 | 0.511116 | 0.692384 | 0.692881 |
| concat | 13 | 0.537518 | 0.511671 | 0.691858 | 0.692723 |
| concat | 14 | 0.503430 | 0.501804 | 0.694934 | 0.696080 |
| concat | 15 | 0.503253 | 0.502460 | 0.691843 | 0.693225 |
| concat | 16 | 0.532827 | 0.514910 | 0.691174 | 0.692548 |
| concat | 17 | 0.514444 | 0.507951 | 0.690867 | 0.692856 |
| concat | 18 | 0.546533 | 0.513599 | 0.689932 | 0.692184 |
| concat | 19 | 0.514176 | 0.504374 | 0.691295 | 0.693528 |
| concat | 20 | 0.519538 | 0.506242 | 0.690549 | 0.693534 |
| concat | 21 | 0.549949 | 0.510955 | 0.688336 | 0.691997 |
| concat | 22 | 0.536051 | 0.518427 | 0.688592 | 0.692868 |
| concat | 23 | 0.544988 | 0.519653 | 0.688075 | 0.692677 |
| concat | 24 | 0.546630 | 0.520209 | 0.688043 | 0.691924 |
| bilinear | 1 | 0.503430 | 0.501804 | 0.695371 | 0.695799 |
| bilinear | 2 | 0.492522 | 0.496386 | 0.695870 | 0.695705 |
| bilinear | 3 | 0.496570 | 0.498196 | 0.696510 | 0.696684 |
| bilinear | 4 | 0.515252 | 0.502150 | 0.692756 | 0.696444 |
| bilinear | 5 | 0.496570 | 0.498196 | 0.702728 | 0.703818 |
| bilinear | 6 | 0.544568 | 0.489674 | 0.689601 | 0.693906 |
| bilinear | 7 | 0.524921 | 0.501553 | 0.688985 | 0.693656 |
| bilinear | 8 | 0.550056 | 0.499884 | 0.688057 | 0.695666 |
| bilinear | 9 | 0.502123 | 0.506516 | 0.710149 | 0.712680 |
| bilinear | 10 | 0.520264 | 0.498907 | 0.695213 | 0.704873 |
| bilinear | 11 | 0.561857 | 0.516743 | 0.681738 | 0.694343 |
| bilinear | 12 | 0.562851 | 0.522413 | 0.678089 | 0.693416 |
| bilinear | 13 | 0.564335 | 0.521569 | 0.676948 | 0.699738 |
| bilinear | 14 | 0.566981 | 0.512639 | 0.678549 | 0.696288 |
| bilinear | 15 | 0.572432 | 0.512431 | 0.679229 | 0.696003 |
| bilinear | 16 | 0.570722 | 0.522725 | 0.669929 | 0.698817 |
| bilinear | 17 | 0.570545 | 0.506707 | 0.677203 | 0.706473 |
| bilinear | 18 | 0.585803 | 0.519933 | 0.665638 | 0.698731 |
| bilinear | 19 | 0.587128 | 0.514380 | 0.666266 | 0.700842 |
| bilinear | 20 | 0.604507 | 0.523199 | 0.658961 | 0.696598 |
| bilinear | 21 | 0.585482 | 0.514425 | 0.668418 | 0.706819 |
| bilinear | 22 | 0.570806 | 0.516890 | 0.672168 | 0.714846 |
| bilinear | 23 | 0.599023 | 0.519507 | 0.655131 | 0.704705 |
| bilinear | 24 | 0.624452 | 0.516739 | 0.653595 | 0.707478 |
| no_video | 1 | 0.503430 | 0.502023 | 0.693053 | 0.693123 |
| no_video | 2 | 0.503430 | 0.501804 | 0.693022 | 0.693110 |
| no_video | 3 | 0.523800 | 0.504085 | 0.693027 | 0.693107 |
| no_video | 4 | 0.535749 | 0.514823 | 0.693019 | 0.693100 |
| no_video | 5 | 0.509191 | 0.499297 | 0.692909 | 0.693062 |
| no_video | 6 | 0.503430 | 0.501804 | 0.692741 | 0.693030 |
| no_video | 7 | 0.545679 | 0.512201 | 0.692776 | 0.693003 |
| no_video | 8 | 0.545348 | 0.515458 | 0.692409 | 0.692861 |
| no_video | 9 | 0.553559 | 0.514815 | 0.692095 | 0.692749 |
| no_video | 10 | 0.503253 | 0.501804 | 0.692128 | 0.692995 |
| no_video | 11 | 0.530290 | 0.515944 | 0.691311 | 0.692592 |
| no_video | 12 | 0.538087 | 0.516240 | 0.691657 | 0.692652 |
| no_video | 13 | 0.538399 | 0.519364 | 0.690801 | 0.692515 |
| no_video | 14 | 0.545319 | 0.512609 | 0.690414 | 0.692228 |
| no_video | 15 | 0.549776 | 0.517096 | 0.689416 | 0.692031 |
| no_video | 16 | 0.551009 | 0.515300 | 0.689485 | 0.692028 |
| no_video | 17 | 0.532361 | 0.517012 | 0.689136 | 0.692899 |
| no_video | 18 | 0.553789 | 0.516137 | 0.688212 | 0.691961 |
| no_video | 19 | 0.552177 | 0.519348 | 0.687763 | 0.691908 |
| no_video | 20 | 0.534771 | 0.517019 | 0.688936 | 0.693579 |
| no_video | 21 | 0.553826 | 0.512155 | 0.686433 | 0.691950 |
| no_video | 22 | 0.540931 | 0.515363 | 0.686350 | 0.692539 |
| no_video | 23 | 0.545398 | 0.516678 | 0.686141 | 0.692961 |
| no_video | 24 | 0.550357 | 0.518455 | 0.685653 | 0.692995 |

## Resources and integrity

- concat: best epoch24, selected91,056/whole91,056 encounters, 295,169 trainable parameters,17.502s.
- bilinear: best epoch12, selected45,528/whole91,056 encounters, 295,168 trainable parameters,17.459s.
- no_video: best epoch19, selected72,086/whole91,056 encounters, 295,168 trainable parameters,17.050s.
- Total55.855s; peak allocated VRAM130,593,792bytes.
- Local RTX3080Laptop16GB,FP32; cloud cost$0. Frozen text weights/video cache
  hash checked before and after; same underlying scene/media/content gates.
- Full video/text/bilinear deployment components total132,050,176 parameters
  before quantization (114,365,184+17,389,824+295,168). No encoder loaded for
  video extraction here; cached features used. This is not all-modality quality.
- Best and final weights saved separately; best reload logits exactly match.
- Old files untouched. Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/video-interaction-01`.
  All probabilities, full config and IDs remain local; hashes/summaries in JSON.

## Smoke failure retained

smoke-01 completed compute but its tiny-scene bootstrap generated undefined
replicates and NaN intervals; its PASS_EXECUTION_ONLY flag does not validate
those intervals. Exclude them from quality conclusions. Preserve the file.
The corrected evaluator omits quality bootstrap for smoke and rejects any
missing-type replicate in full evaluation. Regression tests and smoke-02
passed before the full run. No old result was overwritten or silently repaired.

Four-modality95–98% remains unmet. Static-image readout gains do not
establish pooled-video temporal reasoning. No automatic extra training.

## Decision and bottleneck evidence

Do not adopt bilinear for this pooled-video setup: macro Accuracy gain
over concat is only0.220pp with an interval spanning zero, and selected
NLL/Brier/ECE are worse. The no-video comparison likewise remains uncertain.
Matched-minus-shuffled+1.829pp is suggestive but its interval includes zero;
neither a reliable scene benefit nor complete absence of video information
is established. All three question types remain far below95%.

Bilinear train macro NLL falls0.6781→0.6536 from epoch12 to24 while validation
NLL rises0.6934→0.7075. This is explicit overfitting evidence; best12 is
retained, not the final24. Fixed text preprocessing truncated zero questions
(maximum44 train/45 validation tokens), so truncation does not explain this run.

Further pooled-readout tuning or more epochs is not justified here. Before
another quality run, inspect reusable spatiotemporal tokens and their real
memory/runtime cost at the same frame count. This is a next diagnostic, not
a claim that unpooled tokens will solve video reasoning or a base capacity limit.
