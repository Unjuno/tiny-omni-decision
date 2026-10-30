# Spatial information with a bilinear candidate readout

Execution PASS: two fresh seed17 runs completed24epochs, with identical initial
weights and shuffled question order. Both best reloads match exactly. All
frozen weights and pinned cache hashes retained; global-feature parity passed.
Source commit `e285b575cb601e978951e177c4884ca14f72d267`; exact argv/config in companion JSON.

Each condition uses40,331,392 total parameters,884,992 trainable. Same width
does not imply equal effective input rank: global repeats the mean spatial cell.
Train8,424 unique images/33,696 questions; validation512 images/2,048 questions.
Per run808,704 encounters,775,008 repeats. No new data or sealed audit.
Best selected by earliest minimum validation NLL; final saved separately.

| Condition | Best epoch | Selected encounters | Acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| global | 15 | 505440 | 0.884766 | 0.284292 | 0.154743 | 0.022169 |
| spatial | 16 | 539136 | 0.900391 | 0.253458 | 0.140201 | 0.022108 |

## Per-task selected metrics

| Condition | Task | Acc | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| global | color | 0.970703 | 0.081902 | 0.046703 | 0.020712 |
| global | texture | 0.935547 | 0.243194 | 0.106025 | 0.032508 |
| global | count | 0.634766 | 0.809074 | 0.464733 | 0.053532 |
| global | shape | 0.998047 | 0.002996 | 0.001513 | 0.002333 |
| spatial | color | 0.974609 | 0.059791 | 0.035664 | 0.012743 |
| spatial | texture | 0.923828 | 0.228276 | 0.107183 | 0.043987 |
| spatial | count | 0.705078 | 0.722229 | 0.416147 | 0.059495 |
| spatial | shape | 0.998047 | 0.003538 | 0.001810 | 0.002769 |

## Complete learning curves

| Condition | Epoch | Train Acc | Val Acc | Train NLL | Val NLL |
|---|---:|---:|---:|---:|---:|
| global | 1 | 0.798730 | 0.793457 | 0.527621 | 0.530226 |
| global | 2 | 0.813361 | 0.800293 | 0.483735 | 0.505399 |
| global | 3 | 0.852030 | 0.852051 | 0.389820 | 0.400379 |
| global | 4 | 0.866898 | 0.859863 | 0.331890 | 0.358508 |
| global | 5 | 0.875178 | 0.867676 | 0.313130 | 0.332826 |
| global | 6 | 0.884526 | 0.876953 | 0.282048 | 0.305661 |
| global | 7 | 0.875445 | 0.858887 | 0.326161 | 0.384930 |
| global | 8 | 0.880372 | 0.867188 | 0.293881 | 0.332680 |
| global | 9 | 0.891204 | 0.873047 | 0.261082 | 0.304668 |
| global | 10 | 0.888978 | 0.874512 | 0.276390 | 0.340198 |
| global | 11 | 0.894082 | 0.873535 | 0.271503 | 0.323076 |
| global | 12 | 0.892450 | 0.880371 | 0.262539 | 0.306009 |
| global | 13 | 0.885891 | 0.874023 | 0.279191 | 0.314540 |
| global | 14 | 0.885476 | 0.858887 | 0.275438 | 0.355869 |
| global | 15 | 0.903520 | 0.884766 | 0.232915 | 0.284292 |
| global | 16 | 0.900819 | 0.879395 | 0.236880 | 0.300949 |
| global | 17 | 0.896368 | 0.865723 | 0.266146 | 0.338881 |
| global | 18 | 0.899068 | 0.886719 | 0.242100 | 0.291647 |
| global | 19 | 0.908327 | 0.889160 | 0.226576 | 0.293975 |
| global | 20 | 0.902303 | 0.874512 | 0.229456 | 0.313864 |
| global | 21 | 0.892094 | 0.871094 | 0.279339 | 0.372282 |
| global | 22 | 0.902273 | 0.876465 | 0.253858 | 0.348853 |
| global | 23 | 0.906042 | 0.877930 | 0.226467 | 0.314837 |
| global | 24 | 0.899157 | 0.871582 | 0.248221 | 0.361123 |
| spatial | 1 | 0.807158 | 0.796875 | 0.509604 | 0.515996 |
| spatial | 2 | 0.829980 | 0.813477 | 0.444109 | 0.474014 |
| spatial | 3 | 0.858114 | 0.852539 | 0.379498 | 0.403984 |
| spatial | 4 | 0.885862 | 0.866699 | 0.294624 | 0.334208 |
| spatial | 5 | 0.888948 | 0.881836 | 0.279491 | 0.311759 |
| spatial | 6 | 0.895240 | 0.878418 | 0.257927 | 0.292091 |
| spatial | 7 | 0.893756 | 0.865234 | 0.287687 | 0.365896 |
| spatial | 8 | 0.905330 | 0.886230 | 0.241485 | 0.311522 |
| spatial | 9 | 0.915153 | 0.894043 | 0.207222 | 0.272289 |
| spatial | 10 | 0.913669 | 0.895996 | 0.209626 | 0.289909 |
| spatial | 11 | 0.908001 | 0.889160 | 0.236226 | 0.328612 |
| spatial | 12 | 0.914975 | 0.891113 | 0.222760 | 0.298304 |
| spatial | 13 | 0.909782 | 0.889160 | 0.223066 | 0.304555 |
| spatial | 14 | 0.919605 | 0.889648 | 0.191472 | 0.278034 |
| spatial | 15 | 0.916370 | 0.890625 | 0.211785 | 0.289143 |
| spatial | 16 | 0.931119 | 0.900391 | 0.168861 | 0.253458 |
| spatial | 17 | 0.917409 | 0.888184 | 0.207047 | 0.301962 |
| spatial | 18 | 0.927054 | 0.894043 | 0.179394 | 0.291422 |
| spatial | 19 | 0.934414 | 0.899414 | 0.164273 | 0.284145 |
| spatial | 20 | 0.928508 | 0.886230 | 0.177556 | 0.320532 |
| spatial | 21 | 0.919723 | 0.888672 | 0.207918 | 0.349165 |
| spatial | 22 | 0.943910 | 0.896973 | 0.148432 | 0.280281 |
| spatial | 23 | 0.933642 | 0.885254 | 0.155897 | 0.302459 |
| spatial | 24 | 0.935185 | 0.885254 | 0.162839 | 0.338654 |

## Paired comparisons

- spatial_minus_global: +1.562pp,95% percentile interval[+0.244,+2.881]pp.
- global_minus_small: -0.781pp,95% percentile interval[-2.100,+0.537]pp.
- spatial_minus_small: +0.781pp,95% percentile interval[-0.537,+2.051]pp.

Paired512-image bootstrap,10,000 draws,seed17. Validation has been reused
for model selection; these intervals do not remove selection bias.

## Compute and controls

Total174.723s; peak allocated VRAM309,274,624bytes; NVIDIA GeForce RTX 3080 Laptop GPU; cloud cost$0.
No encoder extraction; existing feature cache only. Rotated-image Accuracy:
- global: 0.097656.
- spatial: 0.096191.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/dinov2-spatial-bilinear-01`.
Weights stay outside Git. Full checkpoint hashes, gradients and task curves
are recorded in JSON. This narrow image diagnostic does not establish
general image reasoning or four-modality95–98% quality.

## Interpretation and next decision

Spatial retention improves the matched-capacity global control by1.56pp
and exact counting by7.03pp (63.48% to70.51%). This supports useful spatial
information for this readout on reused development validation. However,
the gain over the stronger small bilinear reference (89.26%) is only0.78pp
with an interval crossing zero. Keep both artifacts; no product replacement
or robust superiority claim. The new head is737,280 parameters larger.

Against the small reference, color98.83% to97.46%, texture94.14% to92.38%,
shape100% to99.80% decrease while counting64.06% to70.51% improves.
NLL/Brier improve slightly overall, but ECE worsens from0.017840 to0.022108.
This is a measured trade-off, not universally better calibration or quality.

Spatial epoch16 has train Accuracy93.11% versus90.04% validation and
NLL0.168861 versus0.253458. At22 train NLL improves to0.148432 while
validation worsens to0.280281: an overfitting warning. Epoch24 is not best.
Preserve the epoch16 checkpoint; do not extend this fixed run automatically.
The next design should test generalizable spatial/object readout, with a
training-only fit diagnostic before further selector reuse. A linear visual
projection conditioned by candidates does not explicitly model object relations.

Verification: Ruff passes;394 CPU tests pass,1 skips; all CI manifest/catalog
validation commands pass locally. Final commit CI is checked after push.
