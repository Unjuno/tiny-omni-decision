# Frozen shared-text image readout comparison

Execution PASS: two fresh seed17 runs completed24epochs, with identical initial
weights and shuffled question order. Both best reloads match exactly. All
frozen weights and pinned cache hashes retained; global-feature parity passed.
Source commit `e9eaf9a6ba1b4feb5e004704e9d3fa4022f01e5f`; exact argv/config in companion JSON.

Both use 884,992 trainable head parameters. Image-only totals: MiniLM
40,331,392 versus shared Cross-Encoder 45,655,169. Combined-system savings
require reusing the SAME resident Cross-Encoder for text Decision tasks.
This experiment does not implement or validate that entire combined runtime.
Train8,424 unique images/33,696 questions; validation512 images/2,048 questions.
Per run808,704 encounters,775,008 repeats. No new data or sealed audit.
Best selected by earliest minimum validation NLL; final saved separately.

| Condition | Best epoch | Selected encounters | Acc | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| minilm | 16 | 539136 | 0.900391 | 0.253458 | 0.140201 | 0.022108 |
| shared | 23 | 775008 | 0.904297 | 0.255284 | 0.136715 | 0.010872 |

## Per-task selected metrics

| Condition | Task | Acc | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| minilm | color | 0.974609 | 0.059791 | 0.035664 | 0.012743 |
| minilm | texture | 0.923828 | 0.228276 | 0.107183 | 0.043987 |
| minilm | count | 0.705078 | 0.722229 | 0.416147 | 0.059495 |
| minilm | shape | 0.998047 | 0.003538 | 0.001810 | 0.002769 |
| shared | color | 0.966797 | 0.099381 | 0.054033 | 0.017955 |
| shared | texture | 0.916016 | 0.275299 | 0.124787 | 0.038488 |
| shared | count | 0.736328 | 0.637392 | 0.363924 | 0.034826 |
| shared | shape | 0.998047 | 0.009063 | 0.004117 | 0.006114 |

## Complete learning curves

| Condition | Epoch | Train Acc | Val Acc | Train NLL | Val NLL |
|---|---:|---:|---:|---:|---:|
| minilm | 1 | 0.807158 | 0.796875 | 0.509604 | 0.515996 |
| minilm | 2 | 0.829980 | 0.813477 | 0.444109 | 0.474014 |
| minilm | 3 | 0.858114 | 0.852539 | 0.379498 | 0.403984 |
| minilm | 4 | 0.885862 | 0.866699 | 0.294624 | 0.334208 |
| minilm | 5 | 0.888948 | 0.881836 | 0.279491 | 0.311759 |
| minilm | 6 | 0.895240 | 0.878418 | 0.257927 | 0.292091 |
| minilm | 7 | 0.893756 | 0.865234 | 0.287687 | 0.365896 |
| minilm | 8 | 0.905330 | 0.886230 | 0.241485 | 0.311522 |
| minilm | 9 | 0.915153 | 0.894043 | 0.207222 | 0.272289 |
| minilm | 10 | 0.913669 | 0.895996 | 0.209626 | 0.289909 |
| minilm | 11 | 0.908001 | 0.889160 | 0.236226 | 0.328612 |
| minilm | 12 | 0.914975 | 0.891113 | 0.222760 | 0.298304 |
| minilm | 13 | 0.909782 | 0.889160 | 0.223066 | 0.304555 |
| minilm | 14 | 0.919605 | 0.889648 | 0.191472 | 0.278034 |
| minilm | 15 | 0.916370 | 0.890625 | 0.211785 | 0.289143 |
| minilm | 16 | 0.931119 | 0.900391 | 0.168861 | 0.253458 |
| minilm | 17 | 0.917409 | 0.888184 | 0.207047 | 0.301962 |
| minilm | 18 | 0.927054 | 0.894043 | 0.179394 | 0.291422 |
| minilm | 19 | 0.934414 | 0.899414 | 0.164273 | 0.284145 |
| minilm | 20 | 0.928508 | 0.886230 | 0.177556 | 0.320532 |
| minilm | 21 | 0.919723 | 0.888672 | 0.207918 | 0.349165 |
| minilm | 22 | 0.943910 | 0.896973 | 0.148432 | 0.280281 |
| minilm | 23 | 0.933642 | 0.885254 | 0.155897 | 0.302459 |
| minilm | 24 | 0.935185 | 0.885254 | 0.162839 | 0.338654 |
| shared | 1 | 0.699015 | 0.701172 | 0.870047 | 0.881189 |
| shared | 2 | 0.778312 | 0.771973 | 0.607040 | 0.628157 |
| shared | 3 | 0.835322 | 0.831543 | 0.457373 | 0.480507 |
| shared | 4 | 0.860963 | 0.858398 | 0.379705 | 0.398747 |
| shared | 5 | 0.864168 | 0.859375 | 0.364767 | 0.391758 |
| shared | 6 | 0.876692 | 0.872070 | 0.323643 | 0.351732 |
| shared | 7 | 0.882686 | 0.869141 | 0.308371 | 0.358065 |
| shared | 8 | 0.884437 | 0.875977 | 0.309589 | 0.339286 |
| shared | 9 | 0.902629 | 0.886719 | 0.252775 | 0.303220 |
| shared | 10 | 0.906191 | 0.888184 | 0.248835 | 0.301128 |
| shared | 11 | 0.907971 | 0.891602 | 0.236295 | 0.282209 |
| shared | 12 | 0.906695 | 0.893066 | 0.238367 | 0.277415 |
| shared | 13 | 0.910078 | 0.891113 | 0.231455 | 0.279676 |
| shared | 14 | 0.915984 | 0.881348 | 0.218421 | 0.286320 |
| shared | 15 | 0.916667 | 0.889160 | 0.217129 | 0.282842 |
| shared | 16 | 0.911414 | 0.885254 | 0.221508 | 0.295174 |
| shared | 17 | 0.919842 | 0.884766 | 0.199130 | 0.272835 |
| shared | 18 | 0.921890 | 0.895996 | 0.197288 | 0.270493 |
| shared | 19 | 0.924205 | 0.889648 | 0.191677 | 0.278397 |
| shared | 20 | 0.931357 | 0.896484 | 0.193160 | 0.287440 |
| shared | 21 | 0.923136 | 0.893066 | 0.193795 | 0.283645 |
| shared | 22 | 0.936817 | 0.897461 | 0.162246 | 0.256504 |
| shared | 23 | 0.933939 | 0.904297 | 0.165281 | 0.255284 |
| shared | 24 | 0.941002 | 0.897461 | 0.152799 | 0.257456 |

## Paired comparisons

- shared_minus_minilm, accuracy: +0.003906; 95% percentile interval [-0.009277, +0.017090].
- shared_minus_minilm, nll: +0.001825; 95% percentile interval [-0.022179, +0.025024].
- shared_minus_minilm, brier: -0.003486; 95% percentile interval [-0.016287, +0.009433].
- color, accuracy: -0.007812; 95% percentile interval [-0.027344, +0.009766].
- color, nll: +0.039590; 95% percentile interval [+0.005709, +0.074170].
- color, brier: +0.018369; 95% percentile interval [-0.003282, +0.040546].
- texture, accuracy: -0.007812; 95% percentile interval [-0.029297, +0.013672].
- texture, nll: +0.047023; 95% percentile interval [-0.023178, +0.114878].
- texture, brier: +0.017604; 95% percentile interval [-0.012025, +0.046611].
- count, accuracy: +0.031250; 95% percentile interval [-0.013672, +0.074219].
- count, nll: -0.084837; 95% percentile interval [-0.141423, -0.027638].
- count, brier: -0.052223; 95% percentile interval [-0.088265, -0.015763].
- shape, accuracy: +0.000000; 95% percentile interval [-0.005859, +0.005859].
- shape, nll: +0.005525; 95% percentile interval [-0.000729, +0.012309].
- shape, brier: +0.002306; 95% percentile interval [-0.002359, +0.007457].

Paired512-image bootstrap,10,000 draws,seed17. Validation has been reused
for model selection; these intervals do not remove selection bias.

## Compute and controls

Total177.687s; peak allocated VRAM309,274,624bytes; NVIDIA GeForce RTX 3080 Laptop GPU; cloud cost$0.
No encoder extraction; existing feature cache only. Rotated-image Accuracy:
- minilm: 0.096191.
- shared: 0.098145.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/shared-text-image-01`.
Weights stay outside Git. Full checkpoint hashes, gradients and task curves
are recorded in JSON. This narrow image diagnostic does not establish
general image reasoning or four-modality95–98% quality.

## Interpretation constraints

No automatic product adoption. Report per-task trade-offs, including calibration
and exact counting. A wide interval crossing zero is not proof of equivalence.
The same development images selected checkpoints; intervals do not establish
blind generalization or remove repeated-testing/model-selection bias.
Native video, general questions, simultaneous fusion and all-modality95% remain open.

## Measured interpretation and next action

MiniLM 24-epoch history exactly matches its prior spatial run: True.
The shared condition selects23 versus16, therefore selected checkpoints
have different exposure even though both full runs use the same24-epoch budget.
Sharing raises aggregate Accuracy90.04% to90.43% (+0.39pp, image-bootstrap
interval -0.93 to+1.71pp), while NLL slightly worsens0.253458 to0.255284.
Brier improves0.140201 to0.136715 and ECE0.022108 to0.010872.

Counting improves70.51% to73.63%, but its Accuracy-difference interval crosses
zero. Count NLL/Brier improve; color and texture Accuracy each fall0.78pp
and their NLL/Brier worsen. Shape Accuracy stays99.80% but NLL/Brier worsen.
This trade-off does not establish uniform quality preservation or95% image quality.

At selected23, fixed-train Accuracy93.39% versus90.43% validation and NLL
0.165281 versus0.255284. At24 train NLL falls to0.152799 but validation rises
to0.257456 and Accuracy falls89.75%: overfitting warning, no automatic extension.

Keep both readouts. Sharing saves17,389,824 parameters only in a full system
already using this exact Cross-Encoder for text. Video readout compatibility
with the shared text features remains untested; do not remove MiniLM from
the current video runtime or claim a verified sub200M combined candidate yet.
The next integration gate is a matched video text-feature comparison; retain
the existing video encoder/data/controls and report failures without promotion.
