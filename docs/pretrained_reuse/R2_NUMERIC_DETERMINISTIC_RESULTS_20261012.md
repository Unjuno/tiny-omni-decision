# Fresh deterministic 2048-step numeric embedding comparison

Execution PASS; frozen hashes and exact best reload verified by runner.
Same initialization, 16,384-question order and validation IDs as fresh control. Zero-step
Acc/NLL/Brier/ECE match within 1e-6. Both conditions ran 2048 updates.
Original encoder-coverage warm start has inherited training; no exact optimizer resume.
Only additional trainable scope is 343 numeric embedding deltas / 131,712 parameters.
Both total 22,845,313; trainable control 10,647,169 vs numeric 10,778,881.
No new audit/transfer evaluation. Historical nondeterministic runs are separate.
Full configs stay immutable locally; ID lists in Git use counts and order SHA-256.

| Run | Step | Selector | Original Acc | Original NLL | Curriculum Acc | Curriculum NLL |
|---|---:|---:|---:|---:|---:|---:|
| control | 0 | 0.669264 | 0.602362 | 0.877434 | 0.805664 | 0.461095 |
| control | 128 | 0.712874 | 0.615748 | 0.909335 | 0.790039 | 0.516413 |
| control | 256 | 0.710860 | 0.614961 | 0.903995 | 0.780273 | 0.517725 |
| control | 384 | 0.714487 | 0.621654 | 0.895914 | 0.783203 | 0.533060 |
| control | 512 | 0.740970 | 0.602362 | 0.929133 | 0.781250 | 0.552807 |
| control | 640 | 0.762398 | 0.600787 | 0.964166 | 0.790039 | 0.560630 |
| control | 768 | 0.726529 | 0.607480 | 0.933321 | 0.791016 | 0.519737 |
| control | 896 | 0.767739 | 0.601969 | 1.034091 | 0.794434 | 0.501388 |
| control | 1024 | 0.751093 | 0.594095 | 1.040374 | 0.816895 | 0.461812 |
| control | 1152 | 0.755069 | 0.601575 | 1.044934 | 0.803711 | 0.465204 |
| control | 1280 | 0.686454 | 0.608661 | 0.937407 | 0.813965 | 0.435500 |
| control | 1408 | 0.669693 | 0.606693 | 0.907233 | 0.822266 | 0.432154 |
| control | 1536 | 0.692135 | 0.603543 | 0.915884 | 0.802246 | 0.468385 |
| control | 1664 | 0.667203 | 0.615354 | 0.885243 | 0.815430 | 0.449164 |
| control | 1792 | 0.707207 | 0.596850 | 0.965357 | 0.820801 | 0.449057 |
| control | 1920 | 0.719325 | 0.604724 | 1.014587 | 0.824219 | 0.424064 |
| control | 2048 | 0.703486 | 0.607087 | 0.950033 | 0.824707 | 0.456939 |
| numeric_delta | 0 | 0.669264 | 0.602362 | 0.877434 | 0.805664 | 0.461095 |
| numeric_delta | 128 | 0.713240 | 0.615354 | 0.910574 | 0.793945 | 0.515906 |
| numeric_delta | 256 | 0.709929 | 0.615354 | 0.903300 | 0.780762 | 0.516558 |
| numeric_delta | 384 | 0.713805 | 0.619685 | 0.895834 | 0.783203 | 0.531776 |
| numeric_delta | 512 | 0.738303 | 0.605512 | 0.930657 | 0.781738 | 0.545948 |
| numeric_delta | 640 | 0.755363 | 0.602362 | 0.961879 | 0.792480 | 0.548847 |
| numeric_delta | 768 | 0.727120 | 0.606299 | 0.938250 | 0.795898 | 0.515990 |
| numeric_delta | 896 | 0.761000 | 0.602362 | 1.028520 | 0.799805 | 0.493480 |
| numeric_delta | 1024 | 0.743300 | 0.594882 | 1.040013 | 0.827148 | 0.446586 |
| numeric_delta | 1152 | 0.738272 | 0.595276 | 1.040426 | 0.816895 | 0.436118 |
| numeric_delta | 1280 | 0.678365 | 0.609843 | 0.938502 | 0.819336 | 0.418228 |
| numeric_delta | 1408 | 0.663302 | 0.600787 | 0.912029 | 0.828613 | 0.414575 |
| numeric_delta | 1536 | 0.689278 | 0.598819 | 0.918671 | 0.806641 | 0.459885 |
| numeric_delta | 1664 | 0.662255 | 0.612598 | 0.890595 | 0.823730 | 0.433914 |
| numeric_delta | 1792 | 0.705903 | 0.595669 | 0.972240 | 0.835938 | 0.439566 |
| numeric_delta | 1920 | 0.696482 | 0.601969 | 0.995174 | 0.844238 | 0.397791 |
| numeric_delta | 2048 | 0.696826 | 0.608268 | 0.958191 | 0.837402 | 0.435460 |

## Selected checkpoints

| Run | Best step | Source | Acc | NLL | Brier | ECE |
|---|---:|---|---:|---:|---:|---:|
| control | 1664 | original | 0.615354 | 0.885243 | 0.494998 | 0.067484 |
| control | 1664 | curriculum | 0.815430 | 0.449164 | 0.252321 | 0.027511 |
| numeric_delta | 1664 | original | 0.612598 | 0.890595 | 0.497011 | 0.075976 |
| numeric_delta | 1664 | curriculum | 0.823730 | 0.433914 | 0.242692 | 0.030137 |

Selected exposure13312 questions; full run16384; selected unique states5507.
Per-source/task counts and gradients in JSON. Rolling CE is not a fixed-set gap.
Timed run1202.644s, peak VRAM3,554,337,280bytes; cloud$0.
Time excludes initial load/tokenization. Separate best/final weights saved locally.
Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/numeric-deterministic-candidate-01`.
No product adoption or 95% claim; one seed, reused development selector.
Control selected exposure 13312 questions; full 16384; selected unique states 5507.
Control timed run 1169.284s, peak VRAM 3,522,541,568 bytes.

## Paired state comparisons

| Point/source | Accuracy delta |95% interval | NLL delta |
|---|---:|---|---:|
| 512-original | 0.003150 | [0.000393,0.006039] | 0.001524 |
| 512-curriculum | 0.000488 | [-0.005859,0.006836] | -0.006859 |
| 1024-original | 0.000787 | [-0.004447,0.005924] | -0.000361 |
| 1024-curriculum | 0.010254 | [0.001953,0.018555] | -0.015226 |
| 1536-original | -0.004724 | [-0.010322,0.000788] | 0.002786 |
| 1536-curriculum | 0.004395 | [-0.003906,0.013184] | -0.008500 |
| 1920-original | -0.002756 | [-0.009479,0.004005] | -0.019413 |
| 1920-curriculum | 0.020020 | [0.012207,0.027832] | -0.026273 |
| 2048-original | 0.001181 | [-0.004715,0.006931] | 0.008158 |
| 2048-curriculum | 0.012695 | [0.003906,0.021484] | -0.021479 |
| selected-original | -0.002756 | [-0.008297,0.002760] | 0.005352 |
| selected-curriculum | 0.008301 | [-0.001465,0.018066] | -0.015250 |

State-cluster percentile bootstrap10000 draws,seed17; sample-weighted
ratios preserve unequal original group sizes. Reused development data,
no multiplicity correction or claim of a blind product result.

## Interpretation

Both independently select step1664. Numeric deltas improve curriculum NLL
0.449164 to0.433914 and Accuracy81.54% to82.37%, while original-text
Accuracy61.54% to61.26%, NLL0.885243 to0.890595, Brier0.494998 to0.497011
and ECE0.067484 to0.075976 deteriorate. The selected Accuracy-difference
interval crosses zero for both sources. Do not promote on the mean selector alone.

At fixed2048, curriculum Accuracy improves1.27pp (interval+0.39 to+2.15pp),
but original NLL worsens0.008158. These are repeated development comparisons,
not a new blind test or proof that numeric embeddings are the sole bottleneck.
The candidate's mean validation NLL worsens0.662255 at1664 to0.696826 at2048;
keeping the last checkpoint would discard the selected validation result.
Rolling training CE is cumulative over changing examples, so no fixed-train
generalization-gap claim follows. This is a validation-degradation warning.

Each run consumes16384 distinct question IDs/6056 states; selected13312/5507.
These are per-run counts: prior inherited training and reused source records
must not be counted as newly collected lifetime data. Source split is8192+8192
for the full run and6656+6656 at selection. No repeats within this run schedule.

Keep both artifacts, no product adoption, no automatic extension/seed search.
The already frozen shared-text image comparison uses the control checkpoint
selected before these candidate outcomes. No architecture/data/selector changed.
The general four-modality95% target remains unmet; narrow numeric improvement
does not resolve text semantics, transfer, video reasoning or joint fusion.

Total timed local work for this pair2371.929s, excluding initial loading/tokenizing.
Concurrent bounded CPU checks also ran; timings do not isolate hardware speed gains.
No paid compute. Energy/electricity cost was not measured. Both best/final hashes
and runtime settings are retained; weights-only artifacts are not strict resumes.
