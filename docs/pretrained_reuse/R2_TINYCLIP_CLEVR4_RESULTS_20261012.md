# TinyCLIP fixed image grounding result

Execution PASS. Source4beed5b; no training, calibration, prompt search or checkpoint selection. All512previously observed validation images and four original ten-option taxonomies retained. Media hashes and ordered labels matched.83,101,185parameters actually loaded. No sealed audit or legacy artifact writes.

| Condition | Task | Accuracy | NLL | Brier | ECE15 |
| --- | --- | ---: | ---: | ---: | ---: |
| correct_image | color | 0.74414 | 0.96428 | 0.39640 | 0.09538 |
| correct_image | texture | 0.25781 | 2.20532 | 0.85990 | 0.09710 |
| correct_image | count | 0.20703 | 2.16535 | 0.86798 | 0.15394 |
| correct_image | shape | 0.71875 | 0.94144 | 0.42276 | 0.14809 |
| correct_image | macro | 0.48193 | 1.56910 | 0.63676 | 0.12363 |
| rotated_image | color | 0.08398 | 3.92708 | 1.35332 | 0.59026 |
| rotated_image | texture | 0.09766 | 2.86737 | 1.01830 | 0.25285 |
| rotated_image | count | 0.09570 | 2.32691 | 0.90528 | 0.04680 |
| rotated_image | shape | 0.11523 | 3.41837 | 1.21464 | 0.45664 |
| rotated_image | macro | 0.09814 | 3.13493 | 1.12289 | 0.33664 |
| uniform | color | 0.09961 | 2.30259 | 0.90000 | 0.00039 |
| uniform | texture | 0.11133 | 2.30259 | 0.90000 | 0.01133 |
| uniform | count | 0.09961 | 2.30259 | 0.90000 | 0.00039 |
| uniform | shape | 0.11914 | 2.30259 | 0.90000 | 0.01914 |
| uniform | macro | 0.10742 | 2.30259 | 0.90000 | 0.00781 |

Correct-minus-rotated macro Accuracy: 38.38points, paired512-image percentile95% interval [36.28,40.48]points,10,000resamples, seed17. The same four task outcomes stay together per image. This is one deterministic no-fixed-point rotation, not an estimate across all possible image assignments. Same-class matches in the rotated arm were retained.

The aligned model uses image evidence on this restricted task:48.19% vs9.81% rotated. Color74.41% and shape71.88% transfer without training, but texture25.78% and count20.70% remain poor. Native similarity-softmax is not reliably calibrated (macroECE0.1236). Uniform tie-breaking picks the first option, so its measuredAccuracy10.74% reflects class frequencies, not exactly10%.

This does not improve the existing trained VideoPrism readout (~87.9% on this subset). That reference differs in task training and preprocessing; do not infer a backbone ranking. TinyCLIP has genuine pretrained image/text alignment but is not ready to replace the visual path. Candidate vocabularies reveal the task, so arbitrary question comprehension and video temporal reasoning remain unproved. No95% claim.

Local FP32 CUDA execution 5.814s, peak allocatedVRAM 386213376bytes. Labels/media checks precede the timer; no cloud charges.Output `C:\CodexArtifacts\pretrained-reuse-r2\tinyclip-clevr4-grounding-01`; full file hashes in JSON.

Exact saved prediction rows: 6144 (512images ×4tasks ×3conditions); each contains10probabilities. All are finite and normalized. No rows dropped.
