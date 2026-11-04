# Matched512-step numeric embedding result

Execution PASS; frozen hashes and exact best reload verified by runner.
Same initialization,4096-question prefix and validation IDs as parent. Zero-step
Acc/NLL/Brier/ECE match within1e-6. Parent comparison restricted to0..512.
Original encoder-coverage warm start has inherited training; no exact optimizer resume.
Only additional trainable scope is343 numeric embedding deltas /131712 parameters.
Total22,845,313; trainable10,778,881. No new audit/transfer evaluation.

| Run | Step | Selector | Original Acc | Original NLL | Curriculum Acc | Curriculum NLL |
|---|---:|---:|---:|---:|---:|---:|
| parent | 0 | 0.669264 | 0.602362 | 0.877434 | 0.805664 | 0.461095 |
| parent | 128 | 0.712874 | 0.615748 | 0.909334 | 0.790039 | 0.516413 |
| parent | 256 | 0.710860 | 0.614961 | 0.903995 | 0.780273 | 0.517726 |
| parent | 384 | 0.714487 | 0.621654 | 0.895914 | 0.783203 | 0.533060 |
| parent | 512 | 0.740970 | 0.602362 | 0.929133 | 0.781250 | 0.552807 |
| numeric_delta | 0 | 0.669264 | 0.602362 | 0.877434 | 0.805664 | 0.461095 |
| numeric_delta | 128 | 0.713240 | 0.615354 | 0.910574 | 0.793945 | 0.515906 |
| numeric_delta | 256 | 0.709929 | 0.615354 | 0.903300 | 0.780762 | 0.516558 |
| numeric_delta | 384 | 0.713806 | 0.619685 | 0.895834 | 0.783203 | 0.531777 |
| numeric_delta | 512 | 0.738303 | 0.605512 | 0.930657 | 0.781738 | 0.545948 |

## Selected checkpoints

| Run | Best step | Source | Acc | NLL | Brier | ECE |
|---|---:|---|---:|---:|---:|---:|
| parent | 0 | original | 0.602362 | 0.877434 | 0.494861 | 0.019186 |
| parent | 0 | curriculum | 0.805664 | 0.461095 | 0.255456 | 0.030559 |
| numeric_delta | 0 | original | 0.602362 | 0.877434 | 0.494861 | 0.019186 |
| numeric_delta | 0 | curriculum | 0.805664 | 0.461095 | 0.255456 | 0.030559 |

Selected exposure0 questions; full run4096; selected unique states0.
Per-source/task counts and gradients in JSON. Rolling CE is not a fixed-set gap.
Timed run317.901s, peak VRAM3,504,529,920bytes; cloud$0.
Time excludes initial load/tokenization. Separate best/final weights saved locally.
Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/numeric-embedding-training-01`.
No product adoption or95% claim; one seed, reused development selector.

## Decision and limits

Both512-step searches select step0, so neither produces an improved selected
checkpoint. Zero selected exposure means zero NEW updates, not zero inherited
training. Do not package the unchanged initialization as a new quality artifact.

At fixed512, numeric deltas slightly reduce mean NLL0.740970 to0.738303;
original NLL worsens0.929133 to0.930657 while curriculum NLL improves
0.552807 to0.545948. Both remain worse than initialization. Mixed point
estimates from one seed do not establish a reliable quality gain.
Numeric-row gradients are finite/nonzero (last clipped norm0.287804), and
frozen parameters remain unchanged. Failure to improve is not a disconnected
numeric embedding gradient in this execution.

The prior2048-step run improved later, so512 steps cannot establish a plateau
or prove that embeddings are irrelevant. No automatic extension was authorized
by this experiment protocol; any longer comparison needs a new fixed plan.
No fixed training-set evaluation was recorded; rolling CE is not an overfit proof.
Ruff passes;397 CPU tests pass,1skip. Source experiment commit02866a6.
