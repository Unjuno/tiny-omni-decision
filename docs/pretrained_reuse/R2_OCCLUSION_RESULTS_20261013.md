# Dense temporal occlusion comparison — 2026-10-13

## Scope and execution

Completed the predeclared mean/attention comparison on all 99 train videos (198 questions) and 257 development-validation videos (514 questions). Upstream splits and all admitted pairs were retained. This is a narrow initial/final occlusion task, not an Omni release or blind evaluation. See [fixed protocol](R2_OCCLUSION_PAIR_PROTOCOL_20261012.md).

All 356 video features completed with exit code 0: 3,082 eight-frame windows at the fixed 4 fps policy. Admission rechecked 360 feature/report hashes, finite tensors, exact question routing and cross-split prepared-window isolation. Frozen encoders and all legacy artifacts were preserved.
Extraction wall time: 139.40 minutes on CPU. Training used CPU, seed 17, hidden width 64, AdamW LR 0.001 / weight decay 0.01, eight video pairs per batch, 50 epochs. No paid compute. Source commit is recorded in the saved config.

## Selected checkpoints and controls

Earliest minimum validation NLL selected each checkpoint. Best and final weights remain separate; reload is exact. Each run consumed 9,900 question presentations, 198 unique questions / 99 unique videos, and 650 optimizer updates. Mean selected epoch 4 (792 presentations); attention selected epoch 2 (396 presentations). Mean contains 196,864 parameters but uses 147,584; attention uses 196,864. Frozen encoder parameters are additional.

| Model/control | Accuracy | NLL | Brier | ECE | Both conditions correct |
|---|---:|---:|---:|---:|---:|
| Train-only prior | 0.389105 | 1.091510 | 0.663199 | 0.053431 | 0.206226 |
| mean/normal | 0.391051 | 1.089261 | 0.662086 | 0.044944 | 0.206226 |
| mean/zero | 0.276265 | 1.098968 | 0.666904 | 0.057771 | 0.050584 |
| mean/mismatch | 0.392996 | 1.093058 | 0.663464 | 0.045652 | 0.206226 |
| mean/reverse | 0.391051 | 1.089261 | 0.662086 | 0.044944 | 0.206226 |
| attention/normal | 0.389105 | 1.089320 | 0.661018 | 0.034952 | 0.206226 |
| attention/zero | 0.305447 | 1.098853 | 0.666827 | 0.028697 | 0.132296 |
| attention/mismatch | 0.389105 | 1.090430 | 0.661556 | 0.034822 | 0.206226 |
| attention/reverse | 0.389105 | 1.089576 | 0.661188 | 0.033703 | 0.206226 |

## Learning curves

| Model | Epoch | Train Accuracy | Train NLL | Validation Accuracy | Validation NLL | Validation ECE |
|---|---:|---:|---:|---:|---:|---:|
| mean | 1 | 0.449495 | 1.059476 | 0.389105 | 1.094244 | 0.047871 |
| mean | 2 | 0.449495 | 1.054396 | 0.389105 | 1.089351 | 0.034633 |
| mean | 4 | 0.530303 | 1.010922 | 0.391051 | 1.089261 | 0.044944 |
| mean | 10 | 0.681818 | 0.789076 | 0.367704 | 1.217113 | 0.178413 |
| mean | 20 | 0.712121 | 0.598738 | 0.361868 | 1.565521 | 0.300801 |
| mean | 30 | 0.712121 | 0.543771 | 0.332685 | 1.923215 | 0.402004 |
| mean | 40 | 0.717172 | 0.463018 | 0.344358 | 2.188547 | 0.417237 |
| mean | 50 | 0.742424 | 0.452095 | 0.379377 | 2.333983 | 0.382665 |
| attention | 1 | 0.449495 | 1.059284 | 0.389105 | 1.094156 | 0.047978 |
| attention | 2 | 0.449495 | 1.053704 | 0.389105 | 1.089320 | 0.034952 |
| attention | 4 | 0.525253 | 1.006661 | 0.391051 | 1.089570 | 0.040072 |
| attention | 10 | 0.707071 | 0.721226 | 0.322957 | 1.236470 | 0.223474 |
| attention | 20 | 0.722222 | 0.529602 | 0.367704 | 1.718902 | 0.330994 |
| attention | 30 | 0.737374 | 0.437510 | 0.373541 | 2.002232 | 0.368503 |
| attention | 40 | 0.782828 | 0.385404 | 0.354086 | 2.243946 | 0.417619 |
| attention | 50 | 0.803030 | 0.363470 | 0.369650 | 2.561034 | 0.433581 |

## Interpretation and uncertainty

Both heads overfit: later training reduces training loss while validation NLL and calibration worsen. Selected checkpoints do not establish useful video-conditioned decisions: mismatching videos leaves Accuracy essentially unchanged, and both heads score zero on answering both conditions correctly in the 127 videos whose initial/final targets differ. Zero-video degradation alone is insufficient evidence of content use because it changes the feature distribution.

Video-cluster paired bootstrap uses 10,000 draws, seed 17, percentile 95% intervals, retaining both questions per video. Attention minus mean Accuracy interval is [-0.017510, 0.013619], NLL [-0.012682, 0.012516]. Mean minus train-prior Accuracy interval is [-0.013619, 0.017510]; attention minus prior is [0, 0] for per-video mean Accuracy. These conditional intervals do not correct for checkpoint selection or development-set reuse.

Close this fixed comparison without adopting either head. Do not add epochs or claim the encoder family has reached a universal limit. Remaining possibilities include spatial pooling losing object identity, readout/condition representation, limited independent training videos, and task ambiguity. Next work should localize information loss with a bounded train-only representation diagnostic before another fit; do not tune against these validation controls. The full multimodal quality and release gates remain unmet.

## Reproduction and artifacts

```powershell
python -m scripts.train_occlusion_readout --video C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-dense-features-01 --text C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-text-features-01 --output <NEW_OUTPUT>
python -m scripts.report_occlusion_readout --run <NEW_OUTPUT> --output <NEW_REPORT>
```

Original run: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-readout-01`.
Head training/control wall times: mean 5.362s; attention 8.078s. Feature extraction is separate.

| Artifact | SHA-256 |
|---|---|
| perception-occlusion-readout-01/config.json | `d3c0aac8cee69e43d28b8d7ff6e77508594586d96e90b0504f1880f5dd7cad99` |
| perception-occlusion-readout-01/result.json | `84dae4df5202750fe8ef737d1bf4a4544f4f1b42e9cd3422171552e429352c5c` |
| perception-occlusion-readout-report-01.json | `5589ef865cf684d51c00bfe406cce7621e2b05b50a3ba95f0afc67ab32a98933` |
| perception-occlusion-dense-features-01/result.json | `b7d477c34e8faf047804c77750e154d59afed3568a190eaf7cdef5380bc97fb8` |
| perception-occlusion-readout-01/mean/best.pt | `3efd2f1c9352a339a9791f01c164a22128e842a84620649e8b156bf4ef5432c9` |
| perception-occlusion-readout-01/mean/final.pt | `28a3e9c810c3249e188ba3c42e1c9bdd71167aa2cb45b974c8b28069ff422e8d` |
| perception-occlusion-readout-01/mean/history.json | `a45ea51d908431ccd3a96d861e32019053146a6a492b7116a1637cfff9bdbd90` |
| perception-occlusion-readout-01/mean/predictions.json | `c007b4abe0ac7fd149f9d9b79d9dc2b63d921720cdabd77b7acb200c8564fca0` |
| perception-occlusion-readout-01/attention/best.pt | `e07edda28d1683fad567615ace66b58bb207df6739a7cd329af23c4068cd88df` |
| perception-occlusion-readout-01/attention/final.pt | `1effbf6042f21247eb5d2ecbe794bdf2d5aea6f89119fa2f5d9e0af9b91f6e5c` |
| perception-occlusion-readout-01/attention/history.json | `3a9f160e0ce776b195cf5c6ab4a05cb69205b4a5b33cb1bf6f0a7202b91acfb1` |
| perception-occlusion-readout-01/attention/predictions.json | `44e5a94d9f5de819623fe892c7f4f15ebf85bb06fdbc313bfacbf120f746cd76` |
