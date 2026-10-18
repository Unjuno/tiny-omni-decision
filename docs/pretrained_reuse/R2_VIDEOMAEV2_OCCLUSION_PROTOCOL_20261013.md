# Fixed compact-encoder occlusion comparison

Predeclared before extraction or quality evaluation. Same 99 train / 257 development-validation videos, 198 / 514 questions, exact pair hashes and original dense sampling plans as the completed VideoPrism comparison. No sealed audit or new labels. Existing artifacts remain read-only.

The question is whether an action-pretrained, distilled 22.15M VideoMAE V2 encoder provides useful conditional evidence for this task at smaller cost. This is an encoder-system comparison, not a single-variable architecture attribution: pretraining, width, spatial resolution and preprocessing differ.

Each old 4fps eight-frame window supplies the same eight frame identities, duplicated consecutively to satisfy the new model's sixteen-frame input. No additional temporal evidence is introduced, and reveal cuts stay fixed. Duplication differs from native sixteen-distinct-frame sampling; a poor result cannot establish the encoder's best possible performance. RGB full resize224 and mean/std0.5 remain fixed. Use the pinned source/checkpoint in R1_VIDEOMAEV2_SMALL_20261013.md, frozen/eval, CPU four threads, FP32. Cache384-dimensional window features with source/media/content hashes. Reject cross-split prepared content overlap.

Same existing MiniLM question/candidate cache, mean and attention readouts, hidden64, seed17, AdamW lr0.001 weight decay0.01, eight paired videos per batch, fifty epochs, earliest minimum validation NLL selection. Feature input width alone changes in the readout; record its actual parameter count. Save best/final checkpoints, full curves, predictions, input hashes and exact reload evidence. Controls: zero video, cyclic mismatched video and reverse window order. Report Accuracy/NLL/Brier/ECE, initial/final and changed-target pair correctness, train/validation gap, and video-cluster bootstrap (10,000, seed17, percentile95%). Reused development data and checkpoint selection preclude blind-test claims.

No learning-rate, epoch, prompt, layer or seed search. Do not promote solely on smaller size or an aggregate increase: require evidence of useful matched-video contribution and evaluate all remaining product gates separately. No full encoder training, quantization, new paid compute or live-process intervention.

Commands, from the active repository:

```powershell
python -m scripts.extract_occlusion_videomaev2_features --output C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-videomaev2-features-01
python -m scripts.train_occlusion_readout --video C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-videomaev2-features-01 --text C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-text-features-01 --output C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-videomaev2-readout-01
python -m scripts.report_occlusion_readout --run C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-videomaev2-readout-01 --output C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-videomaev2-report-01.json
```

Run later commands only after prior success and input admission. Current status: protocol and width regression tests prepared; execution results will be recorded separately.

Execution ledger: the initial import failed before creating output because an old decoder import required SciPy. The new extractor now uses no legacy decoder import. Attempt `features-01` then stopped after two videos: the new cut assertion incorrectly treated upstream `-1` (no cut) as a frame index. Existing decoder semantics were checked; a regression test now covers sentinel, exclusive boundary and invalid negative values. Failed artifacts are retained. Source `a12a399` started extraction into `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-videomaev2-features-02/`; use this new path in the training command only after successful complete extraction. Twelve targeted CPU tests pass. No readout training or new quality result exists yet.

Matched-model reporting is now implemented with `--reference C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-readout-01`. Exact video/question/target routing, saved prediction hashes and training-policy equality are checked before paired bootstrap. A self-reference replay of the existing run reproduces all metrics and returns exactly zero differences for both heads and all reported metrics. Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-self-reference-verification-01.json`. This verifies comparison mechanics and provides no new model-quality evidence. Alignment and bootstrap tests: three passed. Feature extraction remains the original live `features-02` process; no duplicate extraction or readout training has been started.
