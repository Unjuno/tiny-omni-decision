# BANKING77 BERT text-task source screen — 2026-11-22

## Decision

**Status: `DEVELOPMENT_CANDIDATE_GROUP_IDENTITY_REVIEW_REQUIRED`.** BANKING77 is a bounded text Decision task for comparing compact BERT-family encoders: it contains 13,083 English banking queries labeled with 77 intents, and the PolyAI-maintained dataset card declares CC BY 4.0. The task is single-domain intent routing, not general reasoning or multimodal quality. A TRAIN-only integrity audit passed exact normalized-content grouping, but the source does not establish independent speaker/customer groups. A fixed development split and one frozen MiniLM-L3 readout fit are now complete; the published test file remains unopened.

## Pinned source and rights

- Dataset: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- Resolved repository revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6` (`main` at screen time).
- The PolyAI-maintained card declares **CC BY 4.0**, describes 13,083 queries and 77 intents, and shows a 10,003-row train split and a 3,080-row test split.
- Retrieved only the revision-pinned README and TRAIN parquet. The README is 13,049 bytes, SHA-256 `f3786a5327fd6533c7c0ff3ff654e20ee3d1261d2fa41667381f59a2e42364e1`. The TRAIN parquet is 293,553 bytes, SHA-256 `8dc89b87a188509c42cce5a086e8f7c1ff4a225bdd9ae3f23e6cdfa5806e363a`.
- The publisher's tree metadata lists the test parquet as 92,409 bytes with LFS SHA-256 `1f40f777c33f0edfa6c530ff84fe2878db6490838c12d22d036f626948905341`; this metadata was read, but the test payload was not fetched or opened.

## TRAIN-only integrity findings

The pinned TRAIN file contains 10,003 rows over all 77 intent labels. Class counts range from 35 to 187 examples. NFKC + casefold + whitespace normalization yields 9,999 distinct utterance strings: four duplicate groups containing eight rows, none crossing labels. Mean/median query length is 59.47/47 characters. The full machine-readable audit is stored outside Git at `C:\CodexArtifacts\pretrained-reuse-r2-local\banking77-source-audit-20261122-attempt-01\train-audit.json` (SHA-256 `efa3bd53afd4fc703f38e5b3a8cae81eff334712bb47aab09b9cb819e247f79d`).

This check detects exact normalized duplicates only. It does not prove that queries came from distinct people, separate conversations, or non-overlapping paraphrase families. The dataset card leaves source-language-producer details as “More Information Needed” and provides no participant or conversation IDs. Treat an utterance-level split as content-disjoint development evidence, not a verified independent-speaker split.

## Frozen MiniLM-L3 baseline

The development split uses seed 17 and stratifies by class while assigning normalized utterance groups intact. It contains 8,005 training rows / 8,001 unique content groups and 1,998 validation rows / 1,998 groups; all 77 labels occur in both splits. Row and normalized-content overlap are zero. The split manifest SHA-256 is `d95bb74e6f28eac63012a95d808c6a9a82941e08960c57d52ad3e020c2a183e2`. Split JSONL hashes are recorded in the machine manifest. This is an utterance-content split; it does not prove speaker/customer independence.

The first fixed candidate uses the pinned MiniLM-L3 encoder frozen and trains only a standardized 384→77 linear readout (29,645 trainable parameters). The complete text candidate contains 17,419,469 pre-quantization parameters; the encoder weight file is 69,569,488 bytes and the readout checkpoint is 123,704 bytes. Seed 17, CPU FP32, AdamW (`lr=0.01`, `weight_decay=0.01`, batch 256), maximum 60 epochs, early-stop patience 6 and minimum validation NLL selected epoch 6; the run completed 12 epochs in 19.29 seconds. The environment was Python 3.11.9, PyTorch 2.6.0+cu124 and Transformers 5.6.2.

| Validation metric | Result |
| --- | ---: |
| Accuracy | 89.69% |
| Macro recall | 89.42% |
| Macro F1 | 89.65% |
| NLL | 0.41907 |
| Brier | 0.15487 |
| ECE-15 | 0.02810 |

Save/reload reproduced all 1,998 predictions exactly; maximum probability delta was 0. No source-group confidence interval is reported because source-person/conversation IDs are unavailable. This baseline is below the provisional 95% accuracy goal. It only tests a frozen encoder with a linear readout, so it does not establish a BERT-family ceiling or justify a general text-quality claim. The official test remains unopened.

The run is outside Git at `C:\CodexArtifacts\pretrained-reuse-r2-local\banking77-minilm-l3-readout-v1-attempt-01`. It retains the best readout, full validation probabilities, per-class counts/metrics, epoch history, effective config, metadata and reload receipt. Training code/config are in source commit `fc8d4899a5ef0cb2a8090e3f41dd0a963c25da68`; artifact hashes are in the machine manifest.

## Final-block adaptation follow-up

To test whether a frozen encoder was the limiting factor, a separate run unfroze only the final MiniLM-L3 Transformer block and the same 384-to-77 readout. It used the exact same seed-17 content-grouped train/validation split and pinned base model. It did not load the publisher test partition. Only the training policy changed; output artifacts are separate from the frozen-readout baseline.

The first implementation padded every utterance to 128 tokens. Its CPU throughput was impractical, so that owned attempt was stopped before it saved a checkpoint or metrics. The follow-up uses per-batch dynamic padding; this implementation completed in 209.24 seconds on CPU FP32. It kept the data, seed, base checkpoint, validation preprocessing, and NLL selector fixed, while changing the trainable scope to the final Transformer block plus readout and using an end-to-end AdamW schedule (block LR 2e-5, readout LR 0.01, batch size 32). It completed 6 epochs and selected epoch 2 by the predeclared minimum validation NLL rule. Training CE continued down while validation NLL did not improve after epoch 2, which is an overfitting signal. The selected result was Accuracy 85.84%, macro F1 85.86%, NLL 0.71451, Brier 0.22210, and ECE-15 0.07259. Epoch 4 had the highest accuracy (87.04%) but a worse NLL (0.73260), so it was not selected.

This final-block adaptation did not improve on the frozen readout's 89.69% Accuracy, 0.41907 NLL, 0.15487 Brier, and 0.02810 ECE-15 on this validation split. This is evidence against this particular last-block fine-tuning recipe, not against BERT-family encoders in general. It is one seed on an utterance-content split without verified speaker/customer groups, and therefore remains development-only. No test-set metric, calibration fitting, or product-quality claim is made. The complete run, curves, predictions, hashes, and reload receipt are in `C:\CodexArtifacts\pretrained-reuse-r2-local\banking77-minilm-l3-last-block-v1-attempt-02`; see the machine manifest for hashes and initialization details.

## Relevance to compact BERT-family work

This gives the BERT question a concrete, useful but narrow test: route an English banking query to one of 77 supplied intent IDs and return calibrated option probabilities. The task may diagnose text classification and calibration, but it cannot establish arbitrary-option reasoning or improve the video/audio temporal pathway by itself.

## Frozen MiniLM-L6 comparison

The official [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) declares Apache-2.0 and describes a 384-dimensional sentence encoder; the pinned Hub revision is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Actual local loading counted 22,713,216 encoder parameters and verified a finite short-text forward. Its 90,868,376-byte weight file has SHA-256 `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.

The comparison kept the seed-17 BANKING77 split, feature pooling, 77-way standardized linear readout, optimizer, early-stop rule, validation-NLL selector, and evaluation rows fixed against the L3 baseline. Only the frozen pretrained encoder checkpoint changed. On CPU FP32, 8,005 training rows (8,001 unique content groups) and 1,998 validation rows completed 11 epochs in 32.16 seconds; epoch 5 was selected by minimum validation NLL.

| Model / selected checkpoint | Accuracy | Macro F1 | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| Frozen MiniLM-L3, epoch 6 | 89.69% | 89.65% | 0.41907 | 0.15487 | 0.02810 |
| Frozen MiniLM-L6, epoch 5 | 91.44% | 91.70% | 0.33253 | 0.12766 | 0.02296 |
| L3 final-block adaptation, epoch 2 | 85.84% | 85.86% | 0.71451 | 0.22210 | 0.07259 |

The L6 candidate improves the L3 development-validation point estimates by 1.75 percentage points Accuracy and lowers NLL by 0.08654, Brier by 0.02722, and ECE-15 by 0.00514. It remains below the 95% stretch target. Training CE kept falling after epoch 5 while validation NLL worsened; epoch 8 reached the highest observed accuracy (92.09%) but a worse NLL, so it was not selected. The saved epoch-5 checkpoint reloaded with identical predictions and maximum probability delta 0. No temperature fitting was used.

This is evidence that pretrained encoder choice matters for this narrow text classification task and supports keeping the BERT-family path open. It is not an independent-source result: participant/conversation IDs are unavailable, and the same validation set selected the checkpoint and compared L3 versus L6. The publisher test remains unopened. Do not infer arbitrary-option or multimodal quality from these figures. The L6 experiment's complete config, per-class metrics, predictions, curves, run metadata, file hashes and reload receipt are in `C:\CodexArtifacts\pretrained-reuse-r2-local\banking77-minilm-l6-frozen-readout-v1-attempt-01` and the machine manifest.

## Raw-text to typed-JSON runtime verification

A local CPU runtime now loads the pinned MiniLM-L6 encoder and selected readout, takes raw English text, and returns the fixed 77-class intent probabilities through the repository's typed JSON serializer. Class IDs use the exact numeric-to-name mapping in the publisher README pinned at revision `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6` (README SHA-256 `f3786a5327fd6533c7c0ff3ff654e20ee3d1261d2fa41667381f59a2e42364e1`). The runtime checks the encoder and readout weight hashes before loading, requires the full class option set, and refuses arbitrary free-text option scoring.

The saved L6 readout was reloaded and run over all 1,998 fixed development-validation utterances in original order. Argmax matched the saved training-run predictions on all rows; maximum per-class probability absolute difference was `2.44e-6`. Recomputed metrics were Accuracy `91.44%`, macro F1 `91.70%`, NLL `0.33253`, Brier `0.12766`, and ECE-15 `0.02296`. CPU elapsed time was `24.90 s` for 1,998 one-at-a-time requests. Typed JSON smoke passed. Receipt: `C:\CodexArtifacts\pretrained-reuse-r2-local\banking77-minilm-l6-runtime-verify-attempt-02\reload-verification.json`.

The runtime is an executable narrow text classification example, not general arbitrary-option Decision reasoning. Validation is development-only and participant/conversation group independence is unverified; the publisher test remains unopened. This does not close the broader text, image, audio, video, or multi-input quality requirements.

## Machine receipt

The machine-readable source screen and pins are in [`banking77-text-source-screen-20261122.json`](../../manifests/pretrained_reuse/banking77-text-source-screen-20261122.json). The publisher test and all other sealed-audit data remain unopened. The baseline used the local pinned MiniLM-L3 checkpoint on CPU; GPU was not used.
