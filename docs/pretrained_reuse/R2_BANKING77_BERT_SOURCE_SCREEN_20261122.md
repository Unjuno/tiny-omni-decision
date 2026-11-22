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

The run is outside Git at `C:\CodexArtifacts\pretrained-reuse-r2-local\banking77-minilm-l3-readout-v1-attempt-01`. It retains the best readout, full validation probabilities, per-class counts/metrics, epoch history, metadata and reload receipt. Hashes are in the machine manifest.

## Relevance to compact BERT-family work

This gives the BERT question a concrete, useful but narrow test: route an English banking query to one of 77 supplied intent IDs and return calibrated option probabilities. Existing project candidates MiniLM-L3 (17.39M parameters) and MiniLM-L6 cross-encoder (22.71M) are plausible size-matched comparisons; a larger encoder must be counted against the complete multimodal parameter budget. The task may diagnose text classification and calibration, but it cannot establish arbitrary-option reasoning or improve the video/audio temporal pathway by itself.

The next useful text-side comparison is a single controlled change to the same frozen split: unfreeze only the final MiniLM-L3 encoder block while keeping the same readout, seed, optimizer family and validation-NLL selector. If it does not approach the provisional target without calibration damage, compare the already available MiniLM-L6 only if the complete product parameter budget remains satisfied. Do not use the publisher test for this selection. Do not turn the fixed-label classifier into an arbitrary-option result by relabeling the same task.

## Machine receipt

The machine-readable source screen and pins are in [`banking77-text-source-screen-20261122.json`](../../manifests/pretrained_reuse/banking77-text-source-screen-20261122.json). The publisher test and all other sealed-audit data remain unopened. The baseline used the local pinned MiniLM-L3 checkpoint on CPU; GPU was not used.
