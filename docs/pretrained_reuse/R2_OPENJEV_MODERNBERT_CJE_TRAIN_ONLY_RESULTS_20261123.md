# R2 Open-Jev ModernBERT-CJE-tiny comparison — 2026-11-23

## Decision

This is a narrow, text-only frozen-feature comparison. It does **not** justify
replacing the current MiniLM text component with ModernBERT-CJE-tiny, and it
does not close any multimodal R2 gate. On this controlled English invoice task,
the existing 3-layer MiniLM baseline had better validation macro NLL and Brier;
its accuracy advantage was uncertain under family-level resampling. Every
candidate pair was truncated at the shared 512-token limit, so these results
are conditional on a visibly inadequate long-document input path and must not
be generalized to BERT-family encoders.

## Protocol and integrity

- Source: `ZefanCai/Open-Jev`, configuration `amount-extraction-control-v1`,
  repository revision `c67699e13d0ae25e35b77165a4b6b079bedc8aba`.
- Only the publisher **train** Parquet was read. Its SHA-256 is
  `dd09738e9463fd1efcdf87137600bea0c21270ee729c4475124d684b88d0cbb2`.
  Official validation, calibration, test, and OOD partitions were not loaded.
- The data contains 32,984 synthetic English invoice decisions grouped into
  133 `document_family_id` groups. A deterministic seed-17 split held out 27
  whole families (6,696 decisions) for internal validation; 106 families
  (26,288 decisions) trained the readout. Sample IDs, group IDs, and normalized
  sanitized-content overlap were all zero.
- The model input includes only state `text`, `policy`, and `requested_role`,
  plus the question, decision type, and supplied option. `candidates`,
  `selected`, `selected_id`, sample/group IDs, and metadata are excluded from
  tokenization. Focused tests cover this boundary and family splitting.
- Both encoders remained frozen. The same option scorer was used: a shared
  128-hidden tanh network with scalar option logits. Seed 17, AdamW
  (`lr=0.002`, `weight_decay=0.0001`), up to 8 epochs, patience 3, batch size
  128, and validation macro soft NLL checkpoint selection were fixed for both.
- The experiment read each publisher-train decision once to extract frozen
  features. Only the 26,288 assigned training decisions contributed to readout
  gradients; the 6,696 internal-validation decisions were used for checkpoint
  selection and reporting.

## Models and runtime

| Encoder | Exact revision | Encoder params | Readout params | Total before quantization | Feature extraction |
|---|---|---:|---:|---:|---:|
| ModernBERT-CJE-tiny | `361683a6f681d27083f95b83a585aa44b4e03e4e` | 19,496,960 | 33,025 | 19,529,985 | 4,335.76 s |
| paraphrase-MiniLM-L3-v2 | `4ca70771034acceecb2e72475f72050fcdde4ddc` | 17,241,984 | 49,409 | 17,291,393 | 3,879.50 s |

The trained readouts took 5.02 s and 7.43 s respectively. The whole run lasted
2:17:24.6 wall time. It ran on CPU (Python 3.11.9, PyTorch 2.6.0+cu124,
Transformers 5.6.2, PyArrow 25.0.1; 2 Torch threads). The RTX 3080 was occupied
by a separate process and was not used by this experiment. No paid compute was
used.

Weights and tokenizer files were loaded from exact revisions. ModernBERT's
checkpoint file was 77,991,680 bytes, SHA-256
`ac26a8fffbb00dff4bf5698aa2f825578238c0acc7ab3b59a898ee0ae0211b72`. MiniLM's
`model.safetensors` was 69,569,488 bytes, SHA-256
`cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`.

## Validation results

Metrics compare predictions with the source's one-hot target distributions.
Accuracy is top-1 agreement; NLL and Brier use the target probability vector;
ECE uses 10 equal-width confidence bins and expected correctness. Macro values
give equal weight to the five `question_id` cells rather than to row counts.

| Encoder | Macro accuracy | Macro NLL ↓ | Macro Brier ↓ | Macro ECE ↓ |
|---|---:|---:|---:|---:|
| ModernBERT-CJE-tiny | 0.6757 | 0.7087 | 0.4070 | 0.0621 |
| paraphrase-MiniLM-L3-v2 | 0.6981 | 0.6778 | 0.3843 | 0.0234 |

| Question type (n) | ModernBERT Acc / NLL / Brier / ECE | MiniLM Acc / NLL / Brier / ECE |
|---|---|---|
| currency (1,998) | 0.3018 / 1.2285 / 0.6969 / 0.1102 | 0.3223 / 1.2229 / 0.6906 / 0.0130 |
| direction_known (1,998) | 0.9324 / 0.2597 / 0.1304 / 0.0468 | 0.9324 / 0.2478 / 0.1261 / 0.0102 |
| is_credit (1,863) | 0.7391 / 0.5759 / 0.3872 / 0.0281 | 0.7391 / 0.5742 / 0.3858 / 0.0093 |
| span (405) | 0.5926 / 0.9917 / 0.5133 / 0.0901 | 0.6840 / 0.8568 / 0.4119 / 0.0544 |
| target_present (432) | 0.8125 / 0.4876 / 0.3074 / 0.0353 | 0.8125 / 0.4872 / 0.3072 / 0.0303 |

Both selected checkpoints passed save/reload verification with maximum
probability delta 0. The selected epochs were 6 for ModernBERT and 8 for MiniLM.

### Paired uncertainty diagnostic

A paired bootstrap resampled the 27 held-out document families 2,000 times
(seed 17). Deltas below are MiniLM minus ModernBERT:

| Metric | Difference | 95% percentile interval | Bootstrap probability MiniLM is better |
|---|---:|---:|---:|
| Macro accuracy | +0.0224 | [-0.0295, +0.0753] | 0.7965 |
| Macro NLL | -0.0309 | [-0.0516, -0.0099] | 0.9980 |
| Macro Brier | -0.0227 | [-0.0354, -0.0103] | 1.0000 |

The interval is exploratory and conditional on checkpoint selection on this
same internal validation partition. It is not an independent test result.

## Main limitation and interpretation

All 93,499 option pairs exceeded the 512-token cap for both models (maximum
untruncated pair lengths: 1,086 tokens for ModernBERT and 1,033 for MiniLM).
Truncation was applied only to the first sequence, which contains the visible
state. Therefore this comparison cannot establish how either model performs
when the full invoice is available. The weak currency and span cells, especially
ModernBERT's span result, may reflect lost context as well as representation
quality. The current experiment does not separate those causes.

Open-Jev is synthetic invoice data with programmatically derived labels. It is
useful here as a controlled choice-selection task, but says nothing about
Japanese, real-world decisions, images, audio, video, or joint modalities. The
133 families also limit how precisely the group bootstrap characterizes
generalization. No blind evaluation was performed.

## Artifacts and reproducibility

- Source commit: `a36f54f23a1b716a134c539c941e3ae0a175fa80` (model run code);
  paired-bootstrap analysis helper: `8a3499a8901956d3bc72a30de9a9aee9bef2bce9`.
- Split manifest SHA-256:
  `04578e7550fa90303acca2ffec903deacf231bb04927031d6f430977ff625eea`.
- Run report SHA-256:
  `9b7810073f8c46f9057980d64580b0bf717b4d91efccb36fae804443a25154cf`.
- Paired-bootstrap result SHA-256:
  `2ac293aa9108bf5df1700d01cbd9e9e7d84ccd9d8f708b140422d392dae533a1`.
- Model-run JSON SHA-256: ModernBERT
  `a76ca7c9644e793bcd477808f6650e2b0aac6c289e6a2640f10a390243e4898c`, MiniLM
  `32bb8fca2dac3f39ebf3cc4f0335d5011ede0f01b8d248f9373766f75e68511a`.
- Saved validation-prediction SHA-256: ModernBERT
  `5f1459504623ed8883d42bb2559cf1a11e626d91a61127ac493ab1ae2b056fb9`, MiniLM
  `55a5cd9490836d7e136612c9b257aed8d7a4d9b8d2f626867fa3e70e0c3e5eff`.
- Selected readout SHA-256: ModernBERT
  `ef65008ce1af2007f7792c87f19586fe47ee189db6012938d49f698e9927fcc0`, MiniLM
  `335e5c79a799253bc1a14de771fc539aedb874cf64496de0fa1f9dceae435f21`.
- Results, readouts, and frozen features are under
  `C:\CodexArtifacts\pretrained-reuse-r2-local\openjev-amount-modernbert-comparison-20261123-attempt-01`.
  Parquet data and model weights remain outside Git.

The next justified step is to decide whether long-document text is a required
Decision workload. If it is, measure a context-preserving chunk/window input
against the same frozen encoders on a **newly fixed** group split; do not tune
against this exposed internal validation. Do not run a broad BERT-family sweep
or promote ModernBERT from these results.
