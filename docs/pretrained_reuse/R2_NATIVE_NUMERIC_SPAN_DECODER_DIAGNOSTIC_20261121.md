# R2 frozen numeric-span representation diagnostic

## Question and scope

Can the frozen DeBERTa XSmall text representation linearly retain numeric
magnitudes from the state tokens, before spending more capacity on a decision
readout? This is a TRAIN-only diagnostic over the already fixed 6,144 FIT rows
and 768 state groups from `native-text-programmatic-group-holdout-v1`. It is
not an independent validation result: these groups were part of the prior
TRAIN fit partition. The prior holdout labels/predictions, official validation,
and sealed audit were not loaded.

The diagnostic uses gold JSON numeric spans to pool contextual token features,
then fits one shared ridge decoder (alpha 1.0) in deterministic five-fold
state-group CV (seed 17). It decodes the integer magnitude of each of the
`cost`, `risk`, and `time` fields. For an additional six-task check, the gold
field span, JSON object-to-candidate ID mapping, and task direction (minimum or
maximum) are supplied to the scorer. This oracle information means the choice
accuracy is an upper-bound-style component diagnostic, not production Decision
accuracy or end-to-end JSON behavior. It does not measure text/video/audio/image
quality, calibrated probabilities, or multimodal fusion.

A fixed within-state feature derangement assigns each candidate another
candidate's decoded value while keeping every candidate's own feature away from
itself. Its near-zero accuracy is therefore an anti-alignment control, not a
uniform-choice baseline. Its purpose is to test whether correctly aligned
numeric token representations drive the candidate ranking.

## Frozen inputs and execution

- Source commit: `859dbe757c1d12b46a1eb584c4a91022e92a7de1`.
- Config: `configs/pretrained_reuse/native_numeric_span_decoder_diag_v1.json`, SHA-256 `af2905973fa6bcb74a18b0368b68d0a3b39f44c485a467af39dcf43b76101a98`.
- Language-cache config: `configs/pretrained_reuse/native_language_cache_v1.json`, SHA-256 `daacb73079945d62c357f5d006817e15bfcff0c8893f5e32116053ede4d801c5`.
- Numeric TRAIN source SHA-256: `c34b0d5ff04707b5af059f0886117583efd59808f5ec217fd995333aab644f9d`.
- Fixed split SHA-256: `2a595eee6f442560c83a2f18a22011c68125169058bf70251538dc08d4fde6b2`; it supplies 6,144 FIT IDs / 768 FIT groups.
- Normalized corpus pins and frozen feature-bank SHA are recorded in the run summary. The state features are the raw JSON observation strings, matching the exact frozen cache token count; no `State:` prefix is added.
- Frozen encoder/tokenizer identity: `microsoft/deberta-v3-xsmall`, revision `262ae02f29173eec1c250f90804dc7edc677dcff`; the original `model.safetensors` SHA-256 is `56b89067a4e60362f498f39b7067020911618fad37d64fa23a5f959a04fd4080`, and tokenizer files are individually pinned by SHA-256. No encoder weights were loaded during this diagnostic; only the pinned cached token states and local tokenizer were used.
- Procedure: 9,216 numeric spans, shared 384D ridge, standardization fit within each training fold, fold assignment `SHA256(seed=17:protocol-id:group) mod 5`; fold group counts 171 / 153 / 133 / 163 / 148. Bootstrap: 5,000 paired resamples of whole state groups, seed 17.
- CPU only; no training, optimizer, GPU, or cloud use. Python 3.11.9, NumPy 2.4.6, Transformers 5.6.2, tokenizers 0.22.2; Intel Core i7-12700H. Successful run time 8.71 seconds.
- Successful run began `2026-11-21T10:29:18.671112Z` and ended `2026-11-21T10:29:27.376899Z`.
- Command: `python scripts/diagnose_native_numeric_span_decoder.py --diagnostic-config configs/pretrained_reuse/native_numeric_span_decoder_diag_v1.json --language-config configs/pretrained_reuse/native_language_cache_v1.json --split-manifest C:\CodexArtifacts\pretrained-reuse-r2-local\native-text-state-group-holdout\runs\programmatic-v1\preflight-01\split.json --numeric-train C:\CodexArtifacts\pretrained-reuse-r2\condition-relevance-corpus-01\numeric_train.jsonl --output C:\CodexArtifacts\pretrained-reuse-r2-local\numeric-span-decoder-diagnostic-v1\run-01`.
- External artifacts: `C:\CodexArtifacts\pretrained-reuse-r2-local\numeric-span-decoder-diagnostic-v1\run-01\summary.json` (SHA-256 `c2be6cee96239aac7570aadbb5b7f12fba4dd6e8478346acf4367a7d22bca314`) and `fit-oof-numeric-predictions.jsonl` (SHA-256 `c831eab4af0a37287c327584543c39488c63678691ebeebff96ab59a47a2eb03`). No data or model weights were added to Git.

Two pre-output attempts failed closed: first, the initial token count check exposed that the cached representation is the raw JSON string without a `State:` prefix; second, after correcting that alignment, a report-timing variable was undefined after calculations. Neither created an output directory or altered inputs. Both issues were corrected before the successful run; the raw-string alignment is now covered by a token-count check in the execution path.

## Results

| Target | OOF values | MAE | RMSE | Pearson r |
| --- | ---: | ---: | ---: | ---: |
| Cost | 3,072 | 10.69 | 13.75 | 0.971 |
| Risk | 3,072 | 10.50 | 13.27 | 0.973 |
| Time | 3,072 | 10.41 | 13.29 | 0.972 |
| Macro | 9,216 | 10.54 | 13.44 | 0.972 |

For the six extremum tasks, the oracle-assisted candidate selection scored 88.39% macro Accuracy. Individual task Accuracy was cost minimum 88.80%, cost maximum 85.81%, risk minimum 90.89%, risk maximum 88.93%, time minimum 89.58%, and time maximum 86.33%. The raw-number oracle scored 100% on all six tasks, which verifies target alignment and task scoring. Feature derangement scored 3.65%. The paired ridge-minus-derangement group-bootstrap difference was +84.74 percentage points, 95% percentile interval +83.42 to +86.00 points.

These are regression and Accuracy diagnostics; NLL, Brier, and ECE are not defined for the scalar decoder and were not reported. The bootstrap quantifies variation across these existing FIT state groups only. It does not establish performance on new source groups, natural-language descriptions, other modalities, or production inputs.

## Interpretation and next step

The result supports a narrow conclusion: frozen DeBERTa token states retain substantial linearly recoverable numeric magnitude, and supplying the correct field/object alignment lets a simple decoder rank candidates well on this generated source. This weakens “the encoder discarded all numeric information” as a blanket explanation for the earlier near-chance native readout. The remaining gap may lie in locating the relevant field and candidate, binding the question's requested operation, or the decision readout/training objective. This experiment does not distinguish those causes.

Do not select or promote a product model from this diagnostic. A useful next experiment should use the same FIT-only boundary and explicitly remove one oracle at a time (first learned field/span localization, then question-conditioned min/max selection), with one preregistered grouped protocol and independent task groups before making a quality claim. Do not reopen this same FIT result for hyperparameter tuning. HubEgo still requires the user's account approval; no gated source was accessed.