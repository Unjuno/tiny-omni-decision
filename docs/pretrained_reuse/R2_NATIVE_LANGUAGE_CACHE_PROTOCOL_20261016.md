# Native language cache: fixed execution protocol

Bounded continuation of the existing native candidate input flow. No model
optimization, quality scoring, selection, calibration or final-audit access.
Original model, corpus, prior caches and native smoke artifacts are read-only.

## Encoding

- Original pinned DeBERTa xsmall,70,830,722 parameters including unused classifier.
- Local CUDA,FP32,eager attention,batch1,TF32off,deterministic algorithms,seed17.
- Original text strings; no normalization, semantic rewrite, truncation or dropped rows.
- Single observation <=512tokens: original complete forward, including CLS/SEP.
- Longer observations:510content-token windows with32overlap. Retain first CLS,
  each original content token once in original order, and last SEP. Maximum8192
  output tokens. No claim of full-context attention or improved semantic quality.
- Question/candidate uses original paired tokenization and masked mean including
  specials. Reject pairs>512 before extraction; no query windowing.
- This cache pins batch1. Earlier native multi-option smoke grouped queries in
  batches; do not assume batchwise bit identity without measurement.

## Inputs and split protection

Exactly six development files from `condition-relevance-corpus-01`: binding_train,
numeric_train,legacy_train,validation,old_validation,legacy_validation. Use their
prior audited SHA-256 values. Reject wrong splits, duplicate IDs, normalized-state
and group overlap. Preserve every row/order/option association through raw-content
SHA keys. Query features can share identical question/candidate strings; frozen
query extraction does not use targets. Observation banks stay separate by split.
No calibration, transfer or sealed-audit file is admitted. These observed
development inputs are not a fresh blind benchmark.

## Execution order and gates

Commit code/config/tests before any model execution. Fresh output directories:

1. `native-language-smoke-01`: first3 TRAIN state hashes and longest TRAIN state;
   first8 TRAIN pair hashes. Exact old native short-observation parity, original
   single-pair masked-mean parity, long-token coverage, individual save/reload.
2. `native-language-cache-01`: requires successful smoke receipt/hash with identical
   source/input pins. All4,382unique states,22,328pairs,29,164original questions;
   precompute complete bank sizes before load/write. Save index, window spans,
   exact input/config/source hashes, frozen weight hashes and resources.
3. `native-language-replay-01`: fresh process, requires complete cache receipt/hash;
   recompute every state/query and compare all saved features exactly. No duplicate
   large output banks. This verifies all persisted features under the chosen profile.

Each process has3600s wall cap,3GiB peak CUDA allocation cap,6GiB OS peak working-set
cap and2GiB disk reserve. Check full planned storage plus reserve before model load.
No runtime package upgrades. A numerical/resource/source mismatch is INCOMPLETE;
do not admit partial banks or relax a gate during execution. Preserve failed output.

The final receipt includes full extraction and final input/output hashing time,
excluding the small final receipt serialization. Print periodic completed counts.
Use model-state before/after equality; no encoder gradients or optimizer exists.
Existing frozen data, legacy runs and their selection rules remain unchanged.

## Scope after success

Success closes language input/cache preparation only. It does not train the common
readout, prove cross-window reasoning, calibrate probabilities or establish95–98%
quality. Next unify both speech/event audio feature paths, freeze the shared
training/evaluation contract, then execute the first common native readout fit.
