# Context adaptation: original-task retention

Frozen final384 all-encoder candidate versus its fixed coverage parent, no training
or checkpoint selection. All2,540 original development IDs/order match the parent
training config. These are observed validation data, not sealed or blind evaluation.

Attempt01 enforced no truncation and stopped before a completed score when an
input exceeded512tokens. Preserved its inputs/protocol. Historical training uses
`truncation=only_second,max_length=512`; attempt02 reproduces that policy for BOTH
models, without dropping questions or changing options. Maximum untruncated length
892tokens;239 candidate/state pairs are truncated per model. This is a limitation
of the inherited benchmark pipeline, not evidence that full states were handled.

| Metric | Coverage parent | Context-adapted final |
|---|---:|---:|
| Accuracy | .635827 | .628346 |
| NLL | .835591 | .876646 |
| Brier | .470606 | .485686 |
| ECE | .018915 | .043872 |

All four aggregate metrics worsen. The Accuracy change is descriptive here;
no statistical significance claim without grouped uncertainty. Together with the
context result, this establishes a measured retention trade-off, not a uniformly
stronger text model. Do not replace the current reference or claim a useful Omni
upgrade from the93% narrow context scores. Preserve the candidate for research.

Pinned base + respective complete encoder/classifier delta loaded, scope verified;
checkpoint hashes checked before/after. Finite logits; local CUDA inference only.
Inputs, prediction probabilities and protocol in
C:/CodexArtifacts/pretrained-reuse-r2/context-original-retention-02.
Both attempts remain. Runner evaluate_context_original_retention_02.py is adjacent
to these directories; its hash and all result-file hashes are retained. The result
explicitly records actual truncation policy and counts. No legacy run changed.

## Truncation-stratified saved-prediction check

Re-tokenized the same inputs without truncation to classify each question by
whether any candidate pair exceeds512tokens. No model inference.239 truncated
candidate pairs correspond to66questions;2,474questions require no truncation.

| Stratum | Parent Accuracy/NLL | Context Accuracy/NLL |
|---|---:|---:|
| No truncation,2474questions | .637025/.834622 | .626920/.877790 |
| Any candidate truncated,66questions | .590909/.871914 | .681818/.833766 |

The untruncated majority loses25correct answers; the truncated subset gains6,
giving the previously reported net loss19. The observed retention loss therefore
cannot be attributed solely to truncation. The66-question subset is small and
post-hoc: no general truncation benefit claim, no exclusions from the benchmark,
and no new model selection. Adjacent R2_CONTEXT_RETENTION_TRUNCATION_20261013.json
records joined input/prediction hashes and exact counts. Aggregate original-task
retention remains an unmet acceptance gate for the context-adapted candidate.
