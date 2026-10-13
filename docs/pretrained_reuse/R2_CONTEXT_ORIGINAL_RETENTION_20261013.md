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
