# Compact NLI metadata follow-up

Pinned three previously unexecuted compact candidates; saved README/config and
SHA256 under `C:/CodexArtifacts/pretrained-reuse-r1/compact-nli-followup-01/`.
No weights, training data, or model inference acquired/executed in this screen.

| Candidate | Source evidence | Decision |
| --- | --- | --- |
| prajjwal1/bert-mini-mnli | Card describes MNLI adaptation, reports68.04% matched/69.17% mismatched; no explicit checkpoint license found in inspected card | REVIEW; not product admitted |
| M-FAC/bert-mini-finetuned-mnli | Card reports best-of-five75.13/75.93%; no explicit checkpoint license in inspected card | REVIEW; best-of-five report is not our selection rule |
| agentlans/TinyBERT_General_4L_312D-nli | Card declares MIT but explicitly warns poor NLI performance; reports59.11% matched | No adoption or training justified by metadata alone |

Numbers are publisher reports on NLI, not local Decision measurements. They are
not comparable to the fixed numeric-condition battery and do not establish
capacity ceilings. Explicit code/base-model permissions must not be silently
substituted for derivative weight permissions. TinyBERT's MIT card declaration
does not resolve every provenance question or demonstrate useful task quality.

Primary cards (immutable revisions and full hashes in companion JSON):

- https://huggingface.co/prajjwal1/bert-mini-mnli/blob/2793a188a2d6f995f9e6a5f73d9dd8b7a3a3aaa6/README.md
- https://huggingface.co/M-FAC/bert-mini-finetuned-mnli/blob/780061727f47254ff763de653920bb8b7e2fd5f2/README.md
- https://huggingface.co/agentlans/TinyBERT_General_4L_312D-nli/blob/a0387dbd317defbc66e06f16aa9c30d869f3350e/README.md

Previously measured MiniLM2/DeBERTa numeric diagnostics are preserved. Do not
restart them, substitute small parameter count for task quality, or label this
metadata screen a completed model comparison. Conditional reasoning remains open.
