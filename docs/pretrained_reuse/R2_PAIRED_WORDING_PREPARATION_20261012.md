# Paired wording training preparation

Status: data prepared and oracle checked; no training or new evaluation started.

The frozen curriculum has 8,192 questions / 1,024 states. Each state sees three
fields and both directions, but only one comparison-word family and one syntax
family. Six domain adjectives (cheapest/priciest/safest/riskiest/earliest/latest)
are absent from this curriculum. Some appear in original training questions;
this is a literal coverage audit, not proof that the pretrained model lacks them.
See R2_COMPOSITIONAL_COVERAGE_AUDIT_20261012.json for actual counts.

## Prepared intervention

Retain original view and add three generic comparison paraphrases to each
extremum/time question. Preserve state, options/order, target, oracle, parent
question and latent-state identity. Count/conditional questions remain once.
26,624 rows represent 8,192 parent questions and 1,024 states, not new assets.
All generated rows pass the existing symbolic oracle. No evaluation corpus is
used as training data. No domain adjectives or diagnostic templates are added.
This tests generic wording consistency, not complete domain-language coverage.

Changing syntax and synonym together is a composite augmentation intervention;
it cannot identify which individual word or syntax caused an effect.

## Required controlled training comparison

Before running, freeze two matched schedules with identical parent IDs/order,
initial checkpoint, optimizer, update budget and original/curriculum ratio.
Control repeats each original question; intervention cycles its available views.
Sample parent questions first, then select a view: uniform row sampling would
quadruple comparison-task weight and confound the result. Preserve count and
conditional task exposure. Count repeated parents separately from unique views.
Use existing development validation only for checkpoint selection; keep all
transfer diagnostics outside selection and never load sealed audit.
No superiority or >=95% claim follows from data preparation.

## Artifacts and failed preparation

First output `paired-wording-training-views-01` is preserved but superseded:
question strings/targets were correct, while comparison_variant metadata still
reported the parent word family. Corrected metadata is regression-tested in
`tests/test_paired_wording.py`; use only `paired-wording-training-views-02`.
Neither output was used for training. No old corpus, weights or config changed.

Output: C:/CodexArtifacts/pretrained-reuse-r2/paired-wording-training-views-02
SHA-256: 3e7722c5bd8494aa5b6f714b55f40f2ee551f82014007687bdb543a3977bb3de
