# Smaller video-language candidate follow-up

Existing MobileCLIP/MobileViCLIP audit remains controlling: inherited model-weight
research restrictions are unresolved for product use. No duplicate download/run.
InternVideo2 small vision distillation is not automatically a small aligned full
video/text model. X-CLIP already counts196,585,729 configuration parameters.

## All-in-one: source inspected, not admitted

Official source:https://github.com/showlab/all-in-one
Pinned commit:ae0e288a8b2c505d48057dd0f8f4c215706611df.
Fetched recursive tree, README and AllInOne/config.py; local SHA256 ledger under
`C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-source-audit-01/`.
README advertises unified raw-video/text models Ti12M,S33M,B110M and links official
Google Drive weights/hparams/logs. Counts are publisher claims, not measured
resident parameters. Paper benchmarks are not our task quality.

README states MIT, but fixed recursive tree contains no license-named file;
initial LICENSE request returned404. The full license notice and explicit weight
coverage remain unresolved; do not infer commercial weight rights from code label.
No checkpoint downloaded or executed. Preserve this candidate as REVIEW pending
license/provenance and checkpoint access checks. Old Python3.7/Lightning setup also
needs isolated compatibility inspection; do not downgrade the working environment.

The unified video/text attention is relevant to the observed failure of separately
encoded candidate features, but superiority is a hypothesis requiring actual
load, resource measurements and fixed condition/evidence controls. No current
candidate resolves the four-modality95–98% goal. Continue R1 admission before
allocating another full-corpus quality experiment.

## Public-link and released-hparams check

Both Tiny/Small official weight landing pages returned200 and named the expected
.ckpt files. This proves landing-page availability, not full weight retrieval.
The two released YAML hparams files were actually downloaded from the README's
Drive IDs and preserved locally:

- Tiny1424bytes,SHA25648e441debf3f9d350fb3b9bab7fca5812f7f93e20f746ae71e8d35453d041b84.
- Small1435bytes,SHA256ed573fc9ad074d3701aa784fde6bc1591fba6881a506826868136fec06033618.

Both specify3frames,224px,12layers,vocabulary30522,max_text_len40,MLM+ITM
pretraining; Tiny hidden192/3heads and Small hidden384/6heads. These are released
training settings, not our experiment choices or proof of runtime capability.
Long existing question-plus-candidate input requires an explicit token-length
check; do not silently truncate conditions or rephrase repeatedly against scores.
Pinned allinone_module.py and heads.py were fetched for compatibility inspection.
The main environment and all prior artifacts are unchanged. Weight license,
checkpoint content, exact resident count and actual forward remain unverified.
Local link-check.json and hparams-fetch.json record HTTP status, sizes and hashes.

## Actual input-length measurement

Fetched BERT uncased vocabulary at immutable revision
86b5e0934494bd15c9632b12f734a8a67f723594; SHA256
07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3.
With BertTokenizer(lowercase), concatenated each unchanged train question, one
space and each candidate, including special tokens and no truncation. Across
594 inputs, lengths39–43;562 exceed40. Saved all row lengths in
`all-in-one-source-audit-01/text-length-audit.json`.

Pinned model code passes max_text_len directly into BertConfig's
max_position_embeddings. Thus enlarging a tokenizer limit alone is insufficient;
a checkpoint position-embedding change would require a separate verified policy.
Do not silently truncate, discard long examples or claim the original input
contract works. A prospective meaning-preserving serialization could be fixed
without score-based prompt search, but is not yet implemented/evaluated.
This diagnostic accesses train only, no weights or validation scores.

## Fixed prospective compact serialization

Implemented `occlusion_statement` for the two exact audited question templates
and eight exact candidate relations. It retains occlusion/similar-object context,
beginning/end, person's viewpoint, hidden object and the complete candidate
relation (including distance vs left-order wording). Unknown inputs fail closed.
It neither reads targets nor maps option indices to geometric ranks.

Using the same pinned BERT vocabulary, all594 train candidate statements fit:
28–32 tokens including special tokens, no truncation. Each example keeps three
distinct statements. Tests cover all16 template/relation combinations and unknown
input rejection. This fixed wording was defined before any model scores, not
selected from a prompt sweep. Semantic adequacy still requires actual task
validation; text-length admission is not model quality or license admission.

Artifact:`all-in-one-source-audit-01/compact-text-audit.json` preserves every
statement, token length and pair-file SHA256. All old inputs/runs are untouched.
