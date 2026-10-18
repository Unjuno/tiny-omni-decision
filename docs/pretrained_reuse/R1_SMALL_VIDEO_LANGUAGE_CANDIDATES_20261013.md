> Correction: earlier tokenizer-based length and real-text smoke claims are superseded by the tokenizer correction below.

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

## Distribution response and publisher clarification search

Publisher-repository issue searches for license/commercial returned zero results;
this is absence of discovered clarification, not proof of permission or prohibition.
Checkpoint search located issues4,17,19 but their issue bodies do not resolve
weight licensing. Query results preserved in publisher-issue-check.json.

Direct Tiny download returnedHTTP200 text/html(2468bytes), not a checkpoint.
The Drive page labels all-in-one-tiny.ckpt approximately199M and says its file is
too large for Google's virus scan. This is a scan-size notice, not a detected
malware report. No checkpoint was saved or deserialized. Do not count this HTML
response or the advertised size as downloaded weights or resident model size.
Response metadata preserved in weight-response-probe.json. An optional bs4 parser
was absent; no dependency was installed, and the notice was inspected with stdlib.

Product-use weight rights remain REVIEW. The existing README MIT declaration,
missing full license notice and unverified weight scope must be represented
accurately rather than silently upgrading this candidate to ALLOW.

## Actual Tiny checkpoint acquisition and bounded inspection

Processed the publisher-linked Drive download confirmation for local research
inspection; product license remains REVIEW. Actual file208,484,108bytes, SHA256
`d79415e85a6ade9d14e932bead53fadd7c61248fcd72808e2c6d86d46e2e9ca7`.
Saved separately at `C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-weight-audit-01/`.
No legacy model/file was modified.

Initial weights_only=True load rejected one Lightning ModelCheckpoint global.
Static torch global scan found only that external symbol, in a503-entry ZIP.
A second bounded inspection retained weights_only=True and mapped that exact
symbol to a locally defined inert marker class (no methods, no Lightning import).
No unrestricted pickle load was used. The checkpoint contains model state plus
optimizer/scheduler/callback/AMP training state; file size is not inference size.

State dictionary contains167 tensor entries totaling17,359,140 elements. This is
not an exact resident-parameter count: tied weights/buffers/unused training heads
must be reconciled with instantiated inference modules. Tensor names/shapes/dtypes
are saved in tensor-audit.json; initial rejection and static scan are also retained.
Actual model construction, strict loading, preprocessing/forward parity and
product-use rights remain unverified. Do not claim a functioning model yet.

## Checkpoint head accounting and inference source

Fetched pinned vision-transformer and temporal-roll source; hashes saved in
runtime-source-hashes.json. Inference concatenates repeated text and frame tokens,
rolls image tokens temporally within shared blocks, then pools joint features.
This is cross-modal interaction before readout, unlike our separate frozen towers.
Compatibility with installed timm/Transformers/Lightning remains untested.

Actual checkpoint floating tensor elements:17,359,100;40 integer position IDs
account for the remainder of the earlier17,359,140. Input word embeddings and
MLM decoder weights have distinct storage and unequal values: do not subtract
one as a tied alias. Excluding all mlm_score tensors leaves11,430,914 floating
elements for the proposed matching inference path. This remains state accounting,
not a verified instantiated resident count or authorized arbitrary layer removal.
A future strict loader must explicitly account for excluded MLM training heads,
position-ID buffers and every inference key, and compare actual forward behavior.
No model forward or quality claim has yet been made.

## First actual module construction: strict-load failure retained

Installed timm0.4.12/einops0.3.2 without dependencies into the separate artifact
folder all-in-one-runtime-deps-01; global packages were not changed. Imported the
pinned original vision source and constructed its Tiny model with pretrained=False
and released hparams. Initial dynamic-import registration needed sys.modules;
that tooling error was corrected before the model-state attempt.

Strict transformer-state load fails: current pinned source expects temporal_embed,
while the actual released checkpoint has no transformer.temporal_embed tensor.
The source uses this parameter in visual_embed, so silently ignoring the missing
key or inventing a zero tensor would not verify the original model. No forward
or successful load is claimed. Reproduction script preserved as
`all-in-one-tiny-weight-audit-01/load_vision.py`.

Next inspect publisher history for the source corresponding to the released
checkpoint; retain strict key checking and avoid synthetic checkpoint repairs.
Also retain source visual_embed's stochastic patch ordering when measuring parity:
evaluation contains torch.multinomial, interacting with temporal token rolling.

## Source-history resolution: actual strict transformer load

File history inspection identified the initial publisher commit
c6f4d5d18109f99ea748b11ed26a4b9239212cf4. Its temporal_embed declaration and use
are commented out. The next March14 revision9917ef5 still failed strict loading;
that attempt was preserved separately. Initial-commit vision,module,heads and
roll sources are pinned/hashes saved in all-in-one-source-initial-01/source.json.

Using the unchanged initial vision source, released Tiny config and actual
checkpoint transformer tensors, strict=True reports all keys matched.
Instantiated transformer parameters:5,524,416. No missing tensor synthesized,
no strict=False fallback, no global environment change. Evidence:
`all-in-one-tiny-weight-audit-01/vision-initial-strict-load.json` and reproducible
`all-in-one-source-initial-01/load_vision.py`.

This resolves this module's source compatibility, not full multimodal forward or
checkpoint provenance beyond key/shape compatibility. Complete text/pooler/ITM
load, original preprocessing, seeded-repeat behavior and actual predictions are
still required before quality experiments. Product license remains REVIEW.

## Actual joint matching forward passed

Constructed the matching inference path using initial-commit vision,TemporalRoll,
Pooler and ITMHead, installed BertEmbeddings, and the unmodified infer method
extracted from the pinned module AST. This avoids importing unrelated Lightning
training/data utilities. Explicitly excludes six mlm_score training-only keys;
every retained key strict-loads with no missing/unexpected entries. The historical
position_ids buffer is made persistent for exact checkpoint accounting.

Actual resident parameters:11,430,914. Synthetic one-video/three-frame224px random
RGB input plus six text tokens produced finite two-class ITM logits. CPU4threads,
FP32:0.060458s first forward,0.045626s second. Reset seed17 before each forward
reproduced logits exactly. This is only synthetic runtime evidence; no real-media
preprocessing, option calibration, video quality or deployment latency claim.
Seed-reset parity does not mean inference is independent of stochastic patch order.

Evidence:`C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-forward-01/result.json`.
Runner:`all-in-one-source-initial-01/forward_probe.py`, SHA256
1103824ff40b7ecfdf24e14cb69f455a47d678f7a4aade828cacd1995eefeb5d.
No fitting/GPU/cloud/legacy modification. Product-use license remains REVIEW.
Next verify original real-video preprocessing, input-sensitive outputs and
save/reload before proposing any quality run.

## Single real-video forward and save/reload

Fixed train video10103, media SHA verified, frames0/229/458 before exclusive cut459,
one predefined initial-condition candidate statement. Applied the pinned pixelbert
composition:Resize256,RandomResizedCrop224,ToTensor,mean/std0.5 with seed17. This is
a smoke policy, not a claim of complete downstream dataset-preprocessing parity.
Random crops can remove relevant objects; a quality protocol must account for this.

Joint CPU FP32 output finite; same patch-order seed repeat exact. Saved matching
state45,784,435bytes and strict-loaded into a new instance; logits match exactly.
First/second forward0.060367/0.044328seconds excludes decode,transform and loading.
No accuracy is calculated from this one video/candidate; ITM logits are not yet
calibrated probabilities over supplied options.

Artifacts:`C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-real-forward-01/`.
Runner:`all-in-one-source-initial-01/real_forward_probe.py`;
SHA2566a244513a7d6b9da08c3edd214584b25433706875d05d91e694c00b4e1dc7513.
Exact resident count remains11,430,914; product licensing remains REVIEW.
No new training, evaluation-set scoring or legacy artifact modification.

## Tokenizer compatibility correction — supersedes prior text checks

Candidate-order control exposed identical scores for all three ordinal candidates.
Installed Transformers BertTokenizer uses `vocab=`, while `vocab_file=` is silently
ignored. The previous constructor had vocabulary size5 and mapped ordinary words
to UNK. Therefore earlier39–43/28–32 length counts and real-text smoke semantic
claims are invalid; preserved original artifacts are not usable quality evidence.
Synthetic hardcoded-token forward, strict module loads and parameter counts are
unaffected. No training or validation quality run used this tokenizer.

Corrected constructor uses the same pinned vocabulary with `vocab=`, asserts30522
entries and distinct first/second/third token IDs. Actual594 original inputs are
41–45 tokens (all594 exceed40); compact inputs are30–34 (none exceed40).
Corrected measurements:all-in-one-source-audit-01/corrected-text-length-audit.json.

Fixed real candidate probe02 has exact candidate-order equivariance. Seed17
uncalibrated probabilities:[0.275253,0.316989,0.407758]; seed18 differs, max logit
delta0.029445. Reversing frames changes logits by up to0.074431. These are one-video
runtime controls, not accuracy or evidence of correct reasoning. Keep seed17;
seed18 is a sensitivity diagnostic, never a seed-selection candidate.
Artifacts:all-in-one-tiny-candidate-controls-02; corrected real save/reload probe
all-in-one-tiny-real-forward-02. Old01 artifacts remain preserved as invalid-text
runs. Future admission must assert vocabulary and actual token IDs before scoring.

## Fail-closed tokenizer admission

The repository now provides `pinned_bert_tokenizer.load_pinned_bert_tokenizer`.
It verifies the vocabulary SHA-256 before importing Transformers, selects the
explicit supported constructor argument, and rejects incorrect vocabulary size,
special IDs, or collapsed ordinal tokens. Regression tests cover both constructor
signatures and invalid input. Real local pinned vocabulary load produced 30,522
entries and ordinal token IDs [101, 2034, 2117, 2353, 102].
Ruff passed; full CPU suite: 653 passed, 1 skipped. This establishes tokenizer
admission only; it does not establish downstream quality or authorize adoption.

## Frozen train-only diagnostic admission

The repository tokenizer helper was exercised on the real candidate-control run03:
all logits exactly reproduce corrected run02, including candidate permutation.
New diagnostic02 scores all 99 training videos / 198 questions, seed17, three
first/middle/last pre-cut frames, frozen ITM positive-minus-negative logits and
uncalibrated option softmax. Compare normal video with normalized black pixels
(-1). No validation/audit, optimization, calibration or checkpoint selection.
This is a training-side diagnostic, not evidence of held-out generalization.

Attempt01 stopped: normalized zero pixels are interpreted as padding by upstream
visual_embed; positional interpolation requests H=0/W=0. Its partial predictions
are not a completed result. Attempt02 uses valid black-image padding masks.
Artifacts: C:/CodexArtifacts/pretrained-reuse-r1/all-in-one-tiny-train-diagnostic-02.
Runner: all-in-one-source-initial-01/train_diagnostic_black.py,
SHA256 a7f6f04293c88159d11a88277d660fe2054ed9d1b803e532f9b25c72aaea65f9.
Completion and quality remain pending at protocol publication.

Completed subsequently: [full train diagnostic results](R2_JOINT_VIDEO_ITM_DIAGNOSTIC_20261013.md).
All 99 videos completed; frozen ITM readout is not adopted (25.25% Accuracy,
zero correct changed-condition pairs). This does not establish a model-family ceiling.
