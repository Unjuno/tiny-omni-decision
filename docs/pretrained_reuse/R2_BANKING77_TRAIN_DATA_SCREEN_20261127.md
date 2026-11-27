# BANKING77 train-only data screen — 2026-11-27

## Disposition

**License gate: candidate ALLOW under the repository's CC-BY-4.0 policy, with
attribution. Quality/task admission: limited to a train-derived, grouped
development probe; it is not admitted as general or product-quality evidence.**
BANKING77 is a useful task-shaped candidate for a later text Decision
comparison: it has 77 supplied intent labels for natural customer-service
queries. It is a single banking domain and cannot establish general text or
multimodal quality. This screen inspected only the publisher's train CSV; the
publisher test data and the project sealed audit were not opened.

## Pinned source and rights

- Publisher repository: `PolyAI-LDN/task-specific-datasets`.
- Train data commit: `9d081458ff52e53cf7e848f414e6e9344e4e6696`.
- Rights/card revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`.
- Hugging Face dataset-card revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`;
  README SHA-256 `f3786a5327fd6533c7c0ff3ff654e20ee3d1261d2fa41667381f59a2e42364e1`.
- Train file: `banking_data/train.csv`, 839,073 bytes, SHA-256
  `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`.
- The same train-file hash was independently confirmed at the rights/card
  revision. The repository `LICENSE` at that revision is CC BY 4.0, SHA-256
  `7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661`.
- The dataset card marks language/annotation creators as expert-generated,
  but leaves collection/source-producer details as “More Information Needed”
  and annotators as `N/A`; it says no additional annotations are included.
- The pinned publisher dataset card declares `cc-by-4.0`; attribution and
  modification notices must be retained if examples or derived data are
  shared. This source declaration does not establish rights in material the
  publisher may not control or settle privacy/personality-rights questions.

The [publisher repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b)
lists the banking data and points to the dataset paper. The [dataset card at
its pinned Hub revision](https://huggingface.co/datasets/PolyAI/banking77/tree/1846984189f2bc0ac51c6fc3a600fa9e1714e0c6)
reports 13,083 queries across 77 intents. The [paper](https://aclanthology.org/2020.nlp4convai-1.5/)
describes a single-domain, fine-grained intent task and explicitly notes that
some intent categories partially overlap.

## Train-only integrity findings

The audit script verifies the input SHA-256 before parsing and writes only
aggregate statistics, row indices and label names; it never writes utterance
text. It inspected 10,003 rows, all 77 expected categories, and the exact
`text,category` schema.

| Check | Result |
|---|---:|
| Train rows / intents | 10,003 / 77 |
| Examples per intent, minimum / median / maximum | 35 / 127 / 187 |
| Repeated normalized-text groups / excess rows | 31 / 31 |
| Conflicting-label normalized groups | 1 |
| Lexical near-duplicate candidate pairs / cross-intent pairs | 786 / 57 |
| Rows flagged by basic email, IBAN, long-digit, sort-code, or phone patterns | 0 |

Grouping the 786 candidate edges by connected components yields 1,192 flagged
rows in 487 components. Forty-two components contain more than one intent
(36 with two labels and 6 with three); the largest components contain 9 rows.
This single-link grouping is deliberately conservative for split planning:
transitive links can join examples that are not direct near-duplicates. It is
not evidence that all rows in a component have the same meaning or that their
labels are wrong. Any future validation split should keep each candidate
component intact and report how many examples this removes from stratified
allocation, while retaining all rows and labels.

Duplicate normalization uses NFKC, case folding, punctuation removal and
whitespace collapse. The conflicting group is row IDs 2002 and 3160: after
normalization, both ask “Has my top up gone through?” but one is labeled
`pending_top_up` and the other `top_up_reverted`. The original wording differs
only by a hyphen. The paper itself notes that these intents are semantically
close. This is an unresolved target ambiguity in the source data; neither row
was relabeled or removed. Keep the pair visible in any future protocol and do
not count either as an unambiguous single-target success without a declared
policy.

The lexical near-duplicate screen uses token-set Jaccard >= 0.80 after
NFKC/casefold/punctuation normalization. Candidate generation requires a
shared word bigram occurring in at most 80 rows; it is intentionally a review
screen, not an exhaustive clustering algorithm. Of 786 flagged pairs, 57 cross
intent labels. Connected components expose 42 multi-label groups. These pairs
may represent hard distinctions, source ambiguity, or ordinary lexical
similarity; none was silently removed or relabeled. They need source/task
review before any model fit. The saved receipt records row IDs, labels and
similarity only, not utterance text. A component-aware split can reduce direct
lexical leakage but cannot establish user/session-disjoint generalization,
because the source has no stable user/session identifiers.

The zero PII-pattern count is only a narrow heuristic scan, not proof that no
personal or sensitive information is present. The dataset card leaves source
language producers, collection details and annotator identity/procedure
unspecified. The CSV also has no persistent user, conversation, session,
annotator or source-event identifier. Exact normalized duplicates can be kept
together, but this metadata cannot establish user/session-disjoint
generalization.

## Follow-up and next gate

A human text review of the 57 cross-intent candidate pairs did not find grounds
to silently correct or remove examples. It confirmed both sensible
modifier-based distinctions and unresolved generic-versus-specific boundary
cases; the exact normalized conflicting pair remains unchanged. A deterministic
whole-component train/validation split was frozen, and one CPU-only frozen
MiniLM readout probe was run. See the [probe report](R2_BANKING77_FROZEN_MINILM_PROBE_20261127.md).
The internal validation reached 92.23% macro Accuracy but was used for
checkpoint selection and is now exposed. Do not retune on it or treat it as a
blind test. The source's missing user/session provenance and known label
ambiguity still prevent broad or product claims. Any next quality comparison
needs new, independently reviewed source groups or an adjudicated label set;
keep the publisher test and project sealed audit closed.

## Reproduction and boundary

From the pinned train file, reproduce the audit with:

```powershell
python scripts/audit_banking77_train_quality.py `
  C:\path\to\train.csv `
  --output C:\path\to\train-quality-audit.json
```

The command rejects any file whose SHA-256 or row count differs from the
declared source. The authoritative train-only CSV and JSON receipt remain
outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2-local\banking77-train-audit-20261127-attempt-01\`.
The JSON receipt is 188,883 bytes with SHA-256
`05b335d4a83e7fddfbbda8f6c5c3829f6895fb7856e1d1ff0fc8e1eb99ce878`; the
audit script SHA-256 is
`0c565c04a952384d2a6aa42c40b1b50f9c6cbfc027269d725d4c8e3102f07f5f`.
No model, feature, checkpoint, publisher test row, or sealed-audit item was
loaded; no GPU or cloud compute was used.

## Primary references

- [PolyAI dataset repository at the pinned rights revision](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b)
- [Pinned BANKING77 dataset card](https://huggingface.co/datasets/PolyAI/banking77/tree/1846984189f2bc0ac51c6fc3a600fa9e1714e0c6)
- [Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders](https://aclanthology.org/2020.nlp4convai-1.5/)
