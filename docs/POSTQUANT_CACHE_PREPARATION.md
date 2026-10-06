# Teacher-cache preparation: next worker command

## Status and boundary

Builds on numerical-core commit `10776ab6b031d027cb7d6d78910541242bc7ff5c`.
This change implements export identity checking and offline cache packaging from
Task 4 of the post-quantization experiment plan. It does not reimplement the
existing numerical policy/loss/ternary core or enable `postquant_trainer`.

No model, tokenizer, media decoder, GPU, optimizer, API or subprocess is used by
this script. It reads only the job and its two explicit JSONL sources. Default
operation is read-only; `--write` packages an already-computed export, not a new
Teacher inference. Do not stop or change the ongoing clean-dev-v2 run to use it.
The final four-modality 95% requirement is unchanged.

## Commands

Run from the repository root, on Windows or Linux:

```shell
python scripts/prepare_postquant_cache.py --job /absolute/path/job.json
python scripts/prepare_postquant_cache.py --job /absolute/path/job.json --output /separate/existing/parent/new-cache --write
python scripts/prepare_postquant_cache.py --job /absolute/path/job.json --verify-package /separate/existing/parent/new-cache
```

On Windows, use quoted paths such as `"C:/CodexArtifacts/cache-job/job.json"`.
Input paths inside the job are relative to its directory; either slash style is
accepted. Output parents must exist, and the output must be new and outside the
frozen source directories. There is no overwrite, execute, resume, auto-select
best, automatic inference, or all-arm loop flag. Abbreviated flags are rejected.

Copy `configs/recovery/teacher_cache_import.example.json` outside Git and supply
actual producer-recorded values. Its default master state and placeholder hashes
intentionally block use. Do not fill in a false reload-verification status to
bypass that guard. `master_state` is an exporter assertion, not proof that this
script reloaded the model; independently retain the real freeze/reload evidence.

## Inputs and identities

`inputs` is a frozen expected-input JSONL manifest. Each row is one complete
identity; `predictions` is an independently saved raw-logit export whose rows are
`{"identity": <same complete identity>, "option_logits": [<raw scores>]}`.
Both files have explicit SHA-256 digests and exactly `expected_rows` rows, in the
same order. IDs are unique. Rows are bounded to 1 MiB; the job is also bounded to
1 MiB. Blank, malformed, duplicate-key, nonfinite and surplus rows fail closed.
The row-count cap is an engineering bound, not a measured optimum.

| Identity field | Meaning / 意味 | Unit | Definition / conditions | Type |
|---|---|---|---|---|
| `purpose` | 用途 | none | Exactly `training` or `development`; must agree in job and both sources | String enum |
| `sample_id` | 入力例の識別子 | none | Nonempty, unique within an export | String |
| `example_sha256` | 元の入力メタデータの指紋 | none | SHA-256 of `canonical_json(example.model_dump(mode="json"))`; includes ordered options, question and source provenance | Hex string |
| `input_sha256` | 実際に処理した入力の指紋 | none | Captured by the pinned exporter from actual processor outputs, under the versioned preprocessing recipe; never fabricated at import | Hex string |
| `option_texts` | 順序付きの選択肢本文 | none | 2..62 unique nonempty strings | String list |
| `option_labels` | 回答ラベル | none | First labels of case-sensitive `A-Z`, `a-z`, `0-9` | String list |
| `option_token_ids` | 実際の回答token ID | 1 | Distinct nonnegative integers captured for the rendered prompt | Integer vector |
| `target` | 正解index | 1 | Integer indexing the ordered options, not a teacher prediction | Integer scalar |
| `media_sha256s` | メディア指紋 | none | Ordered SHA-256 list; empty for media-free Text | Hex-string list |
| `teacher` | 教師と前処理の識別 | none | Exact four fields listed below | Object |
| `option_logits` | 保存済みの生スコア | 1 | One finite number per active option; not probabilities relabeled as logits | Real vector |

`teacher` has exactly `master_sha256`, `model_revision`, `processor_revision`,
and `preprocessing_sha256`. Revisions are immutable 40-character Git hashes;
other digests have 64 lowercase hexadecimal characters. Record the preprocessing
implementation, decoder/backend, sampling/frame policy and tensor-fingerprint
recipe under the preprocessing identity. The importer cannot verify a fabricated
fingerprint or establish that the exporter actually used the claimed model.

`canonical_json` sorts keys, uses compact ASCII-escaped UTF-8 JSON and forbids
nonfinite JSON numbers. All logits and indices here are dimensionless; no loss
or probability calculation is performed. Logits are retained without conversion
to probabilities or low-precision storage. The cache key hashes the complete
identity; the final file digest additionally protects the actual logit values.

Use `make_teacher_cache_record(identity, logits)` and
`validate_teacher_cache_record(record, expected)` from `teacher_cache.py` in
producer/consumer integration. A changed target, option order/text/token ID,
model, processor, preprocessing, actual input or media identity must invalidate
the cache. In particular, a changed Video window requires a new input identity.

**Historical `training.output_record` alone is insufficient.** It records raw
option logits but not the complete input/option-token identity. This importer
rejects those legacy rows rather than guessing provenance. A validation export
cannot become a training cache by renaming a file. Future export integration
must capture the complete identity during Teacher evaluation on the frozen
training examples. That inference is not implemented or initiated by this change.
The full dataset split/rights checks, actual model/media hashing and actual
Teacher freeze/reload checks remain mandatory upstream; matching metadata is not
a substitute for them.

## Output and interrupted writes

`--write` validates everything first and then claims a new directory with an
exclusive mkdir. It streams a second validated pass, rechecks all source hashes,
and writes `cache.jsonl`, `cache-manifest.json`, and finally `COMPLETE`.
The marker contains the manifest-file SHA-256; the manifest binds cache content,
ordered IDs, row count, source hashes, purpose and teacher identity.

A failed/interrupted write retains an incomplete directory for diagnosis and
never publishes a valid completion marker. Use a different new output directory
after resolving the cause. No existing directory or input is deleted or replaced.
Consumers must not treat the mere existence of `cache.jsonl` as completion.
`--verify-package` revalidates the frozen source export and the entire package;
a missing marker, tampered file or changed source is an error, not an auto-repair.

The existing launch manifest may point its `teacher_cache` artifact at this
package and use its existing `postquant-tree-v1` digest. The launcher still only
checks artifact identity; the future numerical trainer must call the row
validator against the inputs it actually processes. No runnable backend flag is
added, and no model smoke or training experiment is marked complete.

Local paths containing audit/sealed/heldout components, traversal, URL/UNC source
paths, symlinks and Windows reparse points are rejected before source reads.
This is an accidental-misuse guard, not an adversarial filesystem sandbox or a
way to recognize a secretly renamed audit dataset. Retain immutable input files
throughout preparation and verification.

## Verification and sources

Local tests use synthetic JSON only on Linux/Python 3.13.5; Python 3.11 syntax is
also checked. This is not the project's target-runtime verification by itself.
The added CPU-only Linux/Windows CI runs the new tests on Python 3.11, without
PyTorch installation or model data. Consult the actual CI result, not this
workflow definition, before claiming target-platform success.

- Python 3.11 `argparse`: https://docs.python.org/3.11/library/argparse.html
- Python 3.11 file creation/path semantics: https://docs.python.org/3.11/library/pathlib.html
- Python 3.11 `os.replace` may overwrite: https://docs.python.org/3.11/library/os.html

These sources informed the strict flags, exclusive creation and completion
protocol. We do not claim crash-atomic publication of the entire directory,
scientific quality, correct model inference, or speed from metadata checks.
