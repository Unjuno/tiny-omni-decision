# Action Verb Corpus license recheck — 2026-10-26

## Decision

Keep Action Verb Corpus (AVC) at **REVIEW**. The live Zenodo API still presents a license conflict within the same record: `metadata.license.id` is `cc-by-4.0`, while its `metadata.version` text explicitly says CC BY-NC-SA 4.0 and non-commercial use. Do not choose the more permissive field without clarification from the rights holder. No corpus files, annotations, or media were downloaded or opened in this recheck.

## Evidence

The official [Zenodo record](https://zenodo.org/records/5140014) was queried on 2026-10-26. Its API response reports DOI `10.5281/zenodo.5140014`, record id `5140014`, `updated=2024-07-19T06:25:37Z`, and `license.id=cc-by-4.0`. The same response's `version` field contains the conflicting CC BY-NC-SA 4.0/non-commercial statement. The [OFAI corpus page](https://www.ofai.at/resources/avc) also labels the corpus CC BY 4.0, but that does not resolve the contradictory version-level notice.

The metadata-only snapshot is stored outside Git at:

`C:\CodexArtifacts\pretrained-reuse-r2\action-verb-corpus-license-recheck-20261026-attempt-01\zenodo-record.json`

SHA-256: `1403568e872a19a04f8021a2978cb98ac7ebea2ef8fbb8cc6dac819854200252`.

## Disposition

- Rights for intended product-aligned training: **unresolved**.
- Training/evaluation admission: **REVIEW; do not use**.
- Payload access: **metadata only**.

Reconsider only after the corpus rights holder clarifies which license governs the data and confirms the intended use. This does not change the earlier source screen or any model/data artifact.
