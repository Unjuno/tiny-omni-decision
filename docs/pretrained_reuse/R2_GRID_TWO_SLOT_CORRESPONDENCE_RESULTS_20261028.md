# GRID two-slot correspondence catalog: metadata result

The [fixed two-slot procedure](R2_GRID_TWO_SLOT_CORRESPONDENCE_PROTOCOL_20261028.md)
completed without media access. The separate one-slot attempt was infeasible;
this is a different, explicitly two-slot candidate policy.

| Development partition | Speaker labels | Source alignment codes | Unique quartets | Query roles | Anchors lacking 3 neighbors |
|---|---:|---:|---:|---:|---:|
| Train | 24 | 23,992 | 23,991 | 95,964 | 0 |
| Validation | 8 | 7,999 | 7,999 | 31,996 | 0 |

An independent readback checked each quartet's distinct IDs, four different
correct roles, candidate/role consistency, exact two-slot anchor distances and
unique quartet IDs: zero violations. All source codes appear in at least one
quartet. Each quartet's four roles share the same ordered candidate set, so a
deterministic scorer given only those videos is limited to 25% on those roles.
The many roles reuse speakers and media and must not be treated as independent
observations. Audio-only, swapped/missing media and temporal controls remain
required. This is a frozen metadata catalog, not an executed quality result;
media existence, rights/attribution, synchronization and physical speaker
independence still need validation.

Output:
`C:/CodexArtifacts/pretrained-reuse-r2/grid-two-slot-correspondence-20261028-attempt-01/`;
`summary.json` SHA-256
`b581bb4a70229f8dbb104e3f97e7b3487ac12c5c80af40aaee52bf738a31c876`.
The receipt pins both partition, script, protocol and output hashes. Three CPU
tests and Ruff passed. No media, model prediction or sealed audit was read.

## All-role difficulty accounting

The two-slot guarantee applies to the **construction anchor** and each of its
three selected candidates. The four-role balancing also makes each other
candidate an audio query; its distance to the remaining videos can be larger.
Counting every ordered query-to-other-candidate pair gives:

| Partition | Distance 2 | Distance 3 | Distance 4 |
|---|---:|---:|---:|
| Train | 166,144 | 83,248 | 38,500 |
| Validation | 55,562 | 27,758 | 12,668 |

Only 25,314 of 95,964 train query roles and 8,457 of 31,996 validation query
roles have all three alternatives at distance 2. No role or candidate was
removed. These are code-slot distances, not measured phonetic similarity or
model difficulty. Report metrics by actual query distance if the task is run;
do not call every query a two-slot-only comparison.

The reproducible metadata-only audit is
`C:/CodexArtifacts/pretrained-reuse-r2/grid-quartet-distances-20261028-attempt-03.json`,
SHA-256 `02cb30293315d667ac70a233cbc95a83aaaee57fb900b27e4222ee49229a1cc1`.
Attempts 01 and 02 are preserved; the final script differs only by formatting
fixes. Two CPU tests and Ruff passed; no media or predictions were opened.

## Reuse and cluster accounting

The train quartets derive from 24 speaker labels, with 998–1,000 quartets per
speaker and 23,992 distinct audio codes; each audio code appears in 1–10 query
roles. The validation quartets derive from only eight speaker labels, with
999–1,000 quartets each and 7,999 distinct audio codes; an audio code appears
in 1–11 roles. These are code-level counts before media verification. They
confirm that tens of thousands of roles are highly dependent and that future
uncertainty must resample at least by speaker, not individual questions. Eight
validation speaker clusters provide limited precision even if media and person
isolation are later established.

Metadata-only receipt:
`C:/CodexArtifacts/pretrained-reuse-r2/grid-quartet-reuse-20261028-attempt-01.json`,
SHA-256 `c68cac88d6c9251af113a504466ef4fb1f1a2bf63fad5539b684d4f43c6a6362`.
Two focused tests and Ruff passed.

## Model-input boundary

`tiny_omni_decision.grid_correspondence_input.build_input` maps a quartet and
verified caller-supplied media to an immutable public input containing only
audio bytes, four ordered video byte strings and opaque option IDs. It returns
the target separately for evaluation. It rejects malformed target mappings,
missing/extra media, empty media and identical video byte strings. The boundary
does not itself verify source hashes, identity, synchronization, consent or
semantic correctness; those remain caller gates. A synthetic integration test
connects it to the existing `NativeAVCorrespondence` JSON interface and checks
four probabilities sum to one. Three CPU tests and Ruff passed. No GRID audio,
combined model inference or task accuracy was measured in this check.
