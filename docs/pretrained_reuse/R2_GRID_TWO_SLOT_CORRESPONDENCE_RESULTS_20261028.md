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
