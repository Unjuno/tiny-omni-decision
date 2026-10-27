# R2 GRID train partition speaker-to-audio mapping audit (2026-10-28)

Status: **BLOCKED for GRID training.** A fresh hash-gated, directory-only
comparison of the frozen 24-speaker train split against the official GRID
`audio_25k.zip` reports 14,997 alignment codes without an audio member under
the same speaker folder. Fifteen of 24 train speakers are affected. Across the
full train+validation split, 18,997 of 31,991 codes fail same-folder matching.
No speaker remapping was performed.

## Method and fixed inputs

Re-ran the existing fail-closed audit:

```powershell
python -m scripts.audit_grid_audio_folder_mapping `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\grid-train-speaker-source-audit-20261028-attempt-01
```

The output was written outside the repository. The audit verifies the frozen
split hashes and full official audio archive SHA-256, then inspects archive
member names and compares them with the pinned split. It does not extract,
decode, listen to, or infer identities from media. The expected blocked status
exited with code 2 after saving the audit JSON; elapsed time was 4.35 seconds.

| Input | SHA-256 |
| --- | --- |
| Train split | `f6f045931aee53e576fca22efda0bc888a983aed685a30edc38b2918ff156817` |
| Validation split | `c1e77098327841fadc283bb14813c38de94e07fd16728d9cc3602bc261961e11` |
| Official `audio_25k.zip` | `d89331b1db49770ca0d0c676e75b6e35725ecb48cf35915e8482285af2594134` |
| Audit script | `ac106439806b13ae6414a58c0774d677346bf92396c133cbc87dfea1f3e7d768` |
| Result JSON | `e0606f0cfc263c6102766a31c10337fe09dfc6b7f3d6c43efa6dd968fc2b1e84` |

## Findings

- 32 alignment speakers, 31,991 alignment codes, and 34 audio folders were
  present in the checked indices.
- The 24 train speakers are fixed by the split receipt. Fifteen had one-to-one
  code sets absent from the same-named audio folder; 14,997 train codes were
  unmatched. Nine train speakers passed same-folder code matching.
- The eight validation speakers contribute the remaining 4,000 unmatched
  codes, consistent with the separate direct-bundle audit: four validation
  labels had zero same-label overlap.
- The audit reports unique best overlaps for several differently named
  folders. This is not proof that the speaker folders should be permuted. No
  labels or split records were changed.
- `media_bytes_read=false` and
  `speaker_identity_inferred_or_remapped=false` in the saved result.

## Consequence

Do not train on this frozen GRID split and do not “repair” it by mapping a
speaker to the folder with the highest filename overlap. GRID training requires
a separately verified source-to-record mapping and an independently rebuilt
split before any fitting; preserve this split and audit as a failed integrity
receipt. The corrected eight-speaker direct-source validation probe remains a
narrow development result and does not validate the train partition or prove
cross-corpus speaker independence.

The exact machine-readable result is retained at
`C:/CodexArtifacts/pretrained-reuse-r2/grid-train-speaker-source-audit-20261028-attempt-01/mapping-audit.json`.
