# FSD50K training-source feasibility

Official source: https://zenodo.org/records/4060432 (DOI10.5281/zenodo.4060432),
linked by the authors' https://fsannotator.upf.edu/fsd/release/FSD50K/.
Retrieved document, metadata and ground-truth ZIPs only, verified all three
against published MD5 and saved SHA-256. Eval files remain compressed and unread;
only dev metadata, dev CSV, vocabulary and documentation were opened. No audio
download, model fitting or evaluation occurred.

## Rights and coverage

Dataset-level CC BY4.0 is additional to each clip's license. Dev metadata has
20,017 CC BY3.0,14,959 CC0,4,616 CC BY-NC3.0 and1,374 Sampling+ clips. For this
product research, admit only explicit CC0/CC BY candidates with attribution;
exclude NC and Sampling+ rather than treating the dataset license as overriding
clip rights. Official train contains31,566 eligible clips.

| Positive label | Eligible train clips | Distinct uploaders |
| --- | ---: | ---: |
| Bell | 598 | 155 |
| Clapping | 305 | 88 |
| Telephone | 425 | 192 |
| Walk_and_footsteps | 277 | 150 |
| Musical_instrument | 10431 | 404 |
| Knock | 211 | 52 |
| Door | 817 | 287 |
| Water_tap_and_faucet | 189 | 114 |

These are label-presence counts before waveform integrity, duplication or
cross-source checks; they are not an approved frozen training corpus. Uploader
identity is a grouping proxy, not proof of independent recordings. Retain clip
IDs, uploader, license and attribution in any later manifest.

## Critical label constraint

Official documentation says dev annotations can be incomplete and labels are
propagated upward through the ontology. Therefore, an absent class cannot be
silently converted into a verified negative. A naive12-label binary conversion
would introduce false negatives. Any later bounded training design must retain
known-positive versus unknown masks, justify negatives independently, and check
the distinction between FSD clip presence and STARSS prominent-event annotation.
Do not add, remove or relabel existing validation examples based on these counts.

This is a viable source of additional positive coverage, not yet a ready-made
Decision training set. Next gates: freeze training-only IDs without outcome-based
sampling, obtain bounded individual media with archive integrity checks, verify
duplicates and provenance, then preregister a masked-loss comparison and unchanged
legacy validation. Dataset-wide download would need about18GB for dev audio and
does not fit current disk capacity; selective verified acquisition is required.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/fsd50k-metadata-audit-01/`.
Companion JSON preserves archive hashes, license counts and positive coverage.
The initial API request completed after a long connection wait; it was not
restarted. Metadata downloads used the existing process-local IPv4 HTTPS helper,
with TLS validation intact. No existing worktree artifacts or runs changed.
