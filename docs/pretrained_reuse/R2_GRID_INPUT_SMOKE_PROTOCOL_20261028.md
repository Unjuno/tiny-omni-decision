# GRID original real-input smoke

Use the [Zenodo GRID v1.0 record](https://zenodo.org/records/3625687), whose
API declares CC-BY-4.0, and the verified original alignment archive. This is a
technical smoke on original GRID talker `s1`, not a product evaluation, model
selection or evidence of unseen-person generalization. Speaker ID matching
across GRID and Lombard is unknown. Do not train, tune thresholds, compare to
the observed Lombard validation or access a sealed audit.

Before loading media, sort `s1`'s six-character alignment codes and take the
first 16 whose codes do not occur in the hash-pinned Lombard training manifest.
Do not inspect model outputs when choosing them. Check that all six alignment
words agree with the code, that both original audio and original video exist,
and that files are safely decoded and matched by speaker and code. Attribute
the original authors. Record all source archive hashes and the fixed manifest.
Use a separate new external output path; preserve existing model and data.

Acquire the Zenodo audio archive and only `s1.zip`, each in separate new
outputs. Validate complete publisher MD5 and locally compute SHA-256 before
opening either archive. Enforce bounded disk reserve and archive size, never
disable TLS verification or resume from unverified partial bytes. If retrieval
fails, retain the attempt and report the transport error; do not change source
or silently relax integrity. Other speakers and final evaluations remain
untouched. Preliminary audio/video encoder smoke must check shapes, finite
values and runtime without making a quality claim.

The manifest was fixed from the original alignment archive before reading
media: `C:/CodexArtifacts/pretrained-reuse-r2/grid-s1-smoke-manifest-20261028-attempt-01.json`,
SHA-256 `06155424e4a2c88a38afa5962c99af9f551a2a18cf8656c4c8e8125096e3a06b`.
All 16 selected six-word alignments matched their decoded codes and none occurs
in the existing Lombard train manifest. This does not verify media presence or
speaker independence. `scripts/prepare_grid_smoke_manifest.py` records the
pinned inputs and selection rule; no model was run.
