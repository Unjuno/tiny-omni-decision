# Fixed training acquisition:128 clips

Scope:16 clips for each of the eight smoke events, from official FSD50K train,
CC0/CC BY only, ordered by SHA256(`17:<clip_id>`), excluding already-selected
clip and uploader IDs. The selected manifest contains128 distinct clip IDs and
128 distinct uploaders. No model outcomes, media duration or validation results
were used to select individual records. Missing labels remain unknown.

Selection SHA256:
`ca017b24cbe16975c984335355281c6e4f0e2614eca23a27c49d723fddff61d6`.
Local directory: `C:/CodexArtifacts/pretrained-reuse-r2/fsd50k-training-128-01/`.
The acquisition was launched; this document does not claim it has finished.
Authoritative completion requires terminal result.json PASS,128 verified files,
decode/integrity checks and later feature report. Progress.json alone is partial.

```powershell
python -u -m scripts.fetch_fsd50k_training_smoke --per-event 16 --output C:/CodexArtifacts/pretrained-reuse-r2/fsd50k-training-128-01
```

Preflight caps planned compressed transfer plus headers/index at256MiB and
requires2GiB disk reserve. Oversized members or unavailable distinct uploaders
fail closed without replacement. Existing eight-clip outputs remain untouched.
The same ZIP CRC/local-header checks apply. This acquisition does not authorize
assuming cross-source near-duplicate absence or training unknowns as negatives.
No additional learning has started. The feature extractor now accepts explicit
input/output directories so new results do not replace smoke outputs.
