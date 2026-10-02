# Data workspace

Datasets are not committed to this repository.

Expected local layout:

```text
data/
  raw/          downloaded source material
  processed/    normalized decision samples
  manifests/    generated frozen provenance manifests
```

Every processed record should be traceable to a source manifest and should normalize toward:

```json
{
  "id": "source:sample-id",
  "modality": "audio",
  "state": "...",
  "media": "relative-or-resolved-reference",
  "question": "...",
  "options": ["A", "B", "C"],
  "target": 1,
  "source": "dataset-id",
  "source_revision": "...",
  "license": "..."
}
```

Training and evaluation sources must be separated explicitly to reduce contamination.
