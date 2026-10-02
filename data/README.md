# Data workspace

Datasets are not committed to this repository.

Expected local layout:

```text
data/
  raw/          downloaded source material
  processed/    normalized decision samples
  manifests/    generated frozen provenance manifests
```

Normalized processed records use the versioned `DecisionExample` schema in
`src/tiny_omni_decision/schema.py`. Every record carries its immutable source revision and
per-record license/provenance facts. Binary media stays outside JSONL; each media entry is
a path or pinned source reference.

Training/evaluation candidate manifests are kept separate under `manifests/`. Never place
evaluation sources in a training output. Use `dataset-check-splits` before freezing a pair.
The example below is illustrative only:

```json
{
  "id": "source:sample-id",
  "modality": "audio",
  "state": "...",
  "media": [{"kind": "audio", "uri": "hf-dataset://org/data@<sha>/clip.wav"}],
  "question": "...",
  "options": ["A", "B", "C"],
  "target": "B",
  "source": "dataset-id",
  "source_revision": "<40-character commit SHA>",
  "source_record_id": "...",
  "split": "train",
  "provenance": {
    "license": "CC0-1.0",
    "commercial_use": true,
    "derivative_model_training_allowed": true,
    "redistribution_allowed": true,
    "media_redistribution_allowed": true,
    "trust_status": "trusted"
  }
}
```
