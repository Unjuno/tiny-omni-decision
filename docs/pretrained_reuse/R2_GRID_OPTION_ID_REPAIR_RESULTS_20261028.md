# GRID quartet option ID repair: result

The original public candidate IDs were hashes of short GRID transcript codes.
The finite code vocabulary can be enumerated, so those identifiers may expose
content. This repair retains the original catalog and replaces the IDs in a
separate derived catalog with per-quartet `option-0` through `option-3`.
Targets are remapped to the same candidate positions. Quartet metadata,
including codes and speaker IDs, remains private and must not be model input.

| Split | Quartets | Query roles | Derived JSONL SHA-256 |
| --- | ---: | ---: | --- |
| Train | 23,991 | 95,964 | `f9939f17f7cf021f06464f9c652257269a8f36c793ef238f82007d04210d52b7` |
| Development | 7,999 | 31,996 | `96180ca1cf07fed2a61c3c8ed8e4e8a89eb7b686692f9487507bd2fb248386e9` |

Every original and derived row was independently compared: speaker, quartet,
anchor, ordered codes, query roles, and target candidate positions agree.
Each quartet has the four local IDs in order. This is a metadata repair; no
media were processed and no model quality was evaluated.

Command: `python -m scripts.repair_grid_option_ids --output-dir C:/CodexArtifacts/pretrained-reuse-r2/grid-local-option-ids-20261028-attempt-01`.
The generated summary has SHA-256
`98cd6bdc16922e41b4f8dfdfab07ddffa70e88fcd39f4565566789e2b5f317bb`.
Focused tests and Ruff passed. These identifiers are opaque only if the private
catalog fields are excluded from model inputs, as required by the input adapter.
