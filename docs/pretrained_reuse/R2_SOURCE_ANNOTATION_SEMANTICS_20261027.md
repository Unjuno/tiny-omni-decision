# Perception source-label semantics audit

## Finding

The pinned [upstream annotation specification](https://github.com/google-deepmind/perception_test/blob/3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf/README.md#perception-test-annotations)
uses `parent_objects` for objects **related** to the sound segment and allows
`-1` for an object without an annotation. Its `is_visible` example distinguishes
a visible object from an occluded impact point. The source-candidate builder
instead labels every `is_visible == 0` example `none_visible`, regardless of
whether a related object has a visible candidate box. Its instruction asks for
all visual candidates that caused the sound, or `none_visible` if none is visible.
That wording is not established by the upstream field definitions.

## Pinned train-source reconstruction

The audit rebuilt every candidate identity, sampled frame box and ordering
from the original official-train annotations and the pinned builder. All
2,593 retained train-source records matched their saved candidate list; all
1,901 visible-event target lists matched reconstructed related-parent IDs.
This is an implementation correspondence check, not human label validation.

| Retained record property | Count |
|---|---:|
| `none_visible` records | 692 |
| Those with an upstream `-1` parent | 647 |
| Those with at least one known related parent ID | 47 |
| Those where a known related parent is a visible candidate | 30 |
| Related-parent candidate instances in those 30 | 32 |
| Visible-event records with multiple related-parent candidates | 1,102 |

The `-1` and known-parent categories can overlap; do not add them. A candidate
box in the 30 records proves a **related tracked object** was presented, not
that its sound-producing contact point was visible or that it independently
caused the sound. Likewise, `-1` means an object is unannotated, not that no
object exists. We have not inspected the corresponding videos or established
whether the original annotations are correct.

These facts place a **semantic admission gate** on this custom task: do not
interpret `none_visible` as proof that no related object is visible, or the
parent list as a fully verified causal set. Earlier metrics remain historical
measurements of the derived labels and are not withdrawn, but cannot validate
causal multimodal source identification. Before further task-specific training,
define whether the desired target is event visibility, related annotated objects,
or physical causal sources; independently check a fixed representative sample
and distinguish unresolved/unannotated cases. Do not modify the frozen corpus or
relabel hard examples to improve scores. Also retain independent source grouping
and real multimodal necessity as separate unresolved gates.

No media, validation, sealed audit, or new model was accessed. Output:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-source-annotation-semantics-20261027-attempt-01/result.json`,
SHA-256 `6f599092a121dbbf186dcd646a5b57c3ad234084efad905d3b75a1e4a87d96e1`.
The receipt pins the upstream archive, train corpus, builder and audit script.
