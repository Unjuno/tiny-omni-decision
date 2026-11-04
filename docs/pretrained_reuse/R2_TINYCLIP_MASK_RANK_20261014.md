# Frozen semantic ranking of SAM2 mask candidates

## Result: executed, not adopted

The annotation-free inference process completed with exit0 and
`PASS_FROZEN_SEMANTIC_RANK_EXECUTION_ONLY`. All696 existing NMS mask candidates
from the fixed eight known training clips were encoded and ranked. Separate
CPU scoring found no overall gain in distinct similar-object coverage:

| Fixed top3 condition | Covering-object bbox hits /8 | Distinct similar objects matched /24 | Complete groups /8 |
| --- | ---: | ---: | ---: |
| Original SAM2 presence | 2 | 10 | 1 |
| TinyCLIP cup/glass/bowl similarity | 3 | 9 | 1 |
| TinyCLIP person-text control | 1 | 1 | 0 |

Hit threshold remains bbox IoU>=0.5. Distinct group counts use maximum-cardinality
one-to-one assignment: several masks covering one object cannot count as several
objects. All24 annotated group members have a matching candidate in the complete
NMS bank; this is post-hoc geometry opportunity, not deployable selection.
Three predictions cannot cover the four-object group. The original fixed top3
is retained for this comparison, and this structural limit is explicit.

| Video | Group size | NMS candidates | Presence matches | Semantic matches | Control matches | Semantic cover IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10103 | 2 | 55 | 0 | 0 | 1 | 0.019 |
| 10186 | 3 | 186 | 2 | 3 | 0 | 0.975 |
| 10212 | 3 | 43 | 2 | 1 | 0 | 0.878 |
| 10387 | 4 | 168 | 0 | 1 | 0 | 0.018 |
| 1087 | 3 | 67 | 3 | 0 | 0 | 0.000 |
| 10886 | 3 | 64 | 1 | 2 | 0 | 0.000 |
| 10894 | 3 | 73 | 1 | 2 | 0 | 0.528 |
| 10903 | 3 | 40 | 1 | 0 | 0 | 0.493 |

No threshold, prompt, seed, TopK or clip search followed these results. The
small cover-hit increase does not establish improvement, and group coverage
decreases. Do not adopt this fixed ranking pipeline or claim that semantic
object identification is solved. This negative result does not determine the
capacity ceiling of TinyCLIP, SAM2 or a learned temporal association system.

## Frozen design and dataflow

This is an external throwaway R2 spike following the
[source-role audit](R2_PERCEPTION_SOURCE_CUES_20261014.md), not a product provider.
It reuses the unchanged [SAM2 initialization bank](R1_SAM2_AUTO_INIT_20261014.md):
same eight train IDs, original RGB frame0, all nonempty NMS survivors, unchanged
grid/masks/NMS and original presence top3. No new SAM2 inference was performed.

The already audited TinyCLIP checkpoint/revision
`wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M@07a4b0bc751cb64fecd2b661c048c1dd98d69444`
was loaded locally with strict key checks, no downloads or remote code.
[Published model rights and full parameter audit](R1_SMALL_ALIGNED_MODELS_20261012.md)
remain controlling. Its complete registered count is83,101,185, not the39M name.
SAM2 plus TinyCLIP would count122,063,683 registered parameters before other
modalities/readouts/buffers. This arithmetic is not an integrated runtime or
qualified <=200M Omni product; the two were not resident together in this probe.

Each crop uses a model-generated binary mask and its bounding rectangle from
the saved256×256 mask. Source pixel bounds use floor/ceil normalized scaling;
the cropped mask uses nearest resize and outside-mask RGB128. Published local
CLIPProcessor settings are unchanged. The fixed text strings are:

1. `A photo of a cup, glass or bowl.`
2. `A photo of a person.`

The first describes the already known cup-game domain; it is not the original
QA text and does not identify which cup hides the smaller object. The second
is a predeclared text control using identical image features. Normalized image/
text cosine similarities rank candidates, with original proposal indices
breaking ties. No similarity-softmax or calibrated Decision probability is
reported; Accuracy/NLL/Brier/ECE for the QA task were **not evaluated**.

The inference process only reads pinned model files, the annotation-free media
manifest and complete saved proposal banks. Grounded IDs/boxes/answers/action
labels are absent from neural inputs and inference file reads. A separate CPU
scorer opens the hashed training archive after inference completes, verifies
saved feature products and rankings, then matches geometry against source roles.
The investigator already knew these training annotations and prior scores: this
is not blind and is not a new held-out evaluation. All eight clips remain in the
table; source questions/options were not changed. Source roles remain diagnostic
supervision only, with the [temporal/viewpoint limits](R2_PERCEPTION_SOURCE_CUES_20261014.md)
unchanged. No validation or sealed audit was opened.

## Execution and invariants

- Source commit `da3692796107a2ea254922d35d72272db4a22b5d`.
- Local RTX3080 Laptop, Torch2.6.0+cu124, FP32/SDPA, seed17, deterministic,
  TF32 disabled, batch16;300s wall /4GiB Torch allocation caps and12GiB free
  preflight. No training, optimizer, dependency upgrade or cloud charge.
- Python3.11.9, Transformers5.6.2, NumPy2.4.6, Pillow10.4.0, PyAV18.1.0.
- Timed body7.253056s, peak Torch allocation453,818,880bytes and reservation
  509,607,936bytes. Timing excludes Python/import startup and inherited SAM2
  proposal cost. It includes local TinyCLIP load, checks, crop/decode, inference,
  saves and reload; it is not dense-video or end-to-end Decision latency.
- First image batch repeats bitwise. A fresh model reload reproduces a four-crop
  projection batch bitwise. All saved features/scores are finite; normalized
  embeddings, score products and tie-broken ranks were independently rechecked.
- All parameters remained frozen with no gradients. Persistent model-state SHA
  before/after `700c79f803b3e8b8343c43b47096e00c0322e6c62b5d1c949f567331b030171e`.
  Published weight SHA remains
  `112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587`.
- Crop-coordinate/background and injective-matching CPU self-checks passed.
  Stored NMS indices and original top3 match the old bank exactly; no legacy
  model, media, configuration, checkpoint or source output was overwritten.

Local artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/`:

- `perception-tinyclip-mask-rank-01/`: effective config, execution/result,
  eight NPZs with every candidate embedding, text embedding, score, rank,
  pixel bbox, inherited presence score and exact selected indices.
- `perception-tinyclip-mask-rank-score-01.json`: independent annotated geometry
  matrices/counts, no GPU/model calls.
- `perception_tinyclip_mask_rank_01.py`, `perception_tinyclip_mask_score_01.py`,
  `perception_tinyclip_rank_publish_01.py`: external executed scripts with hashes
  in the adjacent [published JSON](R2_TINYCLIP_MASK_RANK_20261014.json).

Effective config SHA `29516a3ac629dacaf154af9b724b9c822915d0f51f865dff140ecf4230d0c0fd`;
inference result SHA `7153542764973c5d09eef2c1d8be41a4ab655f48779c01d17f5bf15210e4aea4`;
score SHA `1cfbdb6c3cf8ae05051b994fe2b6fbb13b1df5bf21919b7c125228d59b0dbff5`.
No weights or feature arrays enter Git.
Local Ruff passed;953 CPU tests passed and1 skipped in15.44s. Model/dataset
manifests, four catalogs and the example policy audit passed; the complete log
is retained at `perception-tinyclip-mask-rank-01/manifest-validation.log`.

## Next useful gate

Unconditional presence and this single static semantic crop ranking do not
recover the full candidate group reliably. Further prompt/threshold tuning on
these eight outcomes is not the next experiment. A learned proposal filter or
input-only temporal association requires a fixed training/validation protocol,
variable group sizes, source supervision isolated from inference, and preserved
raw-candidate opportunity/identity diagnostics. A scalar static cup score still
cannot select the hidden item's associated cover or resolve game-time/actor
coordinates. Establish those interfaces and task-matched evidence before adding
capacity, longer training, shared-Lens/merging or compression. The complete Omni
and useful simultaneous-input quality goals remain active and unmet.
