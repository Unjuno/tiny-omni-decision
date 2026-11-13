# TinyCLIP fixed screen reference

Pinned existing TinyCLIP83,101,185parameters; immutable audited weights reused.
No training, prompt search, calibration or checkpoint selection. Same723 observed
validation regions and8category labels. Fixed prompt: `A screenshot of a user
interface element: {category}.` Official local CLIP processor; exact source/crop
hashes checked. No text annotations supplied as inputs.

Accuracy .408022, balanced Accuracy .122042, NLL1.832211, Brier .779894,
ECE .127905. One-position crop rotation: .369295/.099853/1.905177/.808189/.094314.
Rotation can keep crops within the same screen; it is a weak mismatch control,
not evidence of independent semantic reliance. No significance claim.

Macro recall near1/8 makes this an inadequate category recognizer despite40.8%
sample Accuracy. Expanded DINOv2 head was41.36%/.27314 balanced Accuracy; models
have different pretraining/task training/preprocessing, so no backbone ranking.
Training-prior NLL1.6955 remains better. No adoption or prompt retries.

TinyCLIP plus existing Whisper and22.7M decision text is143.6M before other paths;
adding existing VideoPrism would exceed200M. This is reference-only arithmetic,
not a validated unified model. Temporal video/fusion and high quality remain unmet.

Output: C:/CodexArtifacts/pretrained-reuse-r2/tinyclip-screen-zero-shot-01.
Protocol/predictions/rows and hashes saved; adjacent JSON contains resource and
class metrics. Command `python -m scripts.evaluate_tinyclip_screen`; Ruff passed.
Existing weights and all legacy data unchanged; no sealed audit accessed.
