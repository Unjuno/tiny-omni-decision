# Fixed screen-context comparison protocol

Question: does adding frozen whole-screen context improve the rejected crop-only
screen-region classification? Same48 screens, app split,1740 admitted regions,
8labels, frozen22,056,576-parameter DINOv2. Existing checkpoints remain immutable.
No OCR, annotation text, category names, filenames or app IDs become model inputs.

Extraction completed:48x768 full-screen CLS+patch-mean features, bicubic224x224,
no center crop. Exact repeat on first8 and exact disk reload; source image/weight
hashes verified. Adjacent JSON records resources/hashes. This proves extraction,
not useful screen understanding. Resizing may lose small UI details.

Next fixed comparison: concatenated1536 inputs in both arms: crop+zero versus
crop+screen. Identical Linear1536->8 initialization, seed17, AdamW .001/.01,
batch64,30epochs, train-only normalization (constant dimensions scale1), fixed
example order. Select earliest minimum validation NLL includingepoch0 exactly as
previous pilot; preserve selected/final weights and every epoch. No new tuning.
Report Accuracy/balanced Accuracy/NLL/Brier/ECE per class and app. Both arms have
12,296 head parameters; encoder remains frozen. The zero arm is a matched new
control, not a replacement for the historical768-input head.

On the selected context head also evaluate deterministic different-screen context
within validation (fixed cyclic shift of screen IDs), keeping crop and labels fixed.
This diagnoses reliance, not an independent test. Require improvement over the
matched crop control and report class/metric trade-offs; no product adoption from
four observed validation apps. Full-screen-only and spatial/bbox interventions
are not included in this first bounded comparison. No audit split is opened.

Artifacts: C:/CodexArtifacts/pretrained-reuse-r2/groundcua-fullscreen-features-01.
Exact executed extraction script is retained there; repo edit after execution only
wraps a long string for lint. No comparison training has started in this stage.
