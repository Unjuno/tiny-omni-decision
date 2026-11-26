# ViLT fixed24-epoch learning curve

Fresh seed17; the entire first8-epoch history reproduces exactly.
No resume or re-extraction. Pinned cache/config/history hashes verified.
8424 train images/33696 questions;512 observed validation images/2048 questions.
All four10-option taxonomies remain intact; no sealed audit or reporting set read.

| Run | Task | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| vilt24 | all | 0.844727 | 0.399479 | 0.207642 | 0.022420 |
| vilt24 | color | 1.000000 | 0.004593 | 0.001631 | 0.004056 |
| vilt24 | texture | 0.816406 | 0.528039 | 0.261811 | 0.047368 |
| vilt24 | count | 0.568359 | 1.035132 | 0.554031 | 0.052032 |
| vilt24 | shape | 0.994141 | 0.030150 | 0.013096 | 0.006308 |
| vilt8 | all | 0.811035 | 0.471817 | 0.238904 | 0.018794 |
| vilt8 | color | 0.996094 | 0.010858 | 0.004123 | 0.009253 |
| vilt8 | texture | 0.771484 | 0.663302 | 0.323659 | 0.040122 |
| vilt8 | count | 0.488281 | 1.169200 | 0.608363 | 0.038564 |
| vilt8 | shape | 0.988281 | 0.043908 | 0.019473 | 0.009545 |
| dino8 | all | 0.874023 | 0.289319 | 0.157210 | 0.011022 |
| dino8 | color | 0.978516 | 0.059637 | 0.029487 | 0.016945 |
| dino8 | texture | 0.921875 | 0.201637 | 0.109964 | 0.031818 |
| dino8 | count | 0.597656 | 0.889974 | 0.486729 | 0.042458 |
| dino8 | shape | 0.998047 | 0.006029 | 0.002658 | 0.003344 |

ViLT24 minus vilt8: +3.369pp, 95% interval[+2.051,+4.688]pp.

ViLT24 minus dino8: -2.930pp, 95% interval[-4.736,-1.172]pp.

512-image paired percentile bootstrap,10000 draws,seed17; selection bias remains.

| Epoch | Train Acc | Val Acc | Train NLL | Val NLL |
|---:|---:|---:|---:|---:|
| 1 | 0.730057 | 0.725586 | 0.735882 | 0.733183 |
| 2 | 0.763206 | 0.761719 | 0.626032 | 0.623614 |
| 3 | 0.780894 | 0.776367 | 0.561257 | 0.569897 |
| 4 | 0.792972 | 0.790039 | 0.533196 | 0.549524 |
| 5 | 0.798463 | 0.790039 | 0.511047 | 0.512315 |
| 6 | 0.808108 | 0.789551 | 0.495448 | 0.519162 |
| 7 | 0.811788 | 0.798828 | 0.474238 | 0.501768 |
| 8 | 0.817872 | 0.811035 | 0.459255 | 0.471817 |
| 9 | 0.826715 | 0.819824 | 0.437841 | 0.452218 |
| 10 | 0.823688 | 0.817871 | 0.441877 | 0.458432 |
| 11 | 0.833007 | 0.823730 | 0.422092 | 0.449389 |
| 12 | 0.825617 | 0.816895 | 0.437862 | 0.462662 |
| 13 | 0.837191 | 0.827148 | 0.409911 | 0.434616 |
| 14 | 0.837102 | 0.819336 | 0.411671 | 0.448782 |
| 15 | 0.844344 | 0.838379 | 0.392666 | 0.414429 |
| 16 | 0.843987 | 0.828613 | 0.388785 | 0.413669 |
| 17 | 0.842533 | 0.836426 | 0.394334 | 0.431591 |
| 18 | 0.844789 | 0.826172 | 0.388141 | 0.416432 |
| 19 | 0.846154 | 0.840820 | 0.381916 | 0.411032 |
| 20 | 0.842622 | 0.828613 | 0.395473 | 0.424696 |
| 21 | 0.839922 | 0.827637 | 0.399637 | 0.449018 |
| 22 | 0.856096 | 0.840332 | 0.364141 | 0.406385 |
| 23 | 0.852386 | 0.844727 | 0.370708 | 0.399479 |
| 24 | 0.850309 | 0.838379 | 0.379516 | 0.410369 |

## Compute and artifacts

- Best epoch23, selected775008 encounters; whole run808704 (unique33696).
- Rotated-image Accuracy0.099609.
- Cache load0.181s, train/eval 72.767s, total74.132s, peak allocated VRAM2093685248 bytes; cost$0.
- Parameter total remains135,126,073; only147,712 readout parameters trained.
- Best/final files separate; reload exact. No old artifacts changed.
- Directory `C:/CodexArtifacts/pretrained-reuse-r2/vilt-image-curve-01`.
- Original8-epoch config representation string was a DINO scaffold typo;
  executed code/cache used ViLT pooler_output. New config corrects the label;
  original metadata is retained. No result is silently replaced.
- These are narrow taxonomy results, not general image or all-modality95%.

## Interpretation and next priority

Extra optimization improves ViLT Accuracy/NLL/Brier on this observed validation
set, but ECE rises0.01879→0.02242. Selected23 is retained separately from final24.
ViLT remains2.93pp below the much smaller DINO candidate; do not replace it.
Texture81.64% and count56.84% remain well below95% even though color reaches100%.
A single taxonomy's100% must not be presented as image or omni success.

At23 train85.24%/NLL0.3707 versus validation84.47%/NLL0.3995; at24 both worsen.
The curve includes optimization fluctuation and further late improvement; it
does not establish a clean plateau or a model-family capacity limit. Nevertheless
the relative quality/size result does not justify automatically extending ViLT.
Prioritize measuring the stronger39.59M DINO/bilinear candidate's matched learning
curve before further work on the135.13M ViLT stack. No additional run is implied
by this report; its data, initialization and prefix gates must be fixed separately.
