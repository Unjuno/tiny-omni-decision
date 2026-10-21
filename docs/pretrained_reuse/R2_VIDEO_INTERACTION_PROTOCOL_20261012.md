# Video interaction control, fixed before execution

Transfer the successful image interaction parameterization to existing frozen
CLEVRER binary-choice features. Fixed existing3794 train choices/256scenes and
3014 validation choices/200scenes; previously observed development selection data.
Do not use the separate200-scene control set or sealed audit in this experiment.

Three fresh seed17 runs: concat2304→128→1/GELU (295169 parameters), bilinear
video768/text1536→128 dot/sqrt128 (295168), and the same bilinear architecture
with video zeroed in both training and validation. Text1536 remains the existing
question,option,absolute-difference,product feature construction. Preserve all
existing input/model hashes,128-token text preprocessing,8-frame pooled video,
weights equal-question-type/equal-scene,AdamW1e-3/wd1e-3,batch64,24epochs.
Same initialization RNG state for all heads and identical shuffled example order.

Each condition selects earliest minimum validation macro-question-type NLL.
Keep best/final independently, record curves/unique counts/consumption/VRAM/time,
and verify reload. Stop on nonfinite gradients/loss; no quality-based early stop.
Separate12-choice-per-split smoke process first; full runs start fresh.

For each selected scorer also replace each validation scene's video with another
scene by the existing fixed seed17 derangement. Record all probabilities and
Accuracy/NLL/Brier/ECE by explanatory/predictive/counterfactual plus macro.
Paired200-scene bootstrap10,000 draws,seed17,percentile95% macro-Accuracy difference
for bilinear-minus-concat, bilinear-minus-no-video and matched-minus-shuffled.
Selection on this validation remains a limitation, not a blind temporal claim.
Do not increase budget, retrain encoders or promote a product automatically.
