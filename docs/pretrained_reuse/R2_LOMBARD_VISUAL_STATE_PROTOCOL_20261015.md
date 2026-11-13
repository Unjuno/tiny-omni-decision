# Fixed-model visual-state diagnosis

Bounded extension of the completed temporal evaluation at af6db01. User permits
autonomous local research; no new model/seed/budget or quality-based selection.
Previous turn is progress: five real fits, fresh replay, controls, published CI.

Question: does matched visual content cause the high joint score, or can a fixed
training-derived latent visual state reproduce it? Diagnose the fixed selected
joint_broadcast23/joint_ordered23 weights. Secondary video_ordered18 diagnosis
retains full51options and examines cached-token reversal. No adoption/fix assumed.

Use all4080train and889observed development utterances, unchanged398calibration
temperatures, frozen native audio/16frame caches/source/split gates. No calibration
fitting or sealed data. Derive constants only from training features: mean over
all4080selected-model visual final states(FP64 accumulation, FP32 stored), and
first frozen training row(index0, identity recorded), without its labels.

For every fixed model, save native matched logits and a decomposed-path exact
parity check. Record visual GRU final states and across-row variance, input
variance, saturation(abs(h)>=.99), and contribution variance after the first
readout weight's audio/visual slices. These are descriptive, not quality gates.

Same889audio features and targets for controls:
1. matched native visual input;
2. standardized visual input zero with original presence (training pooled-feature
   mean repeated16times, not a natural blank frame);
3. fixed first training visual input repeated for every validation row;
4. reverse16cached visual tokens, not raw-frame encoder re-extraction;
5. fixed training-mean final visual latent with original presence;
6. fixed first training final visual latent with original presence;
7. zero final visual latent, original presence;
8. zero final visual latent, visual-presence bit cleared.

Controls5–8 intervene at the existing readout boundary; no layer weights change.
7versus8isolates the mask while holding audio and zero visual state constant.
Preserve audio masking for video-only model. Record per-slot/macro/minimum/
all-six Accuracy, NLL/Brier/ECE, all logits/probabilities/IDs/targets, exact slot
answer changes and logit differences from matched. Primary paired whole-speaker
bootstrap10000/seed17/linear percentile95%: train-mean-latent minus matched in
each joint model. Nine observed development groups/model selection/single seed
and artificial latent/mask OOD constrain interpretation. No equivalence threshold
or claim of natural missingness/calibration robustness, no control-based tuning.

CPU4threads/interOp1/FP32/deterministic; retain requires_grad flags for parity,
eval/no_grad and zero optimizer/gradient updates. Encoders not loaded. Resource
guard600seconds/3GiB peak/64MiB output/1GiB disk reserve; preserve old GPU8GiBguard.
Code/tests/protocol and all source/runtime/checkpoint/normalizer/receipt inputs
hashed before/after. New output only, no legacy artifact overwrite. Freeze code
before running. A separate fresh process repeats all controls/latents/constants
byte-exactly, hashes/report checks and unchanged state/parameter checks before
publishing. Failed attempts retained, no hash bypass or speculative root cause.
