# FLAN final-block fixed-budget result

Completed384updates, seed17,8examples/update:3,072 unique records across512
training states. Train/development state groups are disjoint. Uses the existing
hash-frozen context corpus; shuffle Python Random17, first3072 without repetition.
This order differs from earlier Cross-Encoder runs, so no isolated model ranking
is claimed. Validation is the existing observed2304-question development set.

Only final decoder block trains (3,147,264 parameters); total76,961,152.
AdamW2e-5,weight decay.01,clip1,dropout disabled,FP32. No scheduler, retries,
seed search or added steps. Conditions and original options are rendered in a
fixed prompt. Letter logits use the supplied option count; no choice reduction.
All tokenized inputs fit512 without truncation. Protocol and example-order hashes
were saved before evaluation/training. No sealed audit was accessed.

| Update | Validation Accuracy | NLL |
| --- | ---: | ---: |
| 0 | .261285 | 1.576577 |
| 128 | .242622 | 1.430145 |
| 256 | .233941 | 1.422501 |
| 384 | .239583 | 1.407907 |

Predefined selector: earliest minimum pooled development NLL including step0.
Selected384; best and final files retained separately. NLL improvement does not
establish better decisions: Accuracy remains poor. No adoption or extra budget
is justified by this run. It does not establish a family-wide capability ceiling.

All frozen tensors exactly match original loaded weights after training.
Best reload reproduces first16 saved validation probabilities exactly; this is
a bounded reload check, not a full independent validation rerun. No OOM or
nonfinite loss/gradient. Total train/evaluation/check time38.82s excluding load;
peak CUDA allocation442,663,936bytes. RTX3080Laptop, local only, no cloud charge.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/flan-last-block-context-01/`:
protocol, ordered IDs, complete step losses/gradient norms, four validation
prediction arrays, per-context Accuracy/NLL/Brier/ECE, best/final deltas and
result. Companion JSON contains metrics and artifact hashes. No old artifacts
were overwritten. Command: `python -m scripts.train_flan_context`.

The77M reference cannot simply replace text in the existing VideoPrism+Whisper
stack under200M. No compression or product architecture change is implied.
Transfer tests are not run: in-domain conditional quality already fails, so
further benchmark exposure is not warranted for this candidate.
