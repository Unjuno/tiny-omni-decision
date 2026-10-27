# DROID matched-history sensor ablation on train folds

The [predeclared procedure](R2_DROID_HISTORY_SENSOR_CV_PROTOCOL_20261027.md)
fit two small L2 softmax heads on the same previously inspected 275 train
recordings and the exact prior five outer recording-group folds. Both receive
the same one-hot previous command. The second additionally receives the
previously fixed 112-value sensor temporal representation. Each outer fold
selected L2 using four inner train-only group folds from the fixed grid
`100, 1000, 10000`; every fit selected 100, the smallest supplied value.
All fits converged. The heads have 56 and 840 trainable parameters respectively;
this is a sensor-input ablation with different model capacities and penalties.

| Predictor | Accuracy | Macro accuracy | NLL | Macro NLL | Brier | ECE-10 |
|---|---:|---:|---:|---:|---:|---:|
| History-only learned head | 0.9273 | 0.8109 | 0.7817 | 1.1368 | 0.3429 | 0.4011 |
| History plus sensor learned head | 0.8109 | 0.6050 | 0.8415 | 1.2542 | 0.3785 | 0.2763 |
| Prior add-one transition table, history only | 0.9455 | 0.9002 | 0.3805 | 0.6080 | 0.1338 | 0.1319 |

The predeclared primary difference, history-plus-sensor minus history-only
macro NLL, is **+0.1174**; higher is worse. The sensor arm had lower ECE but
worse Accuracy, NLL and Brier. All three approaches got **0/15** episodes
where the next command changed; these target-defined cases are descriptive,
not a selected evaluation set. On 260 stable-command episodes, learned
history-only Accuracy was 98.08% and history-plus-sensor was 85.77%.

Under this fixed training setup, the extra sensor representation did not help.
The grid's lower boundary was selected in every fit, the two heads have
different parameter counts, and only 15 changes exist; this does not show
that temporal models or sensors cannot help under other conditions. The two
learned heads and add-one table are distinct methods and their inputs must not
be described as identical across all three. Five folds share training data;
the cohort is not blind, independently sampled by site, or evidence of audio,
video, multimodal fusion or the product quality target.

Execution used CPU Python 3.11.16, NumPy 1.26.4 and SciPy 1.17.1, with four
BLAS threads; fitting and result production took 2.54 seconds. No GPU, encoder,
validation rows or sealed audit were used. A separate replay checked all ten
saved heads: training-only means matched exactly, parameter counts matched, and
held-fold probabilities reproduced with maximum absolute difference **0**.
The next task needs prospectively specified independent groups and enough
action-change examples rather than optimizing this previously observed cohort.

Private output:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-history-sensor-cv-20261027-attempt-01/`.
Raw SHA-256 of `summary.json`:
`3ba6385110e2cfe6950c43a5dc6dc6fbfddaad8e40c53a7f0b70271f4e281441`;
`private-predictions.jsonl`:
`64d0831cd5edca353770cb9fb5406e074f319e026378f3784e29b34f696c5b43`.
The output contains all inner-grid scores, exact folds, held predictions,
per-class results, input/protocol hashes and hashes of the ten saved heads.
No product head was created.
