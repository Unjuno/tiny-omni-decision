# Candidate cardinality analysis

Fixed selected checkpoint 1792; saved development predictions only.
No model inference, training, calibration or checkpoint selection.

| Source | Options | Questions | Original Accuracy | Pair win rate | Original NLL | Pair NLL |
|---|---:|---:|---:|---:|---:|---:|
| original | 2 | 988 | 0.739879 | 0.739879 | 0.520126 | 0.520126 |
| original | 3 | 51 | 0.627451 | 0.774510 | 0.807384 | 0.478448 |
| original | 4 | 628 | 0.617834 | 0.799894 | 0.947448 | 0.440520 |
| original | 5 | 866 | 0.531178 | 0.793014 | 1.111512 | 0.419805 |
| original | 6 | 5 | 0.600000 | 0.800000 | 1.394506 | 0.472107 |
| original | 7 | 2 | 0.500000 | 0.916667 | 1.399959 | 0.401705 |
| curriculum | 4 | 1536 | 0.908854 | 0.966363 | 0.220071 | 0.078962 |
| curriculum | 5 | 512 | 0.636719 | 0.875488 | 0.841265 | 0.279712 |

## Interpretation and limitations

Each pair contains the known gold option and one distractor. All distractors
are included, with each question equally weighted within its option-count group.
Pair probability is p(gold)/(p(gold)+p(distractor)); ties score one half.
Original Accuracy uses the saved option order for argmax ties.
This uses gold information to construct pairs: it is not an inference policy
available before knowing the answer, nor a yes/no entailment evaluation.
Removing distractors changes the task and chance level (1/K versus 1/2).
Higher pair scores cannot count toward the original multi-option 95% gate.
Questions with different option counts also differ in content; do not infer
a causal option-count effect from cross-group Accuracy comparisons.
Pair NLL has a different sample space and cannot establish better calibration.
Pairs share questions/states; no independent-pair confidence claim is made.

Keep a variable-option product contract, including two-option questions.
A learned yes/no reformulation would require separately frozen, semantically
equivalent questions and joint consistency evaluation. This analysis does not
prove which trained architecture is superior. No product model is promoted.
