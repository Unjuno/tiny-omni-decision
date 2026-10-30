# Question-only control: actual fixed-budget learning

Before audio acquisition completes, executed the question-only arm using all
fixed admitted development questions. No filename changes or example removal.
This is a language-prior control, not audio understanding or multimodal quality.

Frozen17,389,824-parameter MiniLM; standardized features using train statistics
only. Candidate scorer49,409 trainable parameters, hidden64, GELU;384 reserved
audio coordinates are zero. Seed17, AdamW .001, weight decay .001, batch128,
24epochs. Earliest minimum raw validation NLL selects best; separate calibration
split selects temperature from17 geometric values .25..4. Protocol saved before
training; no post-result search. Later audio/joint arms must retain this schedule,
same width/candidate interface, split IDs and model-selection rule for comparison.

| Validation560 questions | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Train-frequency constant | .566071 | .684406 | .491284 | .002700 |
| Question-only selected | .648214 | .623079 | .433142 | .028331 |

Best epoch8, calibrated temperature1.414214. Sensitivity .703470, specificity
.576132, balanced Accuracy .639801. These results show useful question-only
predictability; do not attribute that performance to audio. They are selected
development results, not a blind estimate. Multiple questions per clip and
uploader require grouped uncertainty for eventual comparisons.

Training consumed55,488 presentations of2,312 unique questions. CPU training,
selection and calibration .653484s excluding feature extraction/setup. Best and
final24 weights saved separately. Option reversal gives identical aligned logits.
Independent-process best reload reproduced all560 probabilities bitwise.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r2/clotho-aqa-question-control-01/`
contains protocol, normalization, best/final, history/result, validation IDs,
probabilities and reload check. Exact command:
`python -m scripts.train_clotho_question_control`. Source base6d43ac4 plus runner
hash recorded in protocol. Ruff passed. Existing artifacts unchanged; no audit,
GPU training, paid compute or product promotion. Audio/joint arms remain pending.
