# CLEVRER temporal-descriptive order control (predeclared)

## Question and scope

Test whether the already-selected frozen VideoPrism + MiniLM readout changes
its decisions when the order of the same eight sampled frames is reversed on
the existing 200-scene descriptive holdout. The primary subset is the existing
`temporal_descriptive` type; `static_descriptive` rows are a within-scene
negative control. This is a fixed-checkpoint sensitivity diagnostic on
development data already evaluated once, not a blind benchmark, new model
selection, or a claim about VideoPrism's unpooled temporal capability.

## Frozen data and labels

- Source: `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef`.
- Raw validation question JSON SHA-256:
  `fdf841678a476655b906165e6ee4b0ed1516785c5ab927c8d0a4614e4e27e10d`.
- Sample validation manifest SHA-256:
  `3bd8a90f32b8977e83e6491452906a9447af7c098103df9cdb6268a92bac37f2`.
- The 400 selected rows cover 200 scenes, exactly one static and one temporal
  descriptive question per scene. All 400 rows match the source question text,
  answer, taxonomy, options, and program regenerated with
  `expand_descriptive_questions`; no mapping mismatch was found.
- Question type is taken from the pre-existing deterministic `classify`
  function: CLEVRER program tokens `after`, `before`, `end`,
  `filter_collision`, `filter_in`, `filter_moving`, `filter_out`, `get_frame`,
  and `start`, with a wording regex fallback only when program is unavailable.
  The selected temporal subset is not filtered using model outputs.
- Targets are the source descriptive `answer`; no labels or options are
  regenerated for this control. The temporal subset includes 69 six-choice,
  69 two-choice, 32 three-choice, and 30 eight-choice questions; therefore
  accuracy is reported alongside NLL/Brier/ECE.

## Frozen control protocol

- Readout: existing VideoPrism checkpoint
  `videoprism-clevrer-expanded-probe-v0/best-readout.pt`, SHA-256
  `f68b04a4079c537772938d27c6456841c5eed19595ab49d0f42dd014211f0dbc`.
- VideoPrism: revision `be719a406d563b66f0ac969e7c94bab8e997c81a`, pinned
  checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`; 8
  uniformly sampled frames, 288 px, frozen encoder, mean pooled tokens.
- MiniLM revision `4ca70771034acceecb2e72475f72050fcdde4ddc`; same frozen
  question/option features for forward and reversed video features.
- Intervention: reverse only the order of the same eight frames. Keep source
  asset, preprocessing, question, options, target, text features, readout,
  option ordering, and metric code fixed.
- Verify the recomputed forward probabilities against the existing saved
  VideoPrism baseline before interpreting the reversed condition.
- Report per-type and two-type macro Accuracy, NLL, Brier, and ECE-15, plus
  reversed-minus-forward paired 95% percentile intervals from 10,000 scene
  bootstrap draws (seed 17). No checkpoint selection, calibration, or training
  is allowed from these results. Intervals are diagnostic and do not make this
  reused validation set blind.
- Interpretation is descriptive: state whether the fixed readout's outputs
  react to reversal on this sample; neither an observed effect nor a null
  interval establishes a backbone or model-family capability limit.

## Reproduction

One evaluation completed on 2026-10-10 23:04:12–23:07:35 UTC (203.30 s total)
from repository commit `03e03d6723cc0d7c0fb5533171d4a055ba2572ee`. It ran on
an RTX 3080 Laptop GPU with PyTorch `2.6.0+cu124` and Transformers `5.6.2`;
peak CUDA allocation was 105,130,496 bytes. Text encoding took 1.36 s, paired
readout inference 0.042 s, and the 10,000-draw bootstrap 200.99 s. No training,
checkpoint selection, calibration, or sealed-audit access occurred.

Inputs and outputs (all external to the repository):

- Forward cache:
  `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-heldout-video-readout-v0-videoprism-features\video-features.npz`,
  SHA-256 `2b69e7253761b80fca970fe5849afde2fdc2e6a9951bb6493a9951fdb1cbb229`.
- Reverse cache:
  `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-heldout-video-readout-v0-videoprism-reverse-features\video-features.npz`,
  SHA-256 `9be358863af734248c6227138d238e304e6f961ad97115cb38114348b33ae753`.
  Its extraction used CPU JAX `0.6.2` over 201 unique train/validation videos,
  took 260.35 s, and peaked at 4,165,611,520 bytes working set.
- Run report:
  `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-temporal-descriptive-order-control-v0\report.json`,
  SHA-256 `7fc31eebfc304083be15e769bd7b5ceb957a67e680cd53ff39addf88d05be31f`.
- Paired predictions:
  `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-temporal-descriptive-order-control-v0\predictions.jsonl`,
  SHA-256 `6ef1a757a215c0a9baebd2d7066d2f6dddc1f09d56128634dc904d3f87bb6ead`.
- Forward recomputation matched the saved baseline to maximum probability
  delta `1.79e-7` (acceptance `<=1e-6`). The validation manifest hash is
  `3bd8a90f32b8977e83e6491452906a9447af7c098103df9cdb6268a92bac37f2`.

| Subset | Forward Acc / NLL / Brier / ECE | Reversed Acc / NLL / Brier / ECE | Reverse − forward |
|---|---|---|---|
| Temporal descriptive (n=200) | .435 / 1.214960 / .657060 / .08166 | .440 / 1.214980 / .657062 / .08668 | +.005 Acc / +.000020 NLL / +.000002 Brier / +.00501 ECE |
| Static descriptive (n=200) | .320 / 1.112112 / .638597 / .10284 | .320 / 1.112183 / .638631 / .10285 | .000 Acc / +.000071 NLL / +.000034 Brier / +.00002 ECE |
| Macro over the two types | .3775 / 1.163536 / .647829 / .09225 | .3800 / 1.163582 / .647847 / .09476 | +.0025 Acc / +.000045 NLL / +.000018 Brier / +.00251 ECE |

Paired 95% scene-bootstrap intervals for reverse-minus-forward temporal
descriptive metrics were Accuracy `[0, .015]`, NLL `[-.000076, .000115]`,
Brier `[-.000051, .000057]`, and ECE `[-.000060, .015041]`. Macro intervals
were Accuracy `[0, .0075]`, NLL `[-.000017, .000110]`, Brier
`[-.000022, .000059]`, and ECE `[-.004978, .009053]`. Thus the point estimate
is one additional correct temporal question, while probability and calibration
changes are tiny; the intervals include zero or have a zero boundary. The
static control's Accuracy did not change.

This provides little evidence that the **current pooled fixed readout** uses
frame order on these reused validation scenes. The near-invariance may also
reflect mean pooling of VideoPrism's spatiotemporal output tokens and the small
effect of this particular intervention. It says nothing decisive about
unpooled features, another readout, the VideoPrism backbone's general temporal
capability, or broader video reasoning. It does not justify a layer/tower
attribution or product decision. Existing feature caches, models, and
checkpoints were not modified; the reverse cache and all result files were
written to new directories.

