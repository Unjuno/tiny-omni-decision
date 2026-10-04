# Teacher v1: high-precision multimodal Decision Teacher

## Result status

The Teacher v1 candidate is frozen from validation. The sealed audit was started
once after that freeze and its result will be added below after the run completes.
The audit result is not used to change the checkpoint, sampling policy, or any
hyperparameter.

Teacher v1 is a quality-only phase. No ternary quantization, final LoRA merge,
Recovery LoRA, or teacher-logit cache was run.

## Evaluation integrity

Teacher v0's evaluation split is an observed legacy reference, not a blind test.
The v1 corpus uses immutable source revisions and seed 17. Train, validation,
and `sealed_audit` are separated by source record, underlying asset, media
identity, and NFKC/casefold/whitespace-normalized content fingerprints. The
manifest records zero pairwise overlap for all four checks. CLEVRER questions
are grouped by their scene/video asset before splitting; all questions from a
scene remain in one split, and the group split is deterministic.

Normal training receives only the train and validation paths. It has no
evaluation-path option, and checkpoint selection fails closed without
independent validation. `freeze-teacher-selection` records candidate, config,
corpus, and audit-manifest hashes. The evaluation command atomically claims the
single sealed-audit attempt before reading its rows. The audit JSONL is stored
under ignored `data/sealed/` and is not used by iterative experiments.

Selection was validation-only. The predeclared score (lower is better) is:

```text
macro NLL + 0.2 × macro Brier + 0.1 × macro ECE
  − 0.25 × macro Accuracy − 0.25 × minimum modality Accuracy
```

The candidate was selected from seed 17 and the frozen weak-modality sampling
policy. Two additional seeds (19 and 23) were run to measure variability; they
were not allowed to replace the primary seed. No audit metric was consulted for
any selection decision.

## Frozen corpus and coverage

Hashes:

| Input | SHA-256 |
|---|---|
| Train JSONL | `5a9d8f3c1608b9cedef0c4c4b1513dcf5e1face4a67acc99c8002849cf07fb2b` |
| Validation JSONL | `ebfa46399f3139f4c97a6e126b3036e00afc274de9433fdb5755c089bb9995a6` |
| Sealed audit data | `227da0f27ffb3fff3e40b3513939d2c180e057db0616206817cbc6d7d4d291bf` |
| Sealed audit manifest | `7e1da12dd681e2c377ebcbaf7ac34e3236bb04fde38ea482b2314ddfbf19712c` |

The immutable audit seed is 17. Its five source revisions and source manifest
hashes are recorded in `data/sealed/durable-teacher-v1/sealed-audit-manifest.json`.
The audit contains 5,689 examples from 1,587 underlying assets. It was built
from source records and assets excluded from the previously observed v0 corpus
where the sources allowed it. Of 15,184 eligible new records, 9,495 were
excluded as previously seen; no v0 evaluation records were reused. The audit
contains 3,637 text, 492 image, 185 audio, and 1,375 video examples. Audio is
limited to 31 speaker/source groups; that scarcity is recorded rather than
filled with seen audio records.

The train corpus has 39,126 records, including 34,267 that were not in the v0
train corpus. The full independent validation corpus has 3,507 records; model
search used a fixed, balanced 472-example validation subset (118 per modality)
so each scheduled checkpoint comparison used equal modality counts. The sealed
audit remains full size and was not sampled down. Only ALLOW-approved sources
were used: Typed Decisions Synth, Open-Jev, Clevr-4, Speech Commands, and
CLEVRER. MMAU, MVBench, OneJev REVIEW/DENY data, and unknown-license media were
not ingested.

The selected seed 17 checkpoint consumed 8,192 distinct training examples,
with zero repeats and 3,731 unique underlying assets:

| Modality | Unique examples consumed | Unique underlying assets |
|---|---:|---:|
| Text | 2,307 | 2,169 |
| Image | 1,272 | 318 |
| Audio | 1,396 | 926 |
| Video | 3,217 | 318 |

| Source | Unique examples consumed |
|---|---:|
| Open-Jev | 1,153 |
| Typed Decisions Synth | 1,154 |
| Clevr-4 | 1,272 |
| Speech Commands | 1,396 |
| CLEVRER | 3,217 |

All 8,192 consumed examples were unique. This is 8× the 1,024 distinct
training examples consumed by the v0 run. The full train corpus has additional
unused examples; sample accounting is kept per run so unused capacity is
visible.

## Sampling-policy search

The architecture and seed stayed fixed for the initial 512-step comparison.
Each policy saw the same 472-example balanced validation subset and had a
one-repeat cap. The fixed selection score selected the normalized weak-modality
policy. Accuracy/NLL pairs below are per modality.

| Policy (weights: audio/image/text/video) | Audio | Image | Text | Video | Macro Acc. | Minimum Acc. | Selection score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced (1/1/1/1) | .890/.405 | .636/1.112 | .678/.765 | .466/1.249 | .670 | .466 | .6943 |
| Weak modalities (1/1.5/1.5/2) | .864/.433 | .653/1.046 | .703/.755 | .458/1.283 | .670 | .458 | .6929 |
| Video priority (1/1.5/1/2.5) | .881/.442 | .661/1.114 | .686/.760 | .424/1.274 | .663 | .424 | .7228 |

The difference between balanced and weak-modality policies at 512 steps was
small. Weak-modality weights won under the predeclared score, mostly through
better image/text NLL while retaining similar audio and video accuracy. The
more aggressive video-priority policy performed worse on minimum and macro
accuracy. The selected weights were therefore frozen at audio 1, image 1.5,
text 1.5, video 2; no source-specific weights were used.

Every attempted run is recorded in the local candidate `experiments.jsonl`
files. Two unsuccessful attempts are retained in those logs: the initial
weak-modality run exposed a metric bookkeeping `KeyError`, and the first
balanced checkpoint promotion hit a transient Windows rename permission error.
After fixing the bookkeeping and checkpoint promotion, the normalized
weak-modality run completed, and the balanced validation metrics were recovered
from its saved checkpoint. A source-starvation issue in the first sampler was
fixed before comparing the normalized policy. These runs are not presented as
successful candidates.

## Learning curve and stopping

The primary seed 17 curve used rank-16 decoder LoRA, validation every 128 steps,
patience 4, and `min_delta=0.0005`. It stopped at the configured 2,048-step
maximum; the validation-selected checkpoint is step 1,920. The table includes
training CE/accuracy, macro validation metrics, minimum modality accuracy, and
per-modality validation accuracy. Full per-source, NLL, Brier, ECE, confidence,
and overfit-signal histories are written to each run's `learning-curves.jsonl`
and `validation-step-*.json` artifacts.

| Step | Train CE | Train Acc. | Val macro Acc. | Min Acc. | Macro NLL | Macro Brier | Macro ECE | Audio Acc. | Image Acc. | Text Acc. | Video Acc. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.5555 | .4883 | .5657 | .3305 | 1.1058 | .5070 | .1255 | .8390 | .5424 | .5508 | .3305 |
| 256 | 1.0643 | .5723 | .5953 | .3559 | 1.0417 | .4882 | .0990 | .8644 | .5508 | .6102 | .3559 |
| 384 | 1.0002 | .5918 | .6568 | .4322 | .9222 | .4458 | .1017 | .8644 | .6356 | .6949 | .4322 |
| 512 | .9342 | .6387 | .6695 | .4576 | .8794 | .4270 | .0987 | .8644 | .6525 | .7034 | .4576 |
| 640 | .8011 | .6875 | .6589 | .4153 | .8756 | .4281 | .1053 | .8814 | .6610 | .6780 | .4153 |
| 768 | .8658 | .6562 | .6864 | .5000 | .8350 | .4121 | .1082 | .8814 | .6525 | .7119 | .5000 |
| 896 | .7812 | .6836 | .6758 | .4661 | .8019 | .4009 | .1122 | .8814 | .6695 | .6864 | .4661 |
| 1,024 | .9119 | .6328 | .6843 | .4746 | .7733 | .3854 | .1084 | .8814 | .6949 | .6864 | .4746 |
| 1,152 | .7866 | .6758 | .6843 | .4576 | .7626 | .3871 | .0873 | .8814 | .7119 | .6864 | .4576 |
| 1,280 | .7432 | .7012 | .7119 | .4746 | .7353 | .3698 | .0977 | .8983 | .7542 | .7203 | .4746 |
| 1,408 | .9132 | .6016 | .6822 | .4237 | .7461 | .3801 | .1038 | .8898 | .6780 | .7373 | .4237 |
| 1,536 | .7422 | .6797 | .6949 | .4915 | .7325 | .3701 | .1032 | .8898 | .7034 | .6949 | .4915 |
| 1,664 | .7111 | .6992 | .6970 | .5000 | .7509 | .3810 | .0890 | .8898 | .6695 | .7288 | .5000 |
| 1,792 | .7221 | .6699 | .7267 | .5339 | .7197 | .3604 | .0891 | .8983 | .7458 | .7288 | .5339 |
| 1,920 | .6788 | .7168 | **.7267** | **.5508** | **.7008** | **.3593** | **.0808** | **.9068** | .7119 | **.7373** | **.5508** |
| 2,048 | .7226 | .7051 | .7034 | .5169 | .6997 | .3642 | .0829 | .8983 | .6949 | .7034 | .5169 |

Validation score improved through step 1,920 and then worsened. At step 2,048,
training accuracy remained high while the selected score, macro accuracy, and
weak-modality accuracy fell; the weak-modality-degraded signal fired. The
best-checkpoint rule retained step 1,920 rather than the final checkpoint.
All modalities improved in validation accuracy, NLL, and Brier versus Teacher
v0 on this same subset; audio and video ECE were slightly worse. This is not a
claim of absence of calibration trade-offs.

## Seeds and variability

Seeds 17, 19, and 23 used the frozen rank-16 policy and a 2,048-step maximum.
Each checkpoint was selected independently using the same validation-only
score and early-stopping rule. Seed 17 remains the primary by design.

| Seed | Best step | Macro Acc. | Min Acc. | Macro NLL | Macro Brier | Macro ECE | Audio Acc. | Image Acc. | Text Acc. | Video Acc. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1,920 | .7267 | .5508 | .7008 | .3593 | .0808 | .9068 | .7119 | .7373 | .5508 |
| 19 | 1,408 | .7055 | .5508 | .7502 | .3747 | .0892 | .8814 | .7119 | .6780 | .5508 |
| 23 | 1,536 | .7034 | .5339 | .7735 | .3888 | .0929 | .8983 | .6949 | .6864 | .5339 |
| Mean | — | .7119 | — | .7415 | .3743 | .0876 | .8955 | .7062 | .7006 | .5452 |
| Sample SD | — | .0129 | — | .0371 | .0147 | .0062 | .0129 | .0098 | .0321 | .0098 |

Per-seed source/modality summaries and full curves are retained under
`artifacts/tiny-omni-decision-teacher-v1/candidates/`. Seeds 19 and 23 did not
replace seed 17 despite their independently selected results. The reported
seed variability is based on validation, not the sealed audit.

| Seed | Modality | Accuracy | NLL | Brier | ECE |
|---:|---|---:|---:|---:|---:|
| 17 | Audio | .9068 | .3756 | .1361 | .0522 |
| 17 | Image | .7119 | .7652 | .3807 | .1017 |
| 17 | Text | .7373 | .6327 | .3565 | .0537 |
| 17 | Video | .5508 | 1.0298 | .5639 | .1154 |
| 19 | Audio | .8814 | .4173 | .1476 | .0488 |
| 19 | Image | .7119 | .8037 | .3766 | .1002 |
| 19 | Text | .6780 | .7252 | .4108 | .1316 |
| 19 | Video | .5508 | 1.0547 | .5638 | .0764 |
| 23 | Audio | .8983 | .3733 | .1387 | .0417 |
| 23 | Image | .6949 | .9568 | .4247 | .1045 |
| 23 | Text | .6864 | .6783 | .3879 | .0908 |
| 23 | Video | .5339 | 1.0857 | .6038 | .1347 |

The three-seed mean and sample standard deviation (`mean ± SD`) for each
validation metric are:

| Modality / aggregate | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Macro over modalities | .7119 ± .0129 | .7415 ± .0371 | .3743 ± .0147 | .0876 ± .0062 |
| Audio | .8955 ± .0129 | .3887 ± .0248 | .1408 ± .0061 | .0475 ± .0054 |
| Image | .7062 ± .0098 | .8419 ± .1013 | .3940 ± .0267 | .1021 ± .0022 |
| Text | .7006 ± .0321 | .6788 ± .0463 | .3851 ± .0273 | .0921 ± .0389 |
| Video | .5452 ± .0098 | 1.0567 ± .0280 | .5772 ± .0230 | .1088 ± .0297 |

## Selected model and validation comparison

- Artifact ID: `tiny-omni-decision-teacher-v1`.
- Base: `google/gemma-4-E2B-it-qat-q4_0-unquantized` at
  `6befbaca7398925921802abd1f277b495b78b738`.
- Rank 16, alpha 32, dropout 0.05, decoder q/v projections resolved from the
  loaded architecture; modality encoders and projector frozen.
- Trainable parameters: 2,678,784 of 5,106,976,288. Adapter weights:
  10,729,912 bytes (about 10.23 MiB). Projector adaptation: none.
- Selected seed/step: 17 / 1,920. Sampling weights: audio 1, image 1.5,
  text 1.5, video 2; one-pass unique-example cap.
- Best-checkpoint reload was verified. The frozen selection lock records the
  candidate/config/corpus/audit hashes and selected validation metrics.
- Local GPU: RTX 3080 Laptop 16 GB. Primary-run mean: 4.02 sec/optimizer step;
  peak allocator usage: 11,565,741,056 bytes. No cloud GPU was used.
- Total local GPU wall time across all 9 logged training attempts (including
  failed pilots) plus the single audit was about 14.19 hours. Cloud GPU time and
  cost were zero. The audit alone took 2,249 seconds.

| Modality | V0 validation Acc. | V1 validation Acc. | V0 NLL | V1 NLL | V0 Brier | V1 Brier | V0 ECE | V1 ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Audio | .8729 | .9068 | .4512 | .3756 | .1601 | .1361 | .0362 | .0522 |
| Image | .5847 | .7119 | 1.3566 | .7652 | .5581 | .3807 | .1479 | .1017 |
| Text | .6102 | .7373 | .8741 | .6327 | .4951 | .3565 | .1517 | .0537 |
| Video | .3729 | .5508 | 1.3941 | 1.0298 | .7020 | .5639 | .1004 | .1154 |

V0 is shown only as a same-subset legacy comparison; neither v0 nor its old
evaluation data participated in model selection. Full v1 source and modality
metrics, including the sealed audit, are recorded in the result manifest after
the audit completes.

## Sealed audit

The candidate, config, corpus hashes, audit manifest, seed, checkpoint, and
selection rule were frozen before this first and only audit evaluation. The
claim file records the single attempt. Evaluation completed on 5,689/5,689
records in 37 minutes 29 seconds using the local RTX 3080 Laptop GPU. No tuning
or additional audit evaluation followed.

Overall audit metrics: accuracy `.6718`, NLL `.7827`, Brier `.4336`, ECE
`.0870`, mean confidence `.7581`. Macro accuracy was `.7170`; the weakest
modality was video at `.5200`.

| Modality | N | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Audio | 185 | .9459 | .1534 | .0680 | .0310 |
| Image | 492 | .6890 | .8547 | .4046 | .1086 |
| Text | 3,637 | .7130 | .6809 | .3871 | .0728 |
| Video | 1,375 | .5200 | 1.1107 | .6162 | .1294 |

| Source | N | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Speech Commands | 185 | .9459 | .1534 | .0680 | .0310 |
| Clevr-4 | 492 | .6890 | .8547 | .4046 | .1086 |
| Open-Jev | 2,764 | .6780 | .7551 | .4312 | .0821 |
| Typed Decisions Synth | 873 | .8236 | .4462 | .2475 | .0441 |
| CLEVRER | 1,375 | .5200 | 1.1107 | .6162 | .1294 |

The sealed-audit result file includes each source/modality count, mean
confidence, all metrics, the selection-lock hash, and checkpoint/audit hashes.
It is local at `data/sealed/durable-teacher-v1/sealed-audit-result.json`; its
hash is included in the checked-in v1 teacher manifest.

## Bottleneck assessment and next step

The 90% target was reached only for audio (94.6%); image (68.9%), text (71.3%),
and video (52.0%) remain below target on the sealed audit. This is the honest
result, not a redefinition of the target. The audit's macro Accuracy is 71.7%,
but aggregate Accuracy is not the product gate.

Video/CLEVRER is the clearest bottleneck: audit Accuracy is 52.0%, NLL 1.1107,
and Brier .6162. The 125 unique video scenes provide many questions but only a
modest number of independent visual/temporal assets; the base video pathway is
frozen, so temporal representation, data coverage, and benchmark difficulty
remain plausible limits. Image/Clevr-4 is 68.9% over 123 images, while Open-Jev
is 67.8% and Typed Decisions Synth is 82.4%; this gap suggests natural-text
source difficulty in addition to model capacity. Audio has only 31 independent
speaker/source groups, so its 94.6% should not be generalized beyond that
limited audit population.

The primary validation score and weakest-modality accuracy improved through
step 1,920 but regressed at step 2,048, and two replicate seeds selected between
1,408 and 1,536. Training CE continued to trend down while validation metrics
became noisy and the best score stopped improving. This is a plateau/overfit
signal at the tested data, architecture, and budget, not proof that the model
family cannot reach 90%. No rank-32, broader-target, or projector experiment was
run: those changes were not justified until this fixed rank-16 learning curve
had more evidence of a stable data-scaled capacity limit. The separately
committed [Video Teacher v2 design](superpowers/specs/2026-10-04-video-teacher-v2-design.md)
proposes testing frame count, decoder target coverage, rank, and CLEVRER task
coverage one factor at a time. It preserves this v1 baseline and has not been
implemented or trained in this phase; any v2 model needs its own clean
validation comparison and a new untouched audit source.

Teacher v1 does improve validation over v0 on all four modalities for Accuracy,
NLL, and Brier on the same 472-example balanced subset. On the audit, v1 is
compared with v0's already-observed legacy result only as context; the audit
sources/records differ, so those values are not a paired improvement claim.
Any later model change must be a separately versioned experiment with a fresh
untouched audit source.

On that same validation subset, sample-weighted Accuracy improved from `.6102`
to `.7267`, NLL from `1.0190` to `.7008`, and Brier from `.4789` to `.3593`;
macro ECE improved from `.1091` to `.0808`.

For historical context, v0's already-observed 2,917-record evaluation reported
Accuracy `.7343`, NLL `.6380`, Brier `.3344`, and ECE `.0344` (per-modality
breakdown remains in [DURABLE_TEACHER.md](DURABLE_TEACHER.md)). V1's new sealed
audit reports `.6718`, `.7827`, `.4336`, and `.0870` respectively. The sets
have different records, sizes, and source balance; these numbers must not be
read as a paired regression or used to retune the frozen v1 candidate.

## Run artifacts

Generated corpora, media, predictions, curves, candidate checkpoints, adapter
weights, and the single audit result are local ignored artifacts. The tracked
manifest at `manifests/teachers/tiny-omni-decision-teacher-v1.json` records
reproducibility hashes and artifact paths without embedding the 5B base. See
the Python `pip-freeze.txt` beside each candidate for exact package versions.
