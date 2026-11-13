# R2 learned mask-pair correspondence — executed 2026-10-14

## Decision and scope

**Execution PASS: all16 fits×512 steps, fixed curves and final reload verified.**
**Reject the appearance-augmented endpoint.** It memorizes training proxy pairs;
scene-held-out finalNLL worsens versus geometry in all8 scenes. No product adoption,
new teacher, larger budget, earlier-checkpoint selection or calibration fit follows.
This is an auxiliary binary instance-correspondence diagnostic, not original
PerceptionTest QA, semantic cue→cover/memory performance or the Omni95–98% target.

All8 clips were already observed training diagnostics. Leave-one-video-out folds
keep each scene entirely outside its training pool, but do not create blind data.
Scorable pair eligibility is derived from source training boxes: unique available
IoU>=0.5 plus model-mask own foreground point in bbox; source IDs provide labels
only. Missing competing source boxes cannot certify full physical membership.
All18,441 retained-region inputs remain unchanged. GTbest masks/source boxes/IDs never
enter input features, runtime seeds, option descriptors or scorer channels.

## Corpus and exposure

31adjacent/24gap8/1endpoint windows perclip; all Cartesian proposals accounted.
**2,628,771** pair opportunities, **5,400** supervised pair rows (1,771same,
3,629different), **2,623,371** unknown/ambiguous and unscored. These remain
explicit ledgers, not negatives or erased hard examples. Only~0.205%
of pair opportunities receive proxy supervision. This conditional metric is not
quality over all proposals or over a reduced version of the original QA benchmark.

Unique media8, source track instances32 (24group+8hidden); cover is a group subset.
Pairs/masks/frame combinations are correlated, not5,400 independent scenes.
Each fit consumes65,536 shuffled pair draws, same order for both arms in a fold.
Perfit unique train pairs3,385–5,191; repeated draws60,345–62,151. All16 fits consume
1,048,576 draws; sum of perfit unique exposures75,600, sum perfit
repeats972,976. Across the entire experiment, distinct pair
rows remain5,400: do not confuse that with the sum across overlapping folds/arms.

Supplemental frozen-label composition: only15 positive pairs link the hidden
source object across times (12 in10103,2 in1087,1 in10894);1,756 positive pairs
link group members. Different pairs comprise288hidden/group and3,341group/group.
These are counts, not a chosen new benchmark or model-selection slice. The probe
primarily supervises group-instance correspondence and cannot establish hidden
cue-memory learning from a high aggregate pair score.

## Fixed architecture/budget

Geometry5→64GELU→1:449parameters. Area appearance768mean/std difference+product
(1536) plus geometry5→64GELU→1:98,753parameters. Stored geometry files2,982–2,994bytes;
appearance files379,084–379,345bytes, including train-only normalizer buffers.
All resident pretrained encoders would still count in a product; these small heads
alone do not satisfy a <=200M Omni release. Arms differ in parameter count, so this
comparison cannot attribute the difference solely to appearance input.

Geometry is model-box center difference/logwidth-logheight difference/time gap.
No source ID/box, scene ID, proposal ordinal or text input. Same CPU seed17, AdamW
LR5e-4/weightDecay1e-4, batch128/clipnorm1,512updates fixed,7train scenes/1held-out.
No held-out checkpoint selection or early stopping. Every prescribed curve point,
fold, stratum and outcome retained. Inference-only control rolls query/destination
appearance by1/7 within each native time, preserving geometry/targets and fixed
weights; this distribution perturbation is not causal semantic evidence.

## Final scene-macro metrics

| Arm/control | Accuracy | Balanced Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| geometry | 0.707013 | 0.679343 | 0.535778 | 0.362786 | 0.121942 |
| appearance | 0.670322 | 0.720866 | 1.875279 | 0.570371 | 0.262296 |
| appearance_mismatch | 0.623554 | 0.500933 | 2.554838 | 0.693140 | 0.337855 |
| prior | 0.660024 | 0.500000 | 0.642153 | 0.449786 | 0.037830 |

Appearance has higher balancedAccuracy but worse Accuracy/NLL/Brier/ECE. Its8
training fits all reach1.0Accuracy; scene transfer/calibration fail this endpoint.
Retain the mixed result instead of declaring every metric worse. The altered
appearance control's balancedAccuracy near0.5 shows that inputs affect predictions;
it does not establish correct cover association or reliable memory.

| Held-out known train clip | Pairs | Geometry Accuracy | Appearance Accuracy | Geometry NLL | Appearance NLL |
|---|---:|---:|---:|---:|---:|
| video_10103 | 209 | 0.497608 | 0.818182 | 0.655597 | 1.219240 |
| video_10186 | 544 | 0.773897 | 0.555147 | 0.509541 | 2.873737 |
| video_10212 | 395 | 0.693671 | 0.794937 | 0.618788 | 0.802446 |
| video_10387 | 496 | 0.542339 | 0.395161 | 0.670369 | 5.109349 |
| video_1087 | 947 | 0.821542 | 0.693770 | 0.416244 | 1.085059 |
| video_10886 | 352 | 0.747159 | 0.573864 | 0.510798 | 2.239822 |
| video_10894 | 2015 | 0.792556 | 0.762283 | 0.407898 | 0.867301 |
| video_10903 | 442 | 0.787330 | 0.769231 | 0.496988 | 0.805278 |

## Learning curves and temporal strata

| Fixed step | Arm | Train Accuracy | Held-out Accuracy | Train NLL | Held-out NLL |
|---|---|---:|---:|---:|---:|
| 0 | geometry | 0.555223 | 0.520658 | 0.698905 | 0.704439 |
| 0 | appearance | 0.518427 | 0.553764 | 0.692532 | 0.685351 |
| 128 | geometry | 0.783397 | 0.709234 | 0.546333 | 0.585501 |
| 128 | appearance | 0.972285 | 0.683039 | 0.105652 | 0.968061 |
| 256 | geometry | 0.786714 | 0.706835 | 0.483157 | 0.551282 |
| 256 | appearance | 0.996770 | 0.671339 | 0.032820 | 1.360479 |
| 384 | geometry | 0.789223 | 0.703554 | 0.453377 | 0.539777 |
| 384 | appearance | 0.999901 | 0.670077 | 0.013100 | 1.646126 |
| 512 | geometry | 0.791790 | 0.707013 | 0.438410 | 0.535778 |
| 512 | appearance | 1.000000 | 0.670322 | 0.006751 | 1.875279 |

Appearance held-outNLL rises128→512 in **everyfold**, while trainingNLL falls.
This is direct overfit evidence, not a fixed-example gap inferred from rolling
minibatches: each curve evaluates its full fixed train/held-out pair pool.
Do not retrospectively choose128 or a lucky fold as the selected model.

Strata0=adjacent,1=gap8,2=first→last. Endpoint counts are small; all counts/exclusions
are retained in the [JSON](R2_MASK_PAIR_ASSOCIATION_RESULTS_20261014.json) and
[learning-curve CSV:80 evaluations/640 rows](R2_MASK_PAIR_ASSOCIATION_RESULTS_20261014.csv).

| Arm | Stratum | Scene-macro Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| geometry | 0 | 0.764016 | 0.437156 | 0.288418 | 0.147568 |
| geometry | 1 | 0.631115 | 0.660341 | 0.457107 | 0.146352 |
| geometry | 2 | 0.729167 | 0.598604 | 0.405440 | 0.238260 |
| appearance | 0 | 0.693897 | 1.818469 | 0.523883 | 0.245833 |
| appearance | 1 | 0.645284 | 1.917472 | 0.618727 | 0.291344 |
| appearance | 2 | 0.568006 | 2.390783 | 0.778418 | 0.416257 |

Pairedscene bootstrap,8 differences,2,000resamples,seed17,percentile95% intervals:
appearance−geometry Accuracy-0.036691
[-0.138962,0.092781];
NLL1.339501
[0.503036,2.361758];
Brier0.207585
[0.014855,0.391451].
Eight previously known groups, proxy eligibility and shared training pools across
folds limit uncertainty interpretation; bootstrap does not retrain fold models.
These are not independent blind-test confidence intervals.

## Actual execution/reproducibility

Source/protocol commit`5dd208bf3df050c3f0680fea4960a91ebe86855d`. ConfigSHA`48068c199b868063c31969379a9c20b64688d6dab9e78a146dc4ee0cc3c63d47`;
pair NPZSHA`6334f2081fc3de37f393cda9941584cec87497dd409111f683c21441aab171d3`,61,575,001bytes outsideGit.
CPU existing Python3.11.9/Torch2.6.0+cu124/NumPy2.4.6,2threads/
1interop, no GPU/new pretrained loads/downloads/cloud. Main body23.007798s,
peak process working set736,698,368bytes (~702.57MiB).
Preparation14.290336s, smoke4.913315s, audit3.409058s;
total measured bodies45.620507s excluding imports. No product latency/energy claim.

```powershell
python -m scripts.train_mask_pair_association --prepare-only --output C:/CodexArtifacts/pretrained-reuse-r2/mask-pair-corpus-01
python -m scripts.train_mask_pair_association --corpus C:/CodexArtifacts/pretrained-reuse-r2/mask-pair-corpus-01 --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/mask-pair-smoke-01
python -m scripts.train_mask_pair_association --corpus C:/CodexArtifacts/pretrained-reuse-r2/mask-pair-corpus-01 --smoke-reference C:/CodexArtifacts/pretrained-reuse-r2/mask-pair-smoke-01 --output C:/CodexArtifacts/pretrained-reuse-r2/mask-pair-learning-01
python C:/CodexArtifacts/pretrained-reuse-r2/audit-mask-pair-learning-01.py
```

These record executed fresh outputs; no automatic rerun/resume or overwrite.
All16 final weights/history/normalizers/predictions/OOF/IDs/targets/control/sampler
orders retained locally; large files never added toGit. There is no best checkpoint:
final512 is the predefined diagnostic endpoint, not a selected Teacher.

Independent audit reconstructs functional linear/GELU scoring directly from weight
arrays, not the producer module; all saved held-out logits/probabilities/control
arrays match byte-for-byte. It recomputes final train/held-out/allstratum metrics,
train-only normalizers, all scene splits/sampler orders,80 curve accounting records,
OOF assembly and paired bootstrap. **Proxy eligibility itself was not independently
reconstructed**; source-role geometry has the earlier separate audit and existing
hand tests. All protected config/source/archive/hash checks pass. Code review's
perstratum exclusion-ledger correction was tested/fixed before execution.

24newtests RED→GREEN; fullCPU1,345passed/1skipped18.71s, Ruff/manifests/fourcatalogs/
policy passed. Read-only review before execution; exact publication-head CI after
push is saved locally. Existing legacy Teacher/base/config/corpora/artifacts unchanged.

## Next decision

Close this fixed appearance+geometry MLP/budget on these observed folds. No extra
epochs, seeds, hidden control tuning or capacity increase based on memorized training
accuracy. Geometry remains a cheap auxiliary reference, not a qualifying release.
Useful next work must address diverse scene/cue supervision and temporal continuity/
identity evidence rather than repeat this same fit. Weak geometry labels, few unique
assets, head capacity and representation transfer remain confounded; no pretrained
model capability ceiling is established.

Semantic cue→cover memory, unlabelled runtime proposal handling, original QA,
native text/image/audio/video joint/OOD/calibration95–98% and deployable<=200M
product remain unachieved. No new Teacher/artifact promotion or quantization.
