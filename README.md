# Tiny Omni Decision

**2026-10-17 画角保持の長期比較完了:** 同じ条件で32,768更新を完走し、best/finalの
全予測の再現も一致。選択checkpointではImage35.40→38.13%、Video33.75→38.00%だが、
Audio61.27→58.18%、macro NLL1.630→1.674となり、採用条件は未達。
動画のAccuracy差の区間は0を含む。元controlと全結果を保持し、95–98%の製品目標は未達。
[全曲線・同条件比較・不確かさ](docs/pretrained_reuse/R2_NATIVE_FULL_EXTENT_RESULTS_20261016.md)。

**2026-10-16 正例被覆の比較学習完了:** 292問の音声イベント負例を未使用の正例へ置換し、
他の提示位置・モデル・予算を固定して32,768更新を完走。best/finalの別プロセス再現も一致。
非speech balanced Accuracyは50.00→54.11%だが、選択checkpointのmacro NLLは
1.630→1.679、Audio Accuracyは61.27→45.06%へ悪化し、採用条件を満たさなかった。
失敗結果と全履歴を保存。次は元controlを使い、画角保持だけを独立比較する。
4モダリティ95–98%は未達、製品昇格なし。
[全曲線・クラス別指標・限界](docs/pretrained_reuse/R2_NATIVE_SUPPORT_RESULTS_20261016.md)。

**2026-10-16 データ／学習パイプライン監査完了:** 消費した音声イベント5種類は正例0。
うち4種類は元TRAINの正例をsamplerが未使用だった。画像は中央75%、動画は中央66.67%の
視野だけを保持し、動画128→8フレームの最大間隔は0.76秒。全件の入力対応・特徴transport・
独立ラベル計算・実入力の蓄積勾配・保存評価集計では取り違えを確認しなかった。
次は正例被覆と視野保持を別々に検証し、その後に時間処理・regularizationへ進む。
既存runは固定。追加学習0、sealed audit未使用、製品目標未達。
[監査結果と次の順序](docs/pretrained_reuse/R2_NATIVE_PIPELINE_AUDIT_RESULTS_20261016.md)。

**2026-10-16 long learning curve completed:** One fixed native attention run
finishes 32,768 updates, followed by exact full best/final replay. Within-run
2,048→32,768 macro Accuracy improves 32.83→45.70%, while NLL worsens 1.740→4.848;
validation NLL selects 8,192. Video TRAIN reaches 100%, but validation 37.25% and
temporal 33.5% show no clear late generalization. Short runs did miss continued
Accuracy gains; this does not demonstrate grokking or a capacity ceiling. Preserve
best/final and all history. Next isolate regularization under a fixed long budget
before combining data/architecture changes. No additional run, audit or promotion.
[All learning curves and limits](docs/pretrained_reuse/R2_NATIVE_LONG_BUDGET_RESULTS_20261016.md).


**Unique-coverage control completed (2026-10-16):** Both fresh-initialized arms
finish2,048 updates with exact best/final/complete TRAIN replay. Replacing repeats
with6,144 unused nonVideo TRAIN questions improves Image29.05→31.54% and
Audio53.03→54.92%, but Video37.50→32.75% fails retention. Do not adopt; preserve
both results. Video exposure is identical, so next diagnose shared-readout
optimization on fixed TRAIN before choosing a small branch or optimizer change.
No capacity ceiling, xLSTM gain, product promotion or95–98% claim.
[Full comparison, curves and uncertainty](docs/pretrained_reuse/R2_NATIVE_COVERAGE_RESULTS_20261016.md).

**Actual consumed TRAIN measured (2026-10-16):** All6,656 distinct consumed
questions and all five initial/fixed-endpoint arrays reproduce exactly. The
attention baseline reaches only25.87% Text,21.92% Image and42.19% Video on TRAIN.
There are43 common tasks,48 TRAIN-only and28 validation-only: full macro averages
are not a direct generalization gap. Next compare repeat versus unused-TRAIN
coverage with matched compute/source/task/option counts; Video has no unused rows
in this frozen corpus. No architecture ceiling, promotion or95–98% claim.
[Full measurements and next decision](docs/pretrained_reuse/R2_NATIVE_CONSUMED_RESULTS_20261016.md).

**Matched-batch comparison completed (2026-10-16):** Both independent/matched
arms finish1,024 updates on the identical8,192-occurrence pool, with exact full
best/final and TRAIN-contrast replay. Matched macro Accuracy30.36%→31.61% does
not improve macro NLL; consumed text both-correct2/442→1/442 and video0/16 in
both. Video Accuracy gain has a scene interval spanning zero; STARSS remains
all-no. Do not adopt or infer an encoder/memory ceiling. Next measure actual
consumed-TRAIN task reachability before another optimization/architecture change.
[All results, learning curves and limitations](docs/pretrained_reuse/R2_NATIVE_MATCHED_BATCH_RESULTS_20261016.md).

**Native state dependence measured (2026-10-16):** Fixed TRAIN contrasts now
compare identical questions/options across different inputs and correct answers.
All1,250pairs reproduce exactly. The2048 endpoint answers both sides correctly
on0/442 consumed text pairs and1/16 consumed video pairs; image predictions
respond to inputs but remain inaccurate. These are constructed TRAIN diagnostics,
not test accuracy or an encoder ceiling. Next compare matched-state batching
with an equal-data/equal-budget control before expanding temporal architecture.
[Evidence, missing conditions and next experiment](docs/pretrained_reuse/R2_NATIVE_STATE_DEPENDENCE_RESULTS_20261016.md).

**Native baseline/LSTM comparison completed (2026-10-16):** Both models finish
2,048 updates and reproduce best/final predictions exactly in fresh processes.
Total parameters are196.255M /197.291M. Fixed-endpoint temporal-video Accuracy
is33.5% for both; LSTM worsens macro NLL and image Accuracy. Do not adopt it or
infer a recurrent-memory benefit. STARSS Accuracy83.38% comes from all-no outputs,
with zero positive recall. Preserve these baselines; prioritize native-value
readout and unique training coverage before expanding architecture search.
All-task95–98%, joint/OOD and deployment gates remain open.
[Measured comparison, curves and next priorities](docs/pretrained_reuse/R2_NATIVE_TEMPORAL_FIT_RESULTS_20261016.md).

**Temporal memory candidate (2026-10-16):** A separate LSTM residual path now
preserves native video spatial tokens and speech timelines. Total197,291,290
parameters; actual GPU plumbing and fresh-process replay pass. Task quality is
unmeasured; xLSTM remains a possible subsequent comparison.
[Measured costs and limits](docs/pretrained_reuse/R2_TEMPORAL_STATE_RESULTS_20261016.md).

**Full input features completed (2026-10-16):** All4,382 language states,
22,328 query pairs and7,410 audio examples now have frozen features. Independent
full recomputation matches every saved value exactly;1,135 source/input pins pass.
Next align other modalities' query features and freeze the common fit contract.
The shared readout remains untrained; no quality claim.
[Execution, hashes and remaining gaps](docs/pretrained_reuse/R2_NATIVE_FULL_FEATURES_RESULTS_20261016.md).

**Common training preparation (2026-10-16):** All456 existing development video
scenes now have persistent8-frame full tokens, exactly matching historical hashes.
Source audit identifies long-text handling and audio-profile gaps before fitting.
No new training or quality result. [Measured preparation](docs/pretrained_reuse/R2_NATIVE_TRAINING_PREPARATION_20261016.md).

**Native candidate actually integrated (2026-10-16):** One original-pretrained
encoder composition totals **196,254,874 parameters before quantization** and
produces JSON from native text/image/audio/video. Fresh-process replay matches
all 21 arrays and eight JSON outputs exactly. The new 889,345-parameter readout
is **untrained**: no task accuracy, calibration or release claim. Next freeze
common native supervision/evaluation and train the readout with retention;
all required-task 95–98%, joint/temporal/OOD and package gates remain open.
[Measured integration and limits](docs/pretrained_reuse/R2_NATIVE_CONDITIONAL_CANDIDATE_20261016.md) ·
[Current strategy](docs/pretrained_reuse/NEXT_STRATEGY_20261016.md).

**Matched condition-relevance fits completed (2026-10-16):** Both arms finish
1,024 updates with 18,432 unique questions and exact selected reload. New16-slice
minimum Accuracy is100% in both, but all trained checkpoints fail47-slice
retention; both select0. The auxiliary adoption gate fails. Three fixed weights
pass exact536-question TRAIN replay. Fixed256-state transfer and independent
output audit now complete: endpoint Accuracy CE93.70% /relevance94.17%, but
new-query/new-option count Accuracy50.00% /50.88%, with an actual syntax slice
at0% for both. No promotion or four-modality claim.
[Measured curves and preserved failures](docs/pretrained_reuse/R2_CONDITION_RELEVANCE_FIT_RESULTS_20261016.md) ·
[Completed transfer and limits](docs/pretrained_reuse/R2_CONDITION_RELEVANCE_TRANSFER_RESULTS_20261016.md) ·
[Current strategy and next gates](docs/pretrained_reuse/NEXT_STRATEGY_20261016.md).

**Next comparison prerequisites measured (2026-10-15):** The declared nine-file
conditional/native candidate plus calibration contains195,655,225 stored tensor
elements including buffers, leaving4,344,775 below200M. This is a conservative
component inventory; combined runtime, complete quality and <=200MB packaging
remain unverified. Independent condition-relevance data is generated:1,024 new
states /28,672 questions; all12 files match a separate-process regeneration.
Old validation31 slices are preserved alongside16 new slices. No model fitting
or quality gain in this preparation. Next freeze one matched CE versus relevance
supervision comparison and its native transfer/retention gates.
[Measured budget](docs/pretrained_reuse/R2_CONDITIONAL_CANDIDATE_BUDGET_20261015.md) ·
[Data coverage and isolation](docs/pretrained_reuse/R2_CONDITION_RELEVANCE_DATA_RESULTS_20261015.md) ·
[Current strategy](docs/pretrained_reuse/NEXT_STRATEGY_20261015.md).

**Current result and next priority (2026-10-15):** The fixed semantic adaptation
has now been connected to the observed256-image bridge. Oracle Accuracy improves
41.80%→73.83%; image-predicted facts41.11%→73.34%. However, all256 count questions
fail even with correct facts:268 oracle errors select true-but-unrequested
attributes. Native perception has54 additional count errors hidden by this failure.
Full26-file fresh replay and independent20480-prediction verification pass.
Prioritize condition-to-property transfer with independent, matched supervision;
close this comparison without prompt/mask/checkpoint tuning. The fit still selects0,
and whole <=200M/native four-modality95–98% gates remain unmet. No promotion.
[Current priorities and stop rules](docs/pretrained_reuse/NEXT_STRATEGY_20261015.md) ·
[Completed image-bridge comparison](docs/pretrained_reuse/R2_SEMANTIC_NATIVE_RESULTS_20261015.md) ·
[Completed fit, curves and limitations](docs/pretrained_reuse/R2_DECISION_SEMANTIC_FIT_RESULTS_20261015.md).

The entries below preserve earlier milestones; the current strategy above takes
precedence for the next action.

**Independent decision supervision generated and verified:** 23,552 new questions
across 2,432 states, with all candidate sets retained and 31 mandatory validation
slices. Full CPU regeneration matches all 13 files. Existing replay/validation
bytes are preserved; no model inference, training or quality improvement is
claimed. Next freeze one semantic fit against this data/retention contract.
[Coverage, integrity, selection rules and remaining work](docs/pretrained_reuse/R2_DECISION_SUPERVISION_RESULTS_20261015.md).

**Frozen representation/readout screen completed:** On1,024 structured-state
questions with all40 options, DeBERTa direct→factorized Accuracy improves
34.08%→59.08%, but minimum-family25% and family retention fail. MiniLM changes
32.42%→28.91%. Both full fresh-process replays match25payloads exactly. No training,
adoption or audit access. Close this fixed rendering screen; define conditional,
numeric and old-task retention coverage together before another semantic fit.
[Results, controls, parameter budgets and limitations](docs/pretrained_reuse/R2_FACTORIZED_REPRESENTATION_RESULTS_20261015.md).

**Matched conditional-contrast comparison completed:** Two fresh seed17 fits
completed 512 updates each on the same new states, replay and budget. The added
objective worsens focus+new-description Accuracy 23.14%→17.38% and
complement+new-description 97.27%→96.39%; both trained endpoints fail retention.
Selected step is 0 in each run. All 86 full-transfer replay files match exactly.
Close this auxiliary-loss comparison; reassess conditional-language representation
and whole-model parameter allocation before another fit. No product promotion.
[Learning curves, uncertainty, error attribution and hashes](docs/pretrained_reuse/R2_CONDITION_CONTRAST_RESULTS_20261015.md).

**Next execution strategy:** Preserve the native-input/JSON baseline; prioritize
conditional-language suitability and retention, then real input grounding,
temporal/joint evidence and integration before quantization.
The fixed teacher factorial diagnostic is complete: binding81.25%→59.38% with
display order alone,74.22% with label mapping alone,42.97% with both. Numeric
remains52.73–54.69%. All768 historical logits and48 planned replay outputs match.
Close this teacher-interface search; no training or product promotion in this trial.
[Concise strategy](docs/pretrained_reuse/NEXT_STRATEGY_20261015.md) ·
[All factorial results and limits](docs/pretrained_reuse/R2_TEACHER_LABEL_FACTORIAL_RESULTS_20261015.md).

**Current strategy:** strengthen conditional semantics and retention coverage,
then validate grounded temporal/multimodal transfer before quantization.
A fixed instruction-diversity comparison improved new focus+description decisions
31.15%→81.64%, but complement+description decisions remained6.45% and old-task
retention failed. Do not adopt the new final weights; keep the earlier KL512
reference. All resident <=200M and four-modality95–98% remain unmet.
[Strategy and stop rules](docs/pretrained_reuse/STRATEGY_20261015.md) ·
[Fixed learning comparison and limits](docs/pretrained_reuse/R2_BINDING_LANGUAGE_RESULTS_20261015.md).

**Teacher prerequisite measured:** Pinned Gemma4 BF16,1,152 TRAIN prompts/64states,
no fitting: explicit/switch98.44%, exclusion81.25%, reversed exclusion42.97%,
numeric54.69%. Both option-order gates fail;36 predeclared replay outputs are exact.
Do not distill this fixed nonthinking label-logit interface. Separate condition
meaning from option-label/position handling before another fit; this is not a model
ceiling or product result. [Evidence and revised strategy](docs/pretrained_reuse/R2_SEMANTIC_TEACHER_RESULTS_20261015.md).

**Replay coverage comparison completed:** At the same512-update budget, replacing
half of original replay with curriculum improves curriculumNLL .4481→.3588,
but originalECE breaches the unchanged retention limit. Observed focus+description
Accuracy falls81.64%→76.27%; complement+description rises6.45%→15.43%.
Keep both research outputs, select0 within this trial, and preserve the earlier
KL512 reference. Verify conditional-semantic teachers on TRAIN before another fit;
no ratio sweep, promotion or quantization. [Full comparison](docs/pretrained_reuse/R2_BINDING_REPLAY_COVERAGE_RESULTS_20261015.md).

**2026-10-15 visual-state diagnosis completed:** A fixed training-derived visual
state preserves every joint answer; the ordered joint GRU is nearly saturated.
Video-only remains order-sensitive (48.01%→21.18% on token reversal). Close this
fusion capacity/budget route and require genuinely joint tasks; no promotion.
[Frozen-model controls, fresh replay and limits](docs/pretrained_reuse/R2_LOMBARD_VISUAL_STATE_RESULTS_20261015.md).

**2026-10-15 native temporal comparison completed:** Five fixed CPU fits and
fresh checkpoint replay pass. Ordered visual readout improves video29.85%→48.01%.
Joint scores99.53%, but mismatched video changes no answers: matched AV semantics
remain unproved, so no Omni promotion. [All curves, controls and limits](docs/pretrained_reuse/R2_LOMBARD_TEMPORAL_RESULTS_20261015.md).
Bare-finger's fixed notebook still leaves the audio-channel conflict unresolved;
[primary-source clarification](docs/pretrained_reuse/R2_BAREFINGER_PRIMARY_CLARIFICATION_20261015.md).

**2026-10-15 accessible contact-source follow-up executed:** Bare-finger source
accounting verifies500 trials/200 shared surface photos. One complete native
trial/image and fresh seven-array replay pass, but paper/reader audio-column
semantics conflict; independent splits and typed source admission remain open.
No RGB video or new fitting. [Actual source and native evidence](docs/pretrained_reuse/R2_BAREFINGER_SOURCE_RESULTS_20261015.md).

**2026-10-15 public native-input follow-up executed:** Fixed real/silence ASR gives
identical text, so RoboMNIST's acoustic counter remains unknown. VisTouch's full
metadata audit finds24 source sessions behind10,498 overlapping stretched clips,
no validation and approximate video timing. Native access/quality remain open;
no new fitting or promotion. [Acoustic evidence](docs/pretrained_reuse/R2_ROBOMNIST_AUDIO_CUE_RESULTS_20261015.md) ·
[Contact-material candidate and data gates](docs/pretrained_reuse/R2_VISTOUCH_SOURCE_RESULTS_20261015.md).

**2026-10-15 native frozen visual frontend executed:** One fixed RoboMNIST pilot
preserves all decoder pixels and TinyCLIP features after1MiB container padding,
while both fixed length-only classifiers change predictions. Fresh model/native
reload passes. This is pixel-processing evidence, not decision/temporal/joint
quality. [Measured controls and remaining gates](docs/pretrained_reuse/R2_ROBOMNIST_CONTAINER_FEATURE_RESULTS_20261015.md).

**2026-10-15 public-source shortcut diagnostic executed:** All2,135 RoboMNIST
groups stay intact in global repetition-block development splits. Camera file
lengths alone reach65.45% activity60 /85.30% robot identification; this is source
shortcut potential, not native Omni quality. Fresh full-array/scalar-metric replay
passes. Preserve this control before any new native fit.
[All conditions, calibration and limitations](docs/pretrained_reuse/R2_ROBOMNIST_LENGTH_RESULTS_20261015.md).

**2026-10-15 public native joint inputs executed:** A fixed RoboMNIST pilot
fetches only seven audio/video/CSI files in186s. All27 native arrays match a
fresh raw-file replay; 450 frames per camera and450×256 CSI points per module.
Common clock, split admission and Decision quality remain unverified; no fitting
or GPU restart. [Measured formats, hashes and next gates](docs/pretrained_reuse/R2_ROBOMNIST_INPUT_RESULTS_20261015.md).

**2026-10-14 complete public joint-input inventory executed:** All60 RoboMNIST
ZIP directories finish in401s with4.69MB response bodies; independent lexical
replay verifies2,135 repetition groups/all17,080 source files. All camera/mic/CSI
views stay together and positionGT is supervision-only. No media payloads, fit,
GPU restart, split admission or joint/OOD quality claim.
[Actual counts, source binding and next gates](docs/pretrained_reuse/R2_ROBOMNIST_DIRECTORY_RESULTS_20261014.md).

**2026-10-14 new component/source checks and incomplete memory run:** The separate
4h native comparison stopped at3,585.95s on its8GiB disk guard; six complete arms
and a partial49-seed arm are saved/CPU checked, with no quality scoring or restart.
RTDETRv2-R18 (20.17M) passes frozen CPU/fresh-model mechanics on one train image.
RoboMNIST's CC BY4.0 source/first ZIP-directory audit identifies34 grouped
audio/video/CSI repetitions; payload, split and joint-quality gates remain open.
[Interruption](docs/pretrained_reuse/R2_NATIVE_MASK_MEMORY_INTERRUPTION_20261014.md) ·
[Detector](docs/pretrained_reuse/R1_RTDETR_V2_R18_CPU_RESULTS_20261014.md) ·
[Public source](docs/pretrained_reuse/R2_ROBOMNIST_SOURCE_20261014.md).

**2026-10-14 native mask-memory smoke executed:** Public cache32 matches native
cache1 exactly on2automatic seeds/32frames; independent192-output audit passes.
Full696seed/two-arm projection9,945s exceeds the fixed7,200s gate, so main is
not started. No fresh full replay or tracking/Decision quality claim.
[Measurements, failed admission and limits](docs/pretrained_reuse/R2_NATIVE_MASK_MEMORY_SMOKE_RESULTS_20261014.md).

**2026-10-14 original QA resolution comparison executed:** Both fresh heads finish
50 epochs / 500 updates on 79 fit / 20 selector TRAIN scenes. Best epoch 8 Accuracy
is 25% for both; final coarse/fine Accuracy is 45%/32.5%, with strong overfitting.
Full saved-output replay and independent metrics/accounting audit pass. Close this
resolution-only fit; require semantic identity/temporal evidence before another fit.
No blind result, product promotion or full Omni quality claim.
[All curves, controls, artifacts and limits](docs/pretrained_reuse/R2_FINE_QA_GROUPED_RESULTS_20261014.md).

**2026-10-14 actual fine inputs + fixed support executed:** All99train clips retain
32×32spatial cells with full coarse regression, first-clip fresh replay and audits.
Same-rule hidden proxy support19→1,957pairs/5→34scenes; same-hidden gap8=30,endpoint0.
This is input/supervision coverage, not Decision quality; no training or promotion.
[Execution, all scenes and remaining gates](docs/pretrained_reuse/R2_FINE_SPATIAL_AND_ROLE_RESULTS_20261014.md).

**2026-10-14 all99 train scenes checked:** Native grid supervision corpus and
independent complete replay/audit pass, but only19 same-hidden pairs across5scenes.
Keep all zero-support scenes; more pair rows do not justify hidden-memory fitting.
No new model training or QA/Omni quality claim.
[Actual support and limitations](docs/pretrained_reuse/R2_GRID_ROLE_CORPUS_RESULTS_20261014.md).

**2026-10-14 correspondence learning executed:** 16fixed CPU fits×512steps on
5,400 weakly proxy-labelled pairs/8knowntrain clips, scene-held-out. Appearance
MLP memorizes train100%; macro NLL1.875 vs geometry0.536 and worse in all8folds.
Reject the endpoint; all curves/controls/weights retained, no Omni quality claim.
[Actual metrics, overfit and limits](docs/pretrained_reuse/R2_MASK_PAIR_ASSOCIATION_RESULTS_20261014.md).

**2026-10-14 foreground corpus + semantic supervision audit:** Eight known train
clips/all18,441 regions now have mean/std appearance; all72 fresh arrays exact.
Training source-box overlap: hidden41/243 available times; hidden+cover30 times,
20 from oneclip. This is geometry/input evidence, not Decision quality.
[Feature bank](docs/pretrained_reuse/R2_NATIVE_MASK_FEATURE_CORPUS_RESULTS_20261014.md) ·
[Role opportunity and limits](docs/pretrained_reuse/R2_NATIVE_ROLE_OPPORTUNITY_RESULTS_20261014.md).

**2026-10-14 native wider inputs completed:** Eight known train clips×32times,
256 full banks/39 reused/217 new, all18,441 NMS proposals retained.
Fixed one-frame fresh replay and full geometry/source-RGB/hash audit passed.
Input admission only; semantic binding and Omni quality remain open.
[Actual completion and limits](docs/pretrained_reuse/R2_NATIVE_DYNAMIC_CORPUS_FULL_RESULTS_20261014.md).

**2026-10-14 mask-region input probe:** One known train clip,32 times/all1,070
NMS candidates,384 mean+384 standard-deviation channels: full9-array fresh replay
exact; independent numerical reference8 rows exact. CPU primary9.74s,2.98MB,
0 added parameters. Input integrity only; semantic/Omni quality remains open.
[Executed scope and limits](docs/pretrained_reuse/R2_MASK_FEATURE_POOL_RESULTS_20261014.md).

**2026-10-14 native saved-prefix audit:**59 saved train frames/60,416 raw
candidates passed independent mask→box/NMS/grid reconstruction and source RGB
hash checks. Two media assets only;4,969 NMS survivors are not unique objects.
Main continues unchanged; full256-frame/fresh-replay admission and quality remain
pending.
[Actual scope and saved audit](docs/pretrained_reuse/R2_NATIVE_DYNAMIC_CORPUS_PREFIX_AUDIT_20261014.md).

**2026-10-14 native wider-input smoke:** Full1,024 candidates saved at two fixed
train times; frame0's five arrays match the old archive, while objects64–127
match cached/fresh sessions at both times. Main extraction
actually started for eight known train clips×32times (39 records reused,217 new).
Full input admission/semantic quality are not complete; follow the live process.
[Smoke evidence and continuation gates](docs/pretrained_reuse/R2_NATIVE_DYNAMIC_CORPUS_SMOKE_20261014.md).

**2026-10-14 fixed SAM2 chunk matrix:** Same64-object chunks16/32/64 all
retain exact masks/boxes/seeds/NMS but exceed the fixed presence-score guard.
Fresh16-array replay exact; cached forward ratios4.50–5.01×, no adoption,
quality or end-to-end claim. Close batch-width search and use native reference
or a separately fixed semantic-input protocol.
[Results and retained failures](docs/pretrained_reuse/R2_SAM2_CHUNK_MATRIX_RESULTS_20261014.md).

**2026-10-14 cached SAM2 batch measurement:** Matched64-object forward median
0.932s→0.185s (~5.04×); discrete prefix masks/boxes/seeds/NMS unchanged,8-array
fresh replay exact. Presence-score error1.335e-5 exceeds the fixed1e-5 guard:
no corpus replacement/adoption or full-grid/quality claim. Preserve the numerical
failure before broader cue/memory data work.
[Actual windows, guards and limits](docs/pretrained_reuse/R2_SAM2_BATCH_DECODER_RESULTS_20261014.md).

**2026-10-14 spatial relation diagnostic:** Same-time motion helps later group-box
association, but nearest-distance hidden-cue→source-cover correspondence is0/18
on the known one-scene train diagnostic. The cue track is in its source bbox2/18.
All controls/unknowns retained,18-array fresh replay and independent audit passed.
Close this simple distance policy; semantic binding/memory and Omni quality remain
open. [Actual comparison and limits](docs/pretrained_reuse/R2_TIMED_SPATIAL_RELATION_RESULTS_20261014.md).

**2026-10-14 time-specific mask tracking:** All1,070 model candidates were tracked
from their own original times over the full558-frame train regression clip.
Primary63.02s/5.45GiB Torch; fresh replay matches all15 arrays exactly. No GT
selection or training. This establishes input routing; semantic target/container
association, native joint quality and product latency remain unqualified.
[Executed primitive and limits](docs/pretrained_reuse/R2_TIMED_MASK_TRACKING_RESULTS_20261014.md).

**2026-10-14 temporal discovery executed:** The fixed train regression scene
completed all32 times/32,768 proposals, with exact cache admission and last-time
reload. A later exposed-item cue has a model-produced mask; source-box opportunity
does not select that item or preserve its container relationship. No quality fit
or adoption. [Actual costs, cue scope and limits](docs/pretrained_reuse/R2_DYNAMIC_DISCOVERY_RESULTS_20261014.md).

**2026-10-14 train-only appearance association:** Moving-position cosine retrieval
beats static sampling on eight reused scenes, but trails original track routing
on every clip. This is a sparse-box geometric proxy, not Decision Accuracy.
Preserve the negative comparison; no nearest-descriptor adoption or new fit.
[Raw matches, controls and remaining gates](docs/pretrained_reuse/R2_TRACK_ASSOCIATION_RESULTS_20261014.md).

**2026-10-14 track-aligned appearance:** All256 model tracks×32times on eight
training clips passed pinned official sampling and exact fresh-process replay.
All48 outputs remain outside Git. This establishes input routing, not object
identity or Decision quality; no fitting or added parameters.
[Measurements and remaining gates](docs/pretrained_reuse/R2_TRACK_APPEARANCE_RESULTS_20261014.md).

**2026-10-14 role-supervised Video QA comparison completed:** Both paired50-epoch
runs retained all514 development questions. Accuracy38.91%→40.27% has a95% interval
including no gain; changed-target pairs both-correct0/127, and temporal-zero control
is better. Preserve the negative/mixed result; no adoption or epoch extension.
[Full curves, controls, exact replay and costs](docs/pretrained_reuse/R2_ROLE_SUPERVISED_QA_RESULTS_20261014.md).

**2026-10-14 later-appearance retention:** Frozen TAPIR saved full spatial maps
at32 fixed times on eight known train clips, with exact fresh-process reload.
Initial descriptor parity passed; later numerical variation is not target
recognition or Decision quality. No additional training or model parameters.
[Execution, resource cost and remaining semantic gate](docs/pretrained_reuse/R2_TEMPORAL_APPEARANCE_20261014.md).

**2026-10-14 object-temporal execution:** All696 model-mask queries tracked
4,979 train frames with unchanged frozen weights and exact checkpoint/RNG replay.
Initial group opportunity24/24 fixes a grid gap; GT-associated future point-in-box
371/500 is diagnostic, not QA Accuracy. Later appearance/target association and
full Omni quality remain open. [Executed state primitive and limits](docs/pretrained_reuse/R2_MASK_TEMPORAL_RESULTS_20261014.md).

**2026-10-14 frozen mask-ranking comparison:** Existing TinyCLIP actually ranked
all696 saved SAM2 NMS candidates. Fixed top3 distinct group coverage10/24→9/24;
cover hits2/8→3/8 do not establish improvement. Preserve the negative result;
no static-ranker adoption, QA quality claim or prompt sweep.
[Execution, controls and limits](docs/pretrained_reuse/R2_TINYCLIP_MASK_RANK_20261014.md).

**2026-10-14 source-cue audit:** Pinned training annotations distinguish hidden
items, covering objects and variable-sized candidate groups on all eight pilot
clips.64 raw frames were decoded/reviewed; input-only target association,
game-time anchors and actor-relative geometry remain unmeasured. No new fit or
quality claim. [Executed source/frame audit](docs/pretrained_reuse/R2_PERCEPTION_SOURCE_CUES_20261014.md).

**2026-10-14 annotation-free initialization probe:** SAM2 generated all8,192
geometric-grid proposals on eight fixed train clips and seeded its own top3
masks. Covering-object bbox opportunity8/8 survives NMS, but top3 coverage2/8.
This is geometry/ranking diagnosis; hidden-item/QA target mapping remains open.
[Executed probe and semantic limits](docs/pretrained_reuse/R1_SAM2_AUTO_INIT_20261014.md).

**2026-10-14 object-memory candidate admission:** SAM2.1 Tiny actually loaded and
processed four train clips×16frames on the existing GPU.38,962,498 parameters,
~695MB peak Torch allocation; repeat/reload and frozen-state checks passed.
Fixed geometric prompting only; automatic discovery/Decision quality unmeasured.
[Executed R1 audit and remaining gates](docs/pretrained_reuse/R1_SAM2_TINY_20261014.md).

**2026-10-14 native AV comparison completed:** All5,367 native feature extractions
and three fixed24-epoch runs passed integrity/reload checks. Audio-only macro slot
Accuracy97.94%, video30.15%, joint97.09%; joint worsens NLL/Brier and Accuracy versus
audio. Retain the narrow audio research reference; no pooled-fusion adoption or
generic Omni95–98% claim. Historical pending entries below describe earlier stages.
[Full metrics, curves, controls and limits](docs/pretrained_reuse/R2_LOMBARD_GRID_RESULTS_20261014.md).

**2026-10-14 native decision smoke:** Fixed51-option six-slot readouts passed
two train-only GPU updates for audio/video/joint, with identical initialization/
sampling and exact reload/JSON checks. No heldout score; full24-epoch learning
awaits the live native visual extraction.
[Executed smoke and pending full comparison](docs/pretrained_reuse/R2_LOMBARD_GRID_READOUT_SMOKE_20261014.md).

**2026-10-14 native AV extraction running:** The original CPU VideoPrism
actually started the fixed5,367-video job after exact native-reference smoke.
Its final integrity/reload checks and joint quality comparison are pending;
the three-arm/full-vocabulary protocol is frozen before readout metrics.
[Execution snapshot and fixed comparison](docs/pretrained_reuse/R2_LOMBARD_GRID_VISUAL_FEATURES_20261014.md).

**2026-10-14 visual backend audit:** Frozen native16-frame Torch/CUDA port
is faster but fails the predeclared full-token numerical gate. Keep the native
CPU extraction path; no quality result or artifact promotion.
[Negative parity experiment](docs/pretrained_reuse/R1_VIDEOPRISM_TORCH_20261014.md).

**2026-10-14 native AV data:** Licensed Lombard Grid speech/video archives are
acquired and audited. Fixed common train4,080/calibration398/validation889 retains
all held-out IDs; four broken training videos and19 unresolved labels remain
documented. A frozen8.21M speech Encoder passed real CPU/CUDA feature checks.
Native joint command quality remains unmeasured.
[Source integrity and next execution](docs/pretrained_reuse/R2_LOMBARD_GRID_MEDIA_20261014.md).

**2026-10-13 CLAP comparison completed:** Same-data joint Accuracy65.18%
versus EfficientAT66.96%, with worse NLL/Brier point estimates and clear later
overfitting. Full-clip extraction, train-only PCA and reload checks passed.
No replacement/promotion; this is reused development validation.
[Results and limits](docs/pretrained_reuse/R2_CLOTHO_CLAP_RESULTS_20261013.md).

**2026-10-13 conditional-audio comparison completed:** All856 licensed sources
and180,004 cross-split waveform pairs were checked before fixed24-epoch learning.
Joint Accuracy66.96% versus question-only64.82% has an uncertain2.14-point gain;
NLL/Brier are slightly worse. Joint predictions depend on matching audio, but
final24 strongly overfits. Keep this as an experiment; no product adoption or
epoch extension. All3,422 questions and annotation disagreements were retained.
[Measured results, learning curves and limits](docs/pretrained_reuse/R2_CLOTHO_AQA_RESULTS_20261013.md).

**2026-10-13 audio coverage comparison:** A fixed128-clip licensed FSD50K
positive-only augmentation completed with the frozen audio encoder. Validation
Accuracy fell89.97%→89.37%; nearly unchanged aggregate NLL hides worse music
performance and zero bell recall. Not adopted; all prior artifacts are retained.
[Measured results and learning curve](docs/pretrained_reuse/R2_FSD_POSITIVE_RESULTS_20261013.md).

> **Current user constraints (2026-10-12):** The finished model must have at most
> **200M total resident parameters before quantization**, covering text, image,
> audio and video (sensors optional), and target **>=95% Accuracy for each modality,
> aspiring to98% or higher**, with calibrated supplied-option probabilities and
> general condition following. These supersede earlier provisional parameter
> wording below. Narrow taxonomy/ASR or aggregate Accuracy cannot satisfy this
> requirement. No current result establishes completion; historical records remain.

Tiny Omni Decision adapts Gemma 4's multimodal representation to return probabilities over supplied choices. Phase 1 established a **text-only decision** path; the current training pipeline supports text, image, audio, and video. The decision readout scores supplied choices and returns a probability distribution instead of generating a response.

The current pretrained-representation research follows the R0–R8 process in [ROADMAP.md](ROADMAP.md). Frozen-feature text, synthetic image, CLEVRER video (including a binary explanatory/predictive/counterfactual readout, a new-scene text/video control, and a temporal-order reversal control), and closed-set English keyword audio probes are recorded in [docs/pretrained_reuse/R2_PATH_A_PROBES.md](docs/pretrained_reuse/R2_PATH_A_PROBES.md). A post-hoc Speech Commands control shows that the saved Audio+Text readout depends on its audio features, but the selected validation is not blind and the task remains narrow ([report](docs/pretrained_reuse/AUDIO_MULTIQUERY_FEATURE_DEPENDENCE_20261011.md)). A full-coverage static CLEVR-4 image probe with frozen VideoPrism reached 87.9% on the previously used 512-image selector and 88.1% on a one-time 1,595-image validation subset; a packaged local replay matched 128 saved decisions across 32 images. The synthetic task, changed encoder and expanded training coverage limit the quality comparison; the runtime bundle is 458 MB and CPU-only replay used about 4.26 GB peak working set. The CLEVRER controls found no clear scene-specific video contribution or temporal-order sensitivity for the current small readout; this is not a backbone capability claim. A dispatcher now replays eight saved assets per modality and records latency distributions, parity, caching, and memory ([report](docs/pretrained_reuse/R2_PATH_A_LATENCY_DISTRIBUTION_20261011.md)). Joint-modality quality remains pending. These development probes are not a product Teacher or release result.

The Gemma 4 material below documents the legacy implementation. The pretrained-representation research direction is tracked in [ROADMAP.md](ROADMAP.md); the isolated inventory, pinned checkpoint audits, evaluation protocol and frozen-feature probes are recorded under [docs/pretrained_reuse](docs/pretrained_reuse/). These are research prototypes, not a complete multimodal Decision pipeline or selected product model.

## Pinned base

- Model and processor: `google/gemma-4-E2B-it-qat-q4_0-unquantized`
- Immutable Hugging Face revision: `6befbaca7398925921802abd1f277b495b78b738`
- Architecture in upstream `config.json`: `Gemma4ForConditionalGeneration`
- Processor/tokenizer in pinned metadata: `Gemma4Processor` / `GemmaTokenizer`
- Upstream license metadata: Apache-2.0; preserve Google DeepMind attribution and follow [Gemma 4 terms](https://ai.google.dev/gemma/docs/gemma_4_license) when distributing derived artifacts.
- HF API metadata reports 5,104,297,539 BF16 parameters. The pinned safetensors header has 5,104,298,467 total elements, including 928 scalar QAT metadata entries. Transformers 5.6.2's config-only model, after tying shared embeddings, counts 5,104,297,504 parameters (35 fewer than the API); see the manifest. The unquantized QAT weights occupy about 10.2 GB on disk; normal training also requires activations, gradients, and optimizer state.

The immutable provenance is in [manifests/base-model.example.yaml](manifests/base-model.example.yaml). Model weights are downloaded from Hugging Face and are never committed here. The tokenizer and processor use the same pinned revision.

## Setup

Python 3.11 is supported. For CPU-only schema/loss tests:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install ".[dev]"
```

For model loading and CUDA training, install the ML extra in an environment with a PyTorch CUDA build matching your driver:

```bash
python -m pip install ".[ml,dev]"
```

Gemma 4 multimodal attention masking currently requires PyTorch 2.6 or newer;
the supported ML extra therefore pins PyTorch 2.6.x and matching torchvision
0.21.x. On Windows with a CUDA 12.4-compatible driver, the official wheel command
is `python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url
https://download.pytorch.org/whl/cu124`.

The pinned model is public. Do not put Hugging Face tokens or other credentials in repository files. Fresh-clone CPU preflight:

```bash
tiny-omni-decision validate-model-manifest manifests/base-model.example.yaml
tiny-omni-decision validate-dataset-manifest manifests/dataset.example.yaml
tiny-omni-decision preflight
ruff check .
pytest -q
```

## Inspect the actual model

This loads the approximately 10.2 GB checkpoint and prints the instantiated architecture, parameter count, module names, candidate linear LoRA targets, modality-related modules, dtypes, memory estimate, and processor/tokenizer classes. Use `--config-only` to instantiate the pinned architecture on `meta` without downloading weights; this does not verify checkpoint tensor shapes:

```bash
tiny-omni-decision inspect-model
tiny-omni-decision inspect-model --config-only
```

No guessed module names are used in the config. The smoke selects exact decoder module paths after loading and verifies that PEFT adapted those paths only.

## Text decision smoke training

The explicit GPU command runs a synthetic one-sample forward pass, CE + Brier loss, backward, optimizer step, LoRA save, base unload/reload, adapter reload, and probability inference:

```bash
tiny-omni-decision smoke-text-decision --seed 17
```

The command refuses CPU execution, guessed target modules, non-finite loss, missing adapter gradients, or gradients on frozen base parameters. It saves the adapter and `run-metadata.json` under `artifacts/smoke-text-decision/`, including base revision/config and seed.

### Hardware expectation

The Phase 1 text smoke was run on an NVIDIA GeForce RTX 3080 Laptop GPU with
16,384 MiB VRAM, PyTorch 2.5.1+cu121, Transformers 5.6.2, and PEFT 0.21.2.
Gemma 4's multimodal path requires PyTorch 2.6. The earlier bounded pipeline
check and the durable teacher run are recorded in
[docs/PHASE3.md](docs/PHASE3.md) and
[docs/DURABLE_TEACHER.md](docs/DURABLE_TEACHER.md).

## Durable Decision Teacher

The reproducible train, validation, and held-out evaluation corpus can be frozen
with:

```bash
tiny-omni-decision freeze-corpus \
  manifests/durable-training-corpus.yaml \
  manifests/durable-heldout-corpus.yaml \
  data/processed/durable-teacher-v0 \
  --max-records-per-source 2048
```

Then run the four-modality Decision LoRA training, validation selection, and
full held-out evaluation:

```bash
tiny-omni-decision train-decision \
  --train-manifest data/processed/durable-teacher-v0/train.jsonl \
  --eval-manifest data/processed/durable-teacher-v0/eval.jsonl \
  --validation-manifest data/processed/durable-teacher-v0/validation.jsonl \
  --config configs/decision/durable_teacher.yaml \
  --output artifacts/tiny-omni-decision-teacher-v0
```

Corpus provenance, hashes, split checks, training setup, measured metrics,
limitations, artifacts, and the later merge/export path are in
[docs/DURABLE_TEACHER.md](docs/DURABLE_TEACHER.md). The earlier 16-step
pipeline test is retained as historical context in [docs/PHASE3.md](docs/PHASE3.md).

## Planned student, compression, and recovery flow

**Next experiment: ternary first, recover with the Teacher, add LoRA only if recovery is insufficient.**

```text
Gemma 4 Decision Teacher: finish → select → reload → freeze
                                              │
                                              │ supervises recovery
Pretrained EmbeddingGemma 2                    │
  → aggressive ternary conversion             │
  → measure initial damage                    │
  → ternary-constrained QAT + distillation ←───┘
  → if sufficient: export / reload / final evaluation / runtime
  → if insufficient: freeze quantized base
       → add one Recovery LoRA and train it with the same Teacher
       → export / reload / re-evaluate
```

Do not first train a separate high-precision decision student. Gemma 4 remains both the source of option-level supervision and the quality reference. The existing Teacher run and artifacts are unchanged; student recovery is planned, not yet demonstrated.

Target the student's large text/backbone, embedding, vision, and audio weights for ternary conversion after pinning and inspecting the actual model. Record every higher-precision exception. During the first recovery stage, each forward uses ternary-quantized target weights even when gradient updates use higher-precision shadow weights.

If that recovery is insufficient, freeze the selected quantized base and train one separate Recovery LoRA. Count its bytes and runtime overhead. Do not silently merge it into the ternary base: a merged correction is generally no longer ternary. The approximately 1.58-bit packing goal is not a measured whole-model size or a performance guarantee.

Keep native backbone/pooling initially and define only the minimal supplied-option readout. The earlier pooling-plus-small-classifier structural fallback is deferred until constrained recovery and LoRA are insufficient; it is not a mandatory stage before quantization. Inspect actual modules before any terminal-path removal.

Reuse compact Teacher option signals, matching the actual choices and their order rather than assuming shared vocabulary IDs. Training, validation selection, and final held-out evaluation remain separate. The detailed sequence is in [ROADMAP.md](ROADMAP.md).

## Current boundary and limitations

- CPU CI covers schema, manifest, token-label mapping, probability normalization, Brier loss, and option reordering. It does not download model weights.
- The ML extra follows the model card's documented Transformers minimum (`>=5.6.2`) and requires PyTorch 2.6 for actual Gemma 4 multimodal forward. Install a wheel matching the local CUDA driver; GPU model loading is not covered by CPU CI.
- The durable Teacher run uses frozen text, image, audio, and video corpora with zero pairwise source-ID and normalized-content overlap. Clevr-4 (CC BY 4.0), Speech Commands (CC-BY-4.0), and CLEVRER (CC0) provide controlled multimodal candidates; OneJev remains excluded pending component-level rights review. The dataset is sampled and the training budget is capped, so the results are not benchmark-generalizing quality claims; see [docs/DURABLE_TEACHER.md](docs/DURABLE_TEACHER.md).
- The current Gemma 4 vocabulary readout requires option labels to tokenize to exactly one distinct token each; it fails closed otherwise. This does not imply a shared Teacher/student tokenizer.
- The synthetic smoke example is a plumbing check, not a quality or calibration evaluation.
- GitHub Actions PR checks validate proposed commits; push checks on `main` validate the resulting merge commit. Both run install, lint, CPU tests, and manifest validation without downloading model weights. The separate 16 GB GPU smoke was run locally.
- OneJev component-level rights review remains unresolved and excluded. EmbeddingGemma 2 revision pinning, integration, ternary-constrained recovery, optional LoRA recovery, and packed runtime support remain open. No student recovery or student-size result is claimed.
- This is a documentation-only plan change. Existing training code, configs, manifests, and checkpoints are unchanged. The Gemma 4 quantization/recovery configs do not implement this new student path.

See [ROADMAP.md](ROADMAP.md), [docs/PHASE0.md](docs/PHASE0.md), and [THIRD_PARTY.md](THIRD_PARTY.md).


### STARSS23 measured update (2026-10-11)

All 78 approved paired recordings and 2,720 five-second windows were processed
with frozen audio/video encoders. The preregistered readout comparison completed:
audio-only validation NLL 0.2704, video-only 0.4108, joint 0.3290. Joint input
did not improve audio-only; zeroing joint video improved NLL to 0.2978.
This closes the initial paired-data execution gap, but not the joint-quality
gate. Two validation rooms, class imbalance and event-specific losses limit
interpretation. See [STARSS23 comparison](docs/pretrained_reuse/R2_STARSS23_COMPARISON_20261011.md)
for metrics, curves, controls and artifact hashes. No product promotion,
compression, additional seed or sealed evaluation followed.

### Bounded text adaptation result (2026-10-12)

Approved final-block adaptation of a 22.7M Cross-Encoder completed three epochs.
On the same 2,540 development questions, Accuracy rose from zero-shot 45.98%
to 58.11%, with NLL 1.19869 -> 0.93779. Only 1.775M existing parameters were
updated; frozen hashes and best reload checks passed. Calibration worsened
slightly in the last epoch. This is synthetic English text evidence, not an
Omni quality or quantized deployment result. See
[bounded adaptation report](docs/pretrained_reuse/R2_TEXT_CROSSENCODER_ADAPT_20261011.md).

### Conditional text limitation (2026-10-12)

The fixed adapted Cross-Encoder failed the approved64-state numerical condition
probe:35.94% Accuracy, always selecting B, with0/128 answer changes when switching
lowest/highest. No input truncation occurred. The earlier58.11% synthetic corpus
result therefore does not qualify general conditional Decision behavior. See
[condition probe](docs/pretrained_reuse/R2_TEXT_CONDITION_PROBE_20261012.md).
No additional training or product promotion followed.

### Full-coverage conditional repair (2026-10-12)

The fixed 2,048-update run completed with 16,384 distinct questions and exact
best reload. Best mean-NLL checkpoint was step768. Curriculum Accuracy changed
24.37% at512 to23.14% at2048; original Accuracy57.83% to57.28%. Slight NLL gains
did not establish useful conditional reasoning. No95% claim or product promotion.
See [full curve and uncertainty](docs/pretrained_reuse/R2_CONDITIONAL_FULL_COVERAGE_20261012.md).

### Encoder-coverage comparison (2026-10-12)

Updating all six existing22.7M-model encoder blocks, with embeddings/pooler frozen,
completed the same2,048-update protocol. Curriculum Accuracy80.57% versus23.14%
for final-block-only; original Accuracy60.24% versus57.28%. Best reload and source
hash checks passed. This is synthetic development evidence; count/conditional
selection remain weak and no four-modality95% claim follows. See
[full curve and controlled comparison](docs/pretrained_reuse/R2_ENCODER_COVERAGE_20261012.md).

### Transfer limitation (2026-10-12)

The selected encoder-coverage weights scored33.59% on the unchanged known-failure
256-question diagnostic, with10/128 fully correct opposite-condition pairs.
Thus80.57% on the training generator's validation does not establish general
Decision reasoning. No truncation occurred; no prompt or checkpoint search followed.
See [transfer evidence](docs/pretrained_reuse/R2_ENCODER_TRANSFER_20261012.md).

### Controlled transfer diagnosis (2026-10-12)

Fixed-weight, paired128-state comparisons found79.79% Accuracy in control,
25.88% after an order-preserving numeric-range shift,54.39% after paraphrasing,
and22.95% with all five perturbations combined. Identifier/field renaming had
small effects. This supports range/wording fragility under the tested conditions;
it does not isolate candidate-count effects or establish broad multimodal quality.
See [all arms, task metrics and uncertainty](docs/pretrained_reuse/R2_TRANSFER_CONTROLS_20261012.md).

### Numeric-range coverage result (2026-10-12)

The numeric-only augmentation run completed2,048updates and best reload. On the
predeclared new transfer set, selected-model Accuracy24.27% versus prior30.66%;
the confidence loss improved but useful reasoning did not. Original curriculum
Accuracy fell80.57%→24.51% at the same budget. No promotion or95%claim. See
[fixed-budget range experiment](docs/pretrained_reuse/R2_NUMERIC_RANGE_20261012.md).


The learned-initialization numeric-range control completed2,048updates: selected
narrow-curriculum validation82.23%, final expanded-range training fit83.50%,
with worse original-text NLL/calibration and extra inherited compute. No new
transfer/audit or multimodal quality claim. See
[record](docs/pretrained_reuse/R2_RANGE_WARMSTART_20261012.md).


A fixed selected-weight comparison on new synthetic states found range
interpolation improvement24.12%→71.00%, but wording regression56.54%→42.09%
and poor extrapolation calibration. No general-model promotion.
[Results](docs/pretrained_reuse/R2_RANGE_GENERALIZATION_RESULTS_20261012.md).


Lexical-range control completed2,048updates (selected1408); wording transfer
remains unmeasured for that checkpoint. Separately, frozen TinyCLIP scored48.19%
macro on CLEVR-4 vs9.81% with rotated images; count/texture remain weak.
[Training](docs/pretrained_reuse/R2_LEXICAL_RANGE_RESULTS_20261012.md) /
[image grounding](docs/pretrained_reuse/R2_TINYCLIP_CLEVR4_RESULTS_20261012.md).


The fixed lexical-transfer follow-up found only a small change on its novel
minimum/maximum wording (80.71%→81.05%), with numeric-calibration trade-offs.
It does not repair the previously observed smallest/largest failure by evidence.
[Results](docs/pretrained_reuse/R2_LEXICAL_TRANSFER_RESULTS_20261012.md).


A paired factorial text diagnosis confirms substantial comparison-word
sensitivity at fixed state/range/structure. Lexical training helps but leaves
smallest/largest far below minimum/maximum. No general-quality claim.
[Diagnosis](docs/pretrained_reuse/R2_COMPARATIVE_WORD_RESULTS_20261012.md).


Comparison-vocabulary training completed2,048updates (selected1664). On fixed new
states/templates, smallest/largest improves59.96%→82.81%, but domain terms remain
40.04% (expanded range34.08%). General four-modality95% remains unmet.
[Training](docs/pretrained_reuse/R2_COMPARISON_VOCABULARY_RESULTS_20261012.md) /
[transfer](docs/pretrained_reuse/R2_VOCABULARY_TRANSFER_RESULTS_20261012.md).


Frozen native Whisper transcript likelihood reaches99.95% on the same1,938
random/hard4-choice transcript questions; rotated audio falls to24.82%/22.03%.
No training or parameter increase; calibration and general spoken reasoning remain
open. [Native audio evidence](docs/pretrained_reuse/R2_NATIVE_AUDIO_RESULTS_20261012.md).


The frozen DINOv2 image baseline completed:85.06% across four synthetic taxonomies,
versus87.89% for the existing VideoPrism reference on the same512selection images.
Its39.59M image/text/readout stack is smaller, but count remains53.71%; no replacement
or general-image95% claim. [Results](docs/pretrained_reuse/R2_DINOV2_IMAGE_RESULTS_20261012.md).


The matched-capacity4x4spatial readout gives76.42% vs76.03% for its global control,
with worse NLL and no clear aggregate gain; both trail the smaller-head85.06%.
No adoption or backbone-limit claim. [Spatial control](docs/pretrained_reuse/R2_DINOV2_SPATIAL_RESULTS_20261012.md).

The 134.5M decoder completed2,048updates with exact best reload. Original text
validation improves30.91%→49.92%, but fixed new-state conditional transfer is23.44%
versus81.05% for the22.7M BERT reference; candidate-order consistency is weak.
No replacement, no four-modality95% claim.
[Learning and transfer results](docs/pretrained_reuse/R2_SMALL_DECODER_RESULTS_20261012.md).

Fixed candidate-position averaging on the134.5M decoder costs4.25forwards/question
and gives21.29%/24.71% for normal/reversed options; no consistent Accuracy gain.
Do not adopt or automatically extend training.
[Rotation diagnostic](docs/pretrained_reuse/R2_CANDIDATE_ROTATION_RESULTS_20261012.md).

The matched DINO spatial/global heads completed24epochs with exact reproduction
of the original first8epochs. Selected global81.35%, spatial82.62%; the paired
spatial gain remains uncertain and both trail the smaller-head85.06% reference.
Count remains46.88% for spatial. No adoption or automatic budget extension.
[Learning curves](docs/pretrained_reuse/R2_SPATIAL_CURVE_RESULTS_20261012.md).

At essentially equal readout size, bilinear image/candidate interaction improves
the frozen DINO reference85.06%→87.40% (paired512-image interval+1.03…+3.66pp),
with lower NLL/Brier. Count improves53.71%→59.77%; general image95% remains unmet.
Retain as a measured candidate, without product promotion or video-quality claims.
[Interaction comparison](docs/pretrained_reuse/R2_BILINEAR_RESULTS_20261012.md).

The image-successful bilinear readout did not clearly transfer to pooled video:
macro Accuracy52.24% vs concat52.02% and no-video51.93%, with differences uncertain.
Validation NLL degrades after selected epoch12 despite better train fit. Do not adopt
or extend pooled training. [Video controls](docs/pretrained_reuse/R2_VIDEO_INTERACTION_RESULTS_20261012.md).

The pinned VideoPrism unpooled-token audit passed on two training scenes:
FP32[8,16,16,768], exact reload/repeat and zero global-mean difference from the
old cache. Full456-scene storage would need2.87GB, exceeding current free disk;
no full extraction or quality claim. [Token audit](docs/pretrained_reuse/R2_VIDEO_TOKEN_AUDIT_RESULTS_20261012.md).

RAM transport now passes a full-size allocation check: 456 slots (2.87 GB),
with two real saved tensors transferred exactly and 454 synthetic zero slots.
No live full extraction or quality result yet.
[RAM transport measurements](docs/pretrained_reuse/R2_TOKEN_RAM_TRANSPORT_20261012.md).

The full456-scene token experiment completed three24-epoch comparisons:
attention52.44%, mean52.72%, no-video51.56% selected macro Accuracy.
Paired intervals do not establish an attention/video benefit; do not adopt or
automatically extend training. [Token readout results](docs/pretrained_reuse/R2_TOKEN_READOUT_RESULTS_20261012.md).

Training-only64-example diagnosis reaches100% with attention and97.68% without
video after512updates. This verifies small-subset fit, not generalization or
the product target. [Fit diagnostic](docs/pretrained_reuse/R2_TOKEN_TRAIN_FIT_RESULTS_20261012.md).

Frozen ViLT's full-option8-epoch image probe reaches81.10% versus DINO87.40%;
paired difference−6.30pp. Color improves, texture/count worsen; no replacement.
The curve is still improving, so no capacity-ceiling claim.
[ViLT comparison](docs/pretrained_reuse/R2_VILT_IMAGE_RESULTS_20261012.md).

ViLT24epochs reproduces its original8-epoch prefix exactly and improves to84.47%
(+3.37pp), still below smaller DINO87.40%; ECE worsens slightly. Retain DINO,
with no clean-plateau or general95% claim.
[Fixed learning curve](docs/pretrained_reuse/R2_VILT_CURVE_RESULTS_20261012.md).

DINO/bilinear24epochs reproduces its first8epochs and reaches89.26% (+1.86pp),
with lower NLL/Brier but higher ECE. Exact counting remains64.06% and accounts
for184/220 errors; no tolerance relaxation or general95% claim.
[DINO curve](docs/pretrained_reuse/R2_DINO_BILINEAR_CURVE_RESULTS_20261012.md).

Spatial/bilinear reaches90.04% on the reused image selector; matched global88.48%.
Counting improves to70.51%, but superiority over the smaller89.26% reference
is uncertain and ECE worsens. Preserve both; general four-modality95% remains unmet.
[Spatial comparison](docs/pretrained_reuse/R2_SPATIAL_BILINEAR_RESULTS_20261012.md).

Matched physical-mirror augmentation gives90.19% versus90.04% image Accuracy,
an uncertain3/2048-answer gain; NLL worsens and counting drops. No adoption.
[Controlled result](docs/pretrained_reuse/R2_MIRROR_CONTROL_RESULTS_20261012.md).

Shared nonlinear cell pooling reduces head size to180736 parameters but gives
89.79% selected image Accuracy; counting66.41% trails spatial bilinear70.51%.
Later training overfits; no counting-fix or product-quality claim.
[Cell readout result](docs/pretrained_reuse/R2_CELLWISE_RESULTS_20261012.md).

Numeric embedding updates pass a matched512-step run, but both candidate and
control select their unchanged initialization. No quality artifact promoted.
[Numeric comparison](docs/pretrained_reuse/R2_NUMERIC_EMBEDDING_RESULTS_20261012.md).

### Deterministic numeric-embedding comparison (2026-10-12)

Both fixed2048-update runs completed and selected step1664. Numeric embedding
updates improved the numeric curriculum but worsened original-text selected
Accuracy/NLL/Brier/ECE. Preserve both; no product adoption or95%claim.
See [learning curves, paired diagnostics and hashes](docs/pretrained_reuse/R2_NUMERIC_DETERMINISTIC_RESULTS_20261012.md).

### Shared text/image feature comparison (2026-10-12)

A same-head24-epoch pair completed. Shared Cross-Encoder features yield90.43%
versus90.04% image Accuracy (paired difference uncertain), improved count NLL
but worse color/texture metrics. No product adoption. Sharing is implemented
for the image probe only; video compatibility and the combined <=200M runtime
remain unverified. [Results](docs/pretrained_reuse/R2_SHARED_TEXT_IMAGE_RESULTS_20261012.md).

### Shared text/video comparison (2026-10-12)

All four fixed24-epoch runs completed; prior MiniLM curves reproduced exactly.
Shared text features yield49.79% video macro Accuracy versus50.18% without video.
Neither encoder shows a clear matched-video benefit; later shared training overfits.
No sharing adoption or combined-runtime quality claim. This closes the sharing
probe, not the <=200M / four-modality95% objective.
[Curves, uncertainty and artifacts](docs/pretrained_reuse/R2_SHARED_TEXT_VIDEO_RESULTS_20261012.md).

### Text error attribution (2026-10-12)

A read-only audit of all2540 saved original-text validation decisions finds950
of977 errors in untruncated inputs; score questions reach46.24% exact Accuracy.
Longer context and dismissing errors as label noise lack supporting evidence.
All questions retained; no training or new inference.
[Attribution and limits](docs/pretrained_reuse/R2_TEXT_ERROR_ATTRIBUTION_20261012.md).

### Broader text coverage continuation completed (2026-10-12)

Storage was freed and a separate2048-update run completed, selecting1792 by the
fixed validation rule. Original textAccuracy improves61.54% to63.58%; curriculum
81.54% to84.08%, with lower NLL/Brier/ECE. This is one reused-development-validation
result, not95% or a causal coverage-only claim. Existing multimodal towers remain
unchanged; selected/final artifacts and all curves are preserved.
[Results, uncertainty and accounting](docs/pretrained_reuse/R2_TEXT_COVERAGE_RESULTS_20261012.md).

### Coverage checkpoint fixed transfer (2026-10-12)

A separate frozen10-condition comparison completed on all10240 diagnostic rows
per model. The new checkpoint scores83.98–84.38% on trained-word base conditions,
73.54–74.61% on expanded ranges,40.14% on domain words and36.13% on expanded
domain-word inputs. Every paired Accuracy-difference interval includes zero;
expanded/domain probability quality worsens. No general-reasoning adoption.
[All metrics, opposite-condition pairs and limits](docs/pretrained_reuse/R2_COVERAGE_TRANSFER_RESULTS_20261012.md).

### Question-side failure isolation (2026-10-12)

An actual-file audit corrects the earlier field-name wording: vocabulary family4
never changes state keys. A fixed2x2 diagnostic on8192 rows/model separates the
alias prefix from the remaining domain wording. For the coverage checkpoint,
adding the prefix lowers generic baseAccuracy84.38% to53.03%; domain wording
without the prefix still scores48.83%. All legacy reference probabilities replay
exactly. Both weaknesses coexist; no template-specific fix or product adoption.
[Factorial results](docs/pretrained_reuse/R2_TRANSFER_FACTORIAL_RESULTS_20261012.md)
and [factual correction](docs/pretrained_reuse/R2_TRANSFER_FIELD_AUDIT_20261012.md).


### Compact pretrained NLI check (2026-10-12)

A pinned DeBERTa xsmall checkpoint has **70,830,722 actual parameters**, including
embeddings. On the unchanged 256-question numerical-condition diagnostic it
scored **30.47% Accuracy**, NLL 1.1430, Brier 0.7001 and ECE 0.2041;
opposite conditions changed no answers (0/128 pairs). No training or prompt
search was performed. This candidate is not adopted, and the result is not
a general capacity ceiling. The four-modality >=95% goal remains unmet.
[Load audit](docs/pretrained_reuse/R1_DEBERTA_XSMALL_20261012.md) and
[fixed diagnostic](docs/pretrained_reuse/R2_DEBERTA_NLI_RESULTS_20261012.md).


### Binary versus variable-option scoring

Saved selected-checkpoint predictions show why reduced-choice Accuracy cannot
replace the full task: the 1,536 four-option curriculum questions have 90.89%
full-choice Accuracy but 96.64% gold-versus-distractor pair wins. Pairs are
constructed using known answers, not an available inference policy or a tested
yes/no reformulation. Retain variable options (including binary questions),
full-task quality gates and calibrated probabilities.
[Cardinality analysis](docs/pretrained_reuse/R2_OPTION_CARDINALITY_20261012.md).


### Named JSON decisions

`DecisionOutput.to_json(question_name, option_names)` returns a JSON object
mapping the supplied question name to option probabilities, for example
`{"danger": {"yes": 0.97, "no": 0.03}}`. Names must be unique within the
option set and follow the same order supplied to inference. Binary and custom
multi-option distributions use the same serializer. Values are neither rounded
nor recalibrated; malformed distributions fail closed. JSON is produced by
Python, with no generated tokens or additional model parameters. This interface
does not establish model quality or calibration.


### Matched wording continuation result

Two matched1,024-update runs completed. Both select their unchanged starting
checkpoint; wording views improve endpoint NLL/Brier relative to repetition,
but neither improves the fixed validation selector over initialization. Neither
continuation is adopted. This is not a capacity ceiling or a generalization
claim. [Full curves and comparison](docs/pretrained_reuse/R2_PAIRED_WORDING_RESULTS_20261012.md).


### Joint candidate readout: measured, not adopted

Matched1,024-update runs completed. Joint processing reduced timed loop cost
and memory, but selected original/curriculum Accuracy58.31%/40.09% trails
independent63.54%/84.08%. Reversing candidates drops joint original Accuracy
to40.79%; independent predictions remain exact. Reject this candidate, without
claiming a joint-architecture capacity ceiling.
[Curves and fixed order diagnosis](docs/pretrained_reuse/R2_JOINT_LEARNING_RESULTS_20261012.md).


### Training-order intervention (2026-10-12)

A matched 1,024-update option-shuffle run completed with exact best reload.
Original/curriculum Accuracy fell to 54.80%/25.29%; reversing options still
changes 897/2,540 and 1,741/2,048 answer identities. The smaller aggregate order
gap does not establish stability. Candidate rejected; all prior artifacts remain.
[Verified results](docs/pretrained_reuse/R2_JOINT_SHUFFLED_RESULTS_20261012.md).


### Descriptive video grounding (2026-10-12)

A frozen-feature, 24-epoch matched comparison completed on 1,337 training and
2,200 validation questions. Video macro Accuracy38.14% versus no-video35.72%
has an interval including zero; matched versus mismatched video shows a small
+2.32pp scene-information signal. Both overfit and are rejected for product use.
This supports a bounded token-readout diagnostic, not more epochs or a temporal
reasoning claim. [Results](docs/pretrained_reuse/R2_VIDEO_GROUNDING_RESULTS_20261012.md).


### Completed descriptive token comparison; transition to R2-M

All three frozen-token modes completed24epochs. Attention39.99% versus mean38.91%
has an uncertain difference; both exceed no-video35.83% on development metrics.
Attention clearly overfits. Preserve as optional-criterion evidence, do not
extend this family; prioritize the new modality-native/sensor-inclusive R2-M.
[Measured report](docs/pretrained_reuse/R2_GROUNDING_TOKEN_RESULTS_20261012.md).


### Shared spatial image comparison (2026-10-12)

Two fixed 24-epoch readouts completed on the same 8,424 train / 512 selector
images. Shared VideoPrism global/spatial Accuracy is 90.77%/90.92%; the paired
+0.15pp difference is uncertain. Spatial counting improves but texture declines,
with a larger train/validation gap. Keep both artifacts; no product adoption or
95% claim. The image stack is 133,475,584 pre-quantization parameters; a complete
Omni runtime remains unverified. [Curves, hashes and controls](docs/pretrained_reuse/R2_SHARED_SPATIAL_RESULTS_20261012.md).


### Shared global runtime export (2026-10-12)

Algebraically folding repeated global cells reduces the image readout from
1,720,576 to 246,016 parameters without changing any of 2,048 saved answers.
Eight raw images reproduce cached features exactly and produce32 matching typed
JSON decisions. This is a fixed-taxonomy runtime check, not a general Omni
quality result. [Measured export and remaining gates](docs/pretrained_reuse/R2_SHARED_GLOBAL_RUNTIME_20261012.md).


### One shared visual encoder process (2026-10-12)

A single114,365,184-parameter VideoPrism tree now serves explicit image and
video profiles. Alternating raw image/video requests exactly reproduce prior
features, with unchanged weight hashes and a measured1.20GB process peak.
This verifies feature sharing, not joint quality or physical backend buffer
deduplication. [Execution and timing limits](docs/pretrained_reuse/R2_SHARED_VISUAL_ENCODER_20261012.md).


### Fixed full-image preprocessing comparison (2026-10-12)

A single matched24-epoch global readout completed using full-image resizing.
Accuracy90.72% versus cropped90.77%, with uncertain aggregate metric differences.
Counting66.41%->67.58% is also uncertain and texture declines. Preserve the
alternative, retain the existing runtime reference, and stop this preprocessing
comparison without an extra variant or budget extension.
[Curves, paired task intervals and hashes](docs/pretrained_reuse/R2_FULL_IMAGE_RESULTS_20261012.md).


### Training-only counting diagnosis (2026-10-12)

On all8,424 training images the frozen global readout has exact counting68.02%,
with2,001 overestimates and693 underestimates. An affine fixed-candidate rewrite
changes no answers within declared numeric tolerances. No new training or
validation evaluation was performed. Optimize a controlled training-only linear
baseline before attributing the remaining weakness to encoder capacity.
[Confusion matrix and limits](docs/pretrained_reuse/R2_COUNT_TRAINING_DIAGNOSTIC_20261012.md).


### Training-only affine optimization (2026-10-12)

A fixed 200-iteration CPU fit on the same frozen image features improved training
count Accuracy 68.02% -> 78.17% and aggregate training Accuracy 91.43% -> 94.46%.
ECE worsened and the convergence gate was not reached. No validation or audit was
loaded; preserve the runtime reference and do not call this generalization gain.
[Protocol, curves and integrity](docs/pretrained_reuse/R2_AFFINE_FIT_RESULTS_20261012.md).


### Fixed affine endpoint validation (2026-10-12)

The frozen 200-iteration endpoint did not transfer its counting training gain:
reused-validation counting Accuracy 66.41% -> 65.82%, with uncertain paired
Accuracy/NLL differences and a larger train-validation gap. Aggregate NLL improves,
but ECE worsens. Preserve the original runtime; stop unregularized extension and
predefine regularization before further fitting. No audit or promotion.
[Matched metrics, gaps and intervals](docs/pretrained_reuse/R2_AFFINE_ENDPOINT_RESULTS_20261012.md).


### Anchored affine regularization (2026-10-12)

Two fixed 200-iteration fits improve counting on reused validation from the
unregularized 65.82% to 70.51%; paired Accuracy/NLL intervals favor regularization.
Coefficient 0.01 has aggregate Accuracy 91.94%, NLL 0.1964, and is the predefined
research recommendation. This remains far below the weakest-task target and is
not product promotion or a blind result. Stop coefficient refinement on these IDs.
[All arms, curves, gaps and limits](docs/pretrained_reuse/R2_AFFINE_REGULARIZATION_RESULTS_20261012.md).


### Actual shared-visual residency (2026-10-12)

A live two-worker CPU smoke holds the shared image/video backbone, text/speech/
event-audio encoders and conservative head tensors at 195,420,792 elements,
within the pre-quantization 200M budget for those components. RSS sum is 1.81 GB
at the snapshot. A replacement checkpoint and Windows launcher PID initially
caused double counting and memory undercounting; the corrected run preserves both
attempts. This is capacity evidence, not a wired or quality-qualified Omni model.
[Exact ledger and integration gaps](docs/pretrained_reuse/R2_SHARED_RESIDENT_BUDGET_20261012.md).


### Shared raw visual decisions (2026-10-12)

One visual worker now handles raw images and a raw video with the existing heads:
32 image decisions and8 video decisions match saved outputs, with zero answer
changes; repeated images are exact. The video path explicitly retains2048 tokens
required by its attention head. This132.21M functional slice excludes the other
modalities and is a runtime parity result, not new quality or joint-fusion proof.
[Execution, JSON outputs and remaining gates](docs/pretrained_reuse/R2_SHARED_VISUAL_DECISIONS_20261012.md).


### Reusable native speech JSON (2026-10-12)

The native speech runtime now encodes each audio once and shares transcript
scoring across candidate sets, returning custom-ID JSON. Eight existing assets /
16 decisions match the original CUDA reference without answer changes. CPU
probabilities also match but its raw-score gate fails; that limitation is retained.
This is transcript-choice runtime parity, not general audio or joint-Omni quality.
[Both backend results and failure records](docs/pretrained_reuse/R2_NATIVE_SPEECH_RUNTIME_20261012.md).

Dense temporal occlusion comparison completed on 99 train / 257 validation videos.
Mean and attention reach 39.11% / 38.91% selected Accuracy, near the train-only prior;
mismatched videos do not reduce Accuracy and later epochs overfit. Neither head
is promoted. [Results and controls](docs/pretrained_reuse/R2_OCCLUSION_RESULTS_20261013.md).

Full spatial-context follow-up completed on the same356 clips: initial-broadcast
and32-time appearance both reach38.91% selected Accuracy, with0/127 changed-target
pairs both correct. Both50-epoch curves overfit; no adoption or budget extension.
[Audited comparison, controls and all curves](docs/pretrained_reuse/R2_SPATIAL_READOUT_RESULTS_20261014.md).

Train-only follow-up confirms the final readout uses both question and scene
information on its training examples; its poor development generalization remains.
No new training or product promotion. [Fixed diagnosis and numeric replay corrections](docs/pretrained_reuse/R2_CONDITION_BINDING_RESULTS_20261014.md).

Native audio-to-four-video correspondence completed on889 observed development
assets: random candidates94.85%, near-command candidates84.08%. Fixed audio/video
heads, no new model training; full predictions and custom-ID JSON reproduce in a
separate process. This is a controlled AV task, with correlated query roles and
explicit uniform missing-input controls, not four-modality95–98% qualification.
[Full metrics, controls, hashes and limitations](docs/pretrained_reuse/R2_LOMBARD_CORRESPONDENCE_RESULTS_20261015.md).

Native AV correspondence now also runs from actual audio/video bytes to custom
JSON. A six-clip TRAIN pilot and separate raw replay pass;48 repeated raw requests
and128 additional decisions reuse states without encoder forwards. Cached-score
latency excludes native encoding. AV122.79M is a functional slice, not complete
Omni quality or packaging. [Execution and exact reference distinctions](docs/pretrained_reuse/R5_NATIVE_AV_CORRESPONDENCE_RESULTS_20261015.md).

TRAIN-only reliability weighting of six native AV slots completed512 fixed
iterations and an exact fresh refit. Near-command Accuracy84.08%→84.11% and
NLL.4133→.3983; random Accuracy and Brier worsen, so the combined prospective
gate fails. Keep equal weights and close this coefficient comparison; no new
runtime/Teacher promotion. The four-modality95–98% objective remains unmet.
[Curves, paired intervals, all attempts and hashes](docs/pretrained_reuse/R2_AV_SLOT_WEIGHTS_RESULTS_20261015.md).
