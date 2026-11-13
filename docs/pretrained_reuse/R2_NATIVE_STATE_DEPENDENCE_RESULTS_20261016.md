# Native入力の取り違え診断：値への対応付けが弱い

2026-10-16。固定TRAIN1,250ペア・2,500元質問を初期重みと2,048更新後で実評価し、別processで全logitsの完全一致を確認した。追加学習0。**実行PASS、最終目標は未達。**

[事前固定手順](R2_NATIVE_STATE_DEPENDENCE_PROTOCOL_20261016.md) · [全指標・source別・hash](R2_NATIVE_STATE_DEPENDENCE_RESULTS_20261016.json)

## 何を測ったか

同じ質問・同じ全候補に対して、入力状態と正解だけが異なる二つのTRAIN例を組にした。候補数2〜40を保持し、同じquery vectorsで両入力を推論。source/task/消費済み区分ごとにseed17で決定し、各区分最大8ペア。後から成績のよいペアを選んでいない。モデルは通常attention版の初期値と固定2,048終点で、最良checkpointの再選択をしていない。

初期値ではすべてのペアで「両方正解」が0。学習後も、特にTextの値への対応が弱い。次表の比率は意図的に異なる正解を持つ組を選んだ**TRAIN診断**であり、元のvalidation Accuracyや未知データの性能ではない。

## 学習後の実測

| 区分 | 入力 | ペア数 | 両方正答 | 入力を変えると回答が変化 | 正解確率の対称効果 |
|---|---|---:|---:|---:|---:|
| consumed | text | 442 | 0/442 (0.00%) | 2.71% | +0.000341 |
| consumed | image | 32 | 3/32 (9.38%) | 68.75% | +0.024161 |
| consumed | audio | 88 | 1/88 (1.14%) | 31.82% | +0.008590 |
| consumed | video | 16 | 1/16 (6.25%) | 6.25% | +0.000162 |
| unconsumed | text | 504 | 0/504 (0.00%) | 3.37% | +0.000149 |
| unconsumed | image | 32 | 0/32 (0.00%) | 68.75% | +0.036159 |
| unconsumed | audio | 136 | 2/136 (1.47%) | 21.32% | +0.001174 |
| unconsumed | video | 0 | 未測定 | 未測定 | 未測定 |

対称効果は各正解候補について「その候補が正しい入力の確率−もう一方の入力の確率」を平均した値。回答が変わっても誤答なら成功に数えない。Accuracy/NLL/Brier/ECE、total variation、正しい方向へ両確率が動く率、全task/source別結果はJSONに保存。

- **Text：**消費済み442ペア・未消費504ペアの両方で、両正答0。元のstateを変更しても多くの回答が固定される。消費済みcondition-relevance332ペアでは回答変化は2ペア、programmatic55ペアでは0。質問・候補に対する傾向は学んでも、状態の正しい値を安定して使うところに至っていない。
- **Image：**消費済み・未消費とも回答は68.75%で変化し、入力を全く無視しているわけではない。ただし両方正解は3/32、0/32。入力感度と正確な値の読み取りを区別する。
- **Audio：**Lombardは消費済み1/48、未消費2/48の両正答。STARSSは40/88ペアとも回答変化・両正答0で、既存のall-no問題と整合する。
- **Video：**消費済み16ペアで回答が変わるのは1ペア。正解確率の平均効果も小さい。ただし静的8・時間的8という小標本で、temporal encoderの能力限界は判断できない。

## 分割・選択の限界

元TRAIN100,580問のうち前回runで消費されたunique6,656問をhistoryから特定。未消費は同じTRAIN所属であり、新しい盲検testではない。今回1,250ペアは2,500異なるrow ID、1,399観測、2,526query pairsを使う。複数の質問で観測が共通するため、2,500件すべてが独立ではない。

91課題×2区分の182セルを保持し、17セルはペア不成立としてnullを記録。Video512TRAIN問はすべて消費済み。text:originalは質問と候補を完全に一致させた異なる正解の組が作れない。そこで条件を緩めたり、成功扱いしたりしていない。消費済み/未消費の構成も異なり、両者の単純差を学習効果の因果推定にしない。

独立の照合でも、全2,500rowのTRAIN所属、消費区分、質問、全候補、元の正解、query vectorキー、観測対応、row非重複が一致した。元corpusと旧validationの変更はない。

## CPUで追加したスケール確認（post-hoc）

固定TRAIN観測をfeature種別ごとに8件取り、重みを更新せず確認した。学習後の投影特徴RMS / modality embedding RMSは、Text2.888/0.946、Image1.453/1.053、Video2.049/1.038、Speech1.648/0.908、Event audio0.309/0.964。少数観測で共通の単純な振幅不足は見られず、一律のスケール変更を正当化しない。

Videoの平均memory vectorの観測間RMS変動は0.0297と小さいが、全tokenの情報量やattentionの利用可能性の測定ではない。Event audioが小さめであることも、単独で失敗原因とは断定できない。正しい読出しが可能かは次の学習比較で検証する。

## 実行と保存

実行source `eaa595543046701a4b91555858fe7dc21f5ccde4`。設定は`configs/pretrained_reuse/native_state_dependence_v1.json`。準備→別GPU smoke7ペア→全probe→全fresh replayを別出力で実行。モデルstate digestは初期・終点とも前後一致し、optimizer更新0。GPU推論は既存特徴を使うreadoutのみで、encoderを再ロードしていない。

| 段階 | 秒 | peak CUDA bytes |
|---|---:|---:|
| plan | 10.97 | GPU不使用 |
| smoke | 11.80 | 66781184 |
| probe | 23.39 | 120703488 |
| replay | 21.84 | 120703488 |

4段階のprocess内wall合計68.00秒。CPUの追加確認は0.31秒。Torch2.6.0+cu124、NumPy2.4.6、psutil7.2.2、FP32/TF32無効、元baselineと同じversion。RTX3080 Laptopを使用。クラウド費用0。raw receiptの`completed_pairs=1152`は最後の途中通知で、終端件数は`executed_pairs=1250`および検証済みshape `(1250,2,40)`。初期/終点とも全件処理を確認済み。

保存先は`C:/CodexArtifacts/pretrained-reuse-r2/native-state-{plan,smoke,probe,replay}-01/`、独立照合は`native-state-verification-01/`、CPU確認は`native-state-scale-audit-01/`。全予測・正解対応・planとSHA・config・環境・時刻・state digestを保存。重みは元runから読み取りだけで、Gitへ追加していない。

```powershell
python -u -m scripts.probe_native_state_dependence --phase prepare --output <new-plan>
python -u -m scripts.probe_native_state_dependence --phase smoke --output <new-smoke> --reference <new-plan> --reference-sha256 <plan-receipt-SHA>
python -u -m scripts.probe_native_state_dependence --phase probe --output <new-probe> --reference <new-smoke> --reference-sha256 <smoke-receipt-SHA>
python -u -m scripts.probe_native_state_dependence --phase replay --output <new-replay> --reference <new-probe> --reference-sha256 <probe-receipt-SHA>
```

実行には前報告と同じ固定PYTHONPATH/環境とsourceを使い、毎回新規出力先を指定する。現在のsourceを無条件に旧receiptへ結び直してはいけない。

## 次の一手

単純にLSTM/xLSTMを増やす段階ではない。通常readoutの状態と正解の対応を学習させるため、**同じ質問・候補で異なる正解を持つ状態を同じ更新へ入れるbatch構成**を、独立に組んだbatchと比較するのが次の限定実験。model・CE・候補・元データ・初期値は維持し、両armのsample ID多重集合、unique数、optimizer、更新数を一致させる。既存の実験を上書きせず別configと出力にする。

現在の4microbatch更新と別のbatch設計を比較する場合は、batch sizeや更新数も同じにした新しい対照を必要とする。異なる候補数やsourceを落としてペアを作ってはならず、成立しない行も元の会計へ残す。診断で選んだ少数ペアだけに学習を絞らず、元のTRAIN会計を保持する。これらのTRAIN例を独立評価へ流用しない。全24,346問・元47text区分・各modality・校正を共通条件で評価する。

この次の比較はまだ未実施。診断はreadoutの弱い入力利用を示すが、encoderに情報がないことや、学習量を増やしても改善不能であることは示さない。200M以下とJSON出力の方針、4モダリティ95–98%、joint/OODの要件は維持し、R2は未完了。

## 検証

11件の回帰テスト追加。全CPU2,074 passed / 1 skipped（30.44秒）、Ruff PASS。実行sourceのCI #1823 success（manifest検証含む）。最終公開commitのCIもpush後に確認する。

## 2026-10-25の追補

この節末の「次の一手」は、2026-10-16時点の計画記録であり、現在の未実施作業を示さない。提案していた同じ質問・候補で異なる正解を持つ状態を同一batchへまとめる比較は、同日付の[matched-state batch比較](R2_NATIVE_MATCHED_BATCH_RESULTS_20261016.md)で実行済み。固定8,192 occurrence、同一初期値、モデル、loss、optimizer、1,024 updates、全validation、非成立行の保持を確認した。結果はmacro NLLと消費済みText/Videoの両正答条件を満たさず、この介入は不採用で終了している。重複実験は行わない。
