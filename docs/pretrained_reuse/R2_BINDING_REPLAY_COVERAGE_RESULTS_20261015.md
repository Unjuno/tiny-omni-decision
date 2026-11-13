# R2 保持データの源を広げた固定比較 — 実測結果

**512更新を完了したが、新しい最終重みは不採用。** curriculumの保持は改善した一方、originalの校正は固定基準を外れ、指示の回帰評価も改善と悪化が混在した。判定は `MIXED_RETENTION_TRADEOFF`。

今回の選択step0は実験内のフォールバック。以前のKL512参照候補を置き換えない。四モダリティ95–98%の完成や、モデル容量の限界を示す結果ではない。

[事前固定プロトコル](R2_BINDING_REPLAY_COVERAGE_PROTOCOL_20261015.md) · [全数値・環境・hash](R2_BINDING_REPLAY_COVERAGE_RESULTS_20261015.json) · [戦略](STRATEGY_20261015.md)

## 何を変えたか

- 実行コード `18f39085a4233006e64fcf2b13454de8af3d1fca`。別プロセスで2更新のTRAIN-only smokeを通した後、seed17・512更新を1本実行した。
- 直前の指示多様化runと同じcoverage親から開始。新規optimizerであり、直前の最終重みを継続したものではない。
- binding TRAIN 2,048問／512状態、168指示文字列、40候補と説明、正解、状態、サンプラーのindex順序、学習量・損失・モデル・validation・selectorを固定した。
- 変更は保持用TRAINの構成。各更新の4binding＋4originalを、4binding＋2original＋2curriculumへ変更した。総問数は同じ。originalの各バッチの前半2問を保持し、後半2問を置き換えた。
- curriculumは既存TRAINからseed17のSHA順で128状態を選び、各状態の8問をすべて使用。指示・候補・ラベルを新しく作り直していない。配合比や追加seedの探索はない。
- 同じcoverage親のlogitを新しい混合TRAINバッチで計算し、保持KLへ使用。係数・温度は1、最初のKLは0。評価結果から教師信号を作っていない。

22,713,601全パラメーター、10,647,169学習対象（6BERT encoder block＋classifier）。embedding/pooler凍結、LoRAではない。FP32/SDPA、dropout無効、AdamW、固定LR2e-5、weight decay0.01、gradient clip1。損失は `.5*bindingCE + .5*(replayCE + KL(parent||student))`。

新しいデータ源を含める代わりに、originalを1,024問減らしている。これは予算を固定した置き換えの効果であり、curriculumを単純に追加した場合の因果効果ではない。独立状態数・候補ペア数・計算量も完全には等しくない。

## 消費データと分割

| 源 | 消費したunique問 | unique状態 | 内訳 |
|---|---:|---:|---|
| binding | 2,048 | 512 | color/texture/count/shape 各512、指示系列8種×256問 |
| original | 1,024 | 962 | choice396、noul412、score216 |
| curriculum | 1,024 | 128 | extremum512、temporal_order256、count128、conditional_selection128 |
| 合計 | 4,096 | 1,602（源を区別） | 反復ID0 |

直前runの源別状態数合計は2,327。総問数を固定しても、8問を持つcurriculum状態を追加した今回は1,602へ減っている。新しいmedia assetは0で、構造化状態を使う条件判断の診断である。

全TRAINとvalidation 5,612問（binding1,024、original2,540、curriculum2,048）のID・正規化状態・group重複を拒否。binding TRAINとsample-orderファイルは元のものとbyte一致。保持基準とvalidation IDs／順序は変更していない。

選択step0はこのrunで0問消費した時点の重み。最終512の4,096問を選択checkpointの学習量とは報告しない。bestとfinalを別保存し、bestはcoverage親と同じSHA。

## 学習曲線

| step | 累積問 | 固定TRAIN Accuracy / NLL | binding validation NLL | original NLL | curriculum NLL | 保持合格 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 27.73% / 2.491533 | 2.189042 | 0.835591 | 0.375370 | yes |
| 128 | 1,024 | 87.50% / 0.325661 | 0.003904 | 0.845219 | 0.346233 | no |
| 256 | 2,048 | 100% / 0.019246 | 0.000169 | 0.857434 | 0.354355 | no |
| 384 | 3,072 | 100% / 0.002571 | 0.000397 | 0.855330 | 0.371314 | no |
| 512 | 4,096 | 100% / 0.001345 | 0.000949 | 0.847545 | 0.358755 | no |

binding validation Accuracyは初期31.74%、学習後の全記録点100%。固定TRAIN probeは同じ最初の256問。TRAIN NLLが下がり続けてもbinding validation NLLは256以降増加しており、最後を自動採用する根拠にはならない。絶対値は小さく、この曲線だけで汎用能力の上限や大きな過学習を断定しない。

## 同条件の保持比較

| 集合・重み | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| original 親step0 | 63.58% | 0.835591 | 0.470606 | 0.018915 |
| original 直前language512 | 63.86% | 0.844994 | 0.474192 | 0.039637 |
| original 今回512 | 63.86% | 0.847545 | 0.476894 | 0.042107 |
| curriculum 親step0 | 84.08% | 0.375370 | 0.214324 | 0.016823 |
| curriculum 直前language512 | 83.15% | 0.448147 | 0.239993 | 0.043105 |
| curriculum 今回512 | 84.62% | 0.358755 | 0.204658 | 0.025217 |

今回512対直前512の、全状態を対応させたbootstrap 10,000反復、seed17、percentile95%（linear）：

- curriculum Accuracy **+1.46484ポイント**、区間 **[+0.19531,+2.73438]**。
- curriculum NLL **−0.089392**、区間 **[−0.114258,−0.065029]**。
- original Accuracy差0、NLL **+0.002550**、区間 **[−0.006640,+0.011771]**。
- binding Accuracy差0、NLL **+0.000781**、区間 **[+0.000421,+0.001213]**。

最終512の保持違反はoriginal ECEのみ。親から **+0.023192**で、固定上限+0.02を超えた。他の最終保持条件は通ったが、上限を結果に合わせて緩めない。全ての学習後checkpointが保持不合格なので、selectorは正しくstep0を選んだ。

これは一つのseedと開発用validationの結果。区間は状態間の不確かさを示し、学習seed間の変動、checkpoint選択の影響、未知タスクへの汎化を解消するものではない。

## 指示の回帰評価

前回すでに観測した256状態×4属性＝1,024問を、同じ40択の7条件で評価した。**新しい盲検評価ではない**。checkpoint選択後に旧language512、今回512、選択step0を評価し、この点数で配合比やcheckpointを選び直していない。

| 条件 | 直前512 Accuracy | 今回512 Accuracy | 今回NLL | 今回Brier | 今回ECE | 条件ペア両正答率 |
|---|---:|---:|---:|---:|---:|---:|
| 既知形式 | 100% | 100% | 0.001179 | 0.000071 | 0.001153 | 100% |
| 対象明示 | 92.19% | 97.75% | 0.254006 | 0.097109 | 0.174325 | 95.57% |
| 除外リストの残り | 17.58% | 24.61% | 1.706028 | 0.816527 | 0.141854 | 3.71% |
| 対象明示＋新説明 | 81.64% | 76.27% | 0.738645 | 0.334333 | 0.193054 | 55.53% |
| 除外＋新説明 | 6.45% | 15.43% | 2.045303 | 0.917345 | 0.234407 | 0.91% |
| 対象明示＋新説明、候補反転 | 81.64% | 76.27% | 0.738645 | 0.334333 | 0.193054 | 55.53% |
| 質問なし | 25% | 25% | 4.987499 | 1.269476 | 0.593891 | 0% |

対象明示＋新説明のAccuracy差は **−5.37109ポイント**、状態単位95%区間 **[−7.22656,−3.61328]**。NLLは逆に−0.036863改善した。Accuracyと確率の改善は同じではない。

除外＋新説明は **+8.98438ポイント**、区間 **[+7.22656,+10.74219]**だが絶対精度15.43%。NLLは−0.265111改善した。両条件とも属性ごとの最低精度・条件ペア基準は未達で、一般的な条件理解の成立とは扱わない。

継承したtransfer gateの6件が不合格。うち対象明示の「3ポイント改善／区間下限>0」は元の指示多様化実験の診断基準であり、今回の保持仮説の主判定とは区別する。今回の主判定は保持基準とcurriculum NLLで、既にMIXEDである。

## 今回の結果から進む方針

1. 保持する能力をTRAINへ含めることには実測の効果がある。ただし今回の固定配合は採用しない。元データを減らす代償と校正の問題を残し、同じvalidationに対する配合比・温度・予算の探索へ進まない。
2. 狭い指示テンプレートを足す反復を止め、TRAIN側で肯定／除外／条件切替／比較の意味を正しく解ける教師を先に検証する。既存の利用許可済み重みと正解を検算できる対照を使い、教師が強いと仮定して蒸留しない。
3. 条件理解の教師信号を確認してから実入力の知覚と対応付ける。正しい構造化状態での失敗と、画像・音声・動画から必要な情報を取り出せない失敗を分ける。時間・共同入力では必要な入力を除く対照を維持する。
4. 全常駐200M以下、JSON確率出力、状態再利用、各必須タスク95–98%の目標を保持する。量子化、製品昇格、main mergeは行っていない。


## 実行・再現性と計算資源

実行判定はPASS。全512更新、整合性検査、選択重みのin-process reloadを完了し、別プロセスの選択出力12ファイルもbyte一致した。回帰評価3モデル×7条件の全スコア・確率・予測・指標・effective config、計65ファイルが新プロセスでbyte一致した。旧language512／旧step0との同条件スコアも完全一致。

独立CPUレビューでは、5記録点の全指標、21条件・84属性指標、21,504予測行と7個の区間を再計算し、一致を確認した。publicationも元スコアから指標・gate・sampler消費・旧control一致を再構築する。研究の再現性合格と製品品質合格は別判定。

| process | 実測本体秒 | peak allocated VRAM bytes | peak host bytes |
|---|---:|---:|---:|
| smoke | 18.302 | 1,950,013,440 | 1,584,381,952 |
| fit | 493.101 | 2,083,247,104 | 1,604,890,624 |
| selected_replay | 63.785 | 321,214,976 | 1,299,755,008 |
| transfer | 379.039 | 348,481,536 | 1,288,384,512 |
| transfer_replay | 383.227 | 348,481,536 | 1,291,563,008 |

合計1337.453秒（22.29分）は5processの記録された処理本体時間。GPU busy時間や専有環境の速度ベンチマークではない。追加のCPU検証・監視・環境起動時間は含めない。cloud費用0、電気代は未測定。

本fitの更新処理142.921秒、定期評価259.451秒、TRAIN親cache推論19.256秒。gradient normはclip前0.411956〜61.313843、clip上限1。NaN/Inf・凍結重み変更・資源停止なし。UTC2026-10-15T05:59:02.884887→06:07:15.985935、固定512更新予算で終了。

環境は既存のWindows/Python3.11.9、PyTorch2.6.0+cu124、CUDA12.4、Transformers5.6.2、NumPy2.4.6、tokenizers0.22.2、safetensors0.8.0、psutil7.2.2。RTX3080Laptop16GiB、FP32/SDPA、CPU4threads、TF32off、決定的Torch、dropoutoff。依存関係の更新なし。

本fitは511,274candidate pair forward（教師cache7,920を含む）。直前fit502,566との差8,708の内訳は、今回新規計算したcache7,920と、置き換えによる学習候補数差788。総問数が等しくても同一演算量とは呼ばない。長いoriginal状態は従来どおりonly_second/512tokensで切り詰め、1,814pair presentations／120unique問を記録。bindingの切り詰めは0。length計測時のTokenizer警告は保存しており、実forwardの最大長は512。

best-deltaは42,620,884bytes、final-deltaは42,620,986bytes。差分重みの容量であり、base/tokenizerを含む製品パッケージ容量ではない。optimizer/RNG等の完全resume状態は保存していない。

## 保存先とhash

外部成果物は `C:/CodexArtifacts/pretrained-reuse-r2/`。既存v1/v2/QAT・旧研究コーパス・重み・ログを変更せず、新しい出力先だけを使用した。重み・media・大型checkpointはGitに追加しない。

| 成果物 | SHA-256 |
|---|---|
| binding-replay-coverage-corpus-01\result.json | `dd737881dcc24808295c245e1c5aca06fe5ac002ec48dc24bf26dd3ef50499ac` |
| configs\pretrained_reuse\binding_replay_coverage_v1.json | `a63f0859e06767dec73a46329531a9042f6dd52eccac36672fa97527a4d00897` |
| binding-replay-coverage-fit-01\effective-config.json | `aaf1c52e37c2a5c757b2e40d409d5a1ad8c92899c20ca8e3d684e3a96aea99de` |
| binding-replay-coverage-smoke-01\result.json | `d16fbac27d9ecd3f78dc639a785f62d5ca8bce2cab0b702b2db511c4bd165698` |
| binding-replay-coverage-fit-01\result.json | `4f3e3cde7d8449ca9e70f727360532e8b3d706fca943f22e07e843586cec518c` |
| binding-replay-coverage-selected-replay-01\result.json | `c932f6dad6a62423eca3ded7bcedda2e71b2360be91087d49b0ab05061d95d57` |
| binding-replay-coverage-regression-01\result.json | `5b7392956ab8a6c4ef4ac3e33055fb23fa03a27cd448e8e0ab0a8904e095042b` |
| binding-replay-coverage-regression-replay-01\result.json | `4490f1d78be2f84b20062b5f265129ddb67268956579f474589adcd45215ad90` |
| binding-replay-coverage-fit-01\best-delta.pt | `b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5` |
| binding-replay-coverage-fit-01\final-delta.pt | `156ffc4329d64a8de6f269b89189e95957b92418cea2fe8fbf2c61bbd6657c38` |
| binding-replay-coverage-publication-01.json | `d1c8c9c296f00aa27508c4ba2e0f11978ec6392947445b35b52ac3b3811f32a2` |

baseは `cross-encoder/ms-marco-MiniLM-L6-v2`、revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`。base audit、全入力・コードpin、TRAIN/validation/回帰集合、サンプラー順序はcompanion JSONで追跡できる。

主要実行コマンド（smokeは追加`--smoke`、fresh replayは完了result SHAを指定し別出力先）：

```powershell
python -m scripts.prepare_binding_replay_coverage --output C:/CodexArtifacts/pretrained-reuse-r2/binding-replay-coverage-corpus-01
python -m scripts.train_binding_replay_coverage --corpus C:/CodexArtifacts/pretrained-reuse-r2/binding-replay-coverage-corpus-01 --corpus-sha256 dd737881dcc24808295c245e1c5aca06fe5ac002ec48dc24bf26dd3ef50499ac --output C:/CodexArtifacts/pretrained-reuse-r2/binding-replay-coverage-fit-01
python -m scripts.evaluate_binding_replay_coverage --corpus C:/CodexArtifacts/pretrained-reuse-r2/binding-replay-coverage-corpus-01 --corpus-sha256 dd737881dcc24808295c245e1c5aca06fe5ac002ec48dc24bf26dd3ef50499ac --fit C:/CodexArtifacts/pretrained-reuse-r2/binding-replay-coverage-fit-01 --fit-sha256 4f3e3cde7d8449ca9e70f727360532e8b3d706fca943f22e07e843586cec518c --output C:/CodexArtifacts/pretrained-reuse-r2/binding-replay-coverage-regression-01
```

各result.jsonに実際のargvを保存。出力先の再利用とhash不一致を拒否する。これは実行済み比較の再現記録であり、追加runの開始指示ではない。

## 未実施・検証

追加seed、延長学習、配合比探索、結果に合わせた温度校正、実画像／音声／動画の追加学習、新しい四モダリティ統合、量子化、製品採用、sealed audit、main mergeは行っていない。教師の条件意味診断は次の作業であり、今回実行済みとは報告しない。

manifest/catalogは25コマンド成功（外部`binding-replay-coverage-manifest-validation-02.json`）。初回は存在しない`tiny_omni_decision.__main__`を指定したためCLI起動に失敗し、記録01を保持した。正しい既存entry `python -m tiny_omni_decision.cli`で再実行し、manifest自体は変更していない。

Ruff pass。CPUテスト1,611 passed／1 skipped（22.61秒）。文書内のlocal linkとcompanion JSONのbyte一致を確認した。結果commitのGitHub Actionsはpush後に同じSHAで確認し、循環hashを避けて最終報告・外部CI記録に保存する。
