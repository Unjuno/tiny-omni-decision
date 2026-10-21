# R2 指示多様化の固定学習比較 — 実測結果

実行と再現確認はPASS。**新しい最終重みは不採用**。指示の多様化は対象を明示する未知構文へ有効だったが、除外から対象を求める構文、校正、旧タスク保持は基準未達。

今回のstep0選択はこの実験内のフォールバックであり、以前のKL512参照候補を置き換えない。四モダリティ95–98%の製品完成でも、22.7Mモデルの能力上限を示す結果でもない。

[事前固定プロトコル](R2_BINDING_LANGUAGE_PROTOCOL_20261015.md) · [全指標・hash・環境JSON](R2_BINDING_LANGUAGE_RESULTS_20261015.json) · [戦略](STRATEGY_20261015.md)

## 実行した内容

- 実行コードを `e33567dcb23ee103e3026726370476d442b5854a` に固定。2更新のTRAIN-only smokeを別プロセスで実行後、seed17の512更新を1本実行した。
- 旧KL512と同じcoverage親から開始。新規optimizerであり、旧KL512の続きではない。22,713,601全パラメーター、10,647,169学習対象（6encoder block＋classifier）。embedding/poolerは凍結。LoRAではない。
- 同じ状態・ラベル・40候補・候補説明・サンプラー順序・学習率・損失・validation・選択規則を保持。変更はbinding TRAINの指示文のみ。8文字列から168文字列、8構文系列に増やした。
- AdamW、固定LR2e-5、weight decay0.01、clip1、dropout無効、4binding＋4original問/update。損失は `.5*bindingCE + .5*(replayCE + KL(parent||student))`、係数と温度1。
- original TRAIN専用の親logit cacheをhashと全行の同一性で検証して再利用。初回KL=0。cache推論を省略した分を学習高速化とは呼ばない。
- 未使用256状態を学習前に固定。過去のTRAIN・診断・validation・transferを合わせた6,428状態を除外。各状態4問、全条件40択。2つの未使用英語構文系列と新候補説明を含む7条件。
- この新しい評価は選択後に初めてモデル評価した。主比較は旧final512対新final512。選択step0も補足として評価。全結果を新プロセスで再実行した。

## 消費量と学習曲線

本学習全体は4,096 unique問、反復ID0。bindingは2,048問／512状態（各属性512問、各構文系列256問／64状態）。original replayは2,048問／1,815状態（choice814・noul824・score410）。合計2,327は源を区別した状態数。新しいmedia assetは取り込んでおらず、構造化状態による診断である。

選択checkpointはstep0なので、このrunの学習消費は0問。最終512の4,096問を選択checkpointの消費量として扱わない。bestとfinalは別ファイルに保持した。

| step | 累積問 | 固定TRAIN Accuracy / NLL | 既知binding validation NLL | original NLL | curriculum NLL | 保持合格 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 27.73% / 2.491533 | 2.189042 | 0.835591 | 0.375370 | yes |
| 128 | 1024 | 85.94% / 0.317472 | 0.002550 | 0.837701 | 0.422856 | no |
| 256 | 2048 | 100.00% / 0.021297 | 0.000361 | 0.849519 | 0.478497 | no |
| 384 | 3072 | 100.00% / 0.003552 | 0.000465 | 0.854361 | 0.448631 | no |
| 512 | 4096 | 100.00% / 0.003284 | 0.000169 | 0.844994 | 0.448147 | no |

既知binding validation Accuracyはstep0の31.74%から全学習後点100%。固定TRAIN probeは最初の256問で、今回の多様化した指示を含む。旧runのTRAIN probeとは指示が異なるため直接の曲線比較はしない。TRAINと既知validationの良好な適合は、未知構文への汎化を証明しない。

全学習後checkpointが旧タスク保持に不合格。最終512ではcoverage親よりoriginal ECEが0.020722増加、curriculumの条件選択Accuracyが3.515625ポイント低下、curriculum NLL/Brier/ECEが0.072778/0.025669/0.026282増加した。いずれも固定基準で判定し、閾値を緩めなかった。

| 新512 − 旧KL512（同じ旧validation） | Accuracy差 | NLL差 | NLL差の状態単位95%区間 |
|---|---:|---:|---:|
| binding | +0.000 pp | -0.001971 | [-0.003453, -0.000815] |
| original | +0.000 pp | +0.007516 | [+0.002448, +0.012630] |
| curriculum | -0.781 pp | +0.054985 | [+0.041026, +0.068942] |

## 未使用構文での同条件比較

以下は同じ新規集合内の比較。前回の8条件の49.22%等と今回の点数を直接引き算しない。各行1,024問／256状態で、7条件は同じ状態を再使用している。独立状態が1,792あるとは数えない。

| モデル | 条件 | Accuracy | NLL | Brier | ECE | 条件ペア両方正答 |
|---|---|---:|---:|---:|---:|---:|
| control | known | 100.00% | 0.000910 | 0.000124 | 0.000876 | 100.00% |
| control | focus | 36.33% | 1.789463 | 0.763819 | 0.043239 | 9.64% |
| control | complement | 4.39% | 2.764745 | 0.997724 | 0.256128 | 0.07% |
| control | focus_descriptions | 31.15% | 2.207658 | 0.834897 | 0.056507 | 9.64% |
| control | complement_descriptions | 1.46% | 3.300161 | 1.061485 | 0.310690 | 0.00% |
| control | reverse_focus_descriptions | 31.15% | 2.207658 | 0.834897 | 0.056507 | 9.64% |
| control | no_question | 25.00% | 3.501199 | 1.179734 | 0.517849 | 0.00% |
| candidate | known | 100.00% | 0.000089 | 0.000000 | 0.000089 | 100.00% |
| candidate | focus | 92.19% | 0.340111 | 0.155611 | 0.154640 | 84.83% |
| candidate | complement | 17.58% | 1.833271 | 0.880599 | 0.213383 | 1.43% |
| candidate | focus_descriptions | 81.64% | 0.775508 | 0.339447 | 0.220563 | 64.78% |
| candidate | complement_descriptions | 6.45% | 2.310414 | 1.009544 | 0.339930 | 0.33% |
| candidate | reverse_focus_descriptions | 81.64% | 0.775508 | 0.339447 | 0.220563 | 64.78% |
| candidate | no_question | 25.00% | 5.354327 | 1.270327 | 0.588949 | 0.00% |
| selected | known | 30.47% | 2.198278 | 0.954342 | 0.349325 | 4.17% |
| selected | focus | 25.88% | 2.751469 | 1.113226 | 0.471511 | 0.65% |
| selected | complement | 19.82% | 2.154687 | 0.893960 | 0.175457 | 0.91% |
| selected | focus_descriptions | 27.05% | 2.203952 | 0.888945 | 0.176531 | 1.63% |
| selected | complement_descriptions | 10.64% | 2.593426 | 1.006051 | 0.294651 | 0.00% |
| selected | reverse_focus_descriptions | 27.05% | 2.203952 | 0.888945 | 0.176531 | 1.63% |
| selected | no_question | 25.00% | 2.932457 | 1.072836 | 0.425144 | 0.00% |

`control`=旧KL512、`candidate`=今回final512、`selected`=今回選択step0（coverage親）。candidateは保持不合格でも固定予算の因果比較のため評価したもので、採用していない。全属性・全4問同時正答・平均confidenceはJSONに保存。

| 主条件 | 旧512 → 新512 Accuracy | 改善の状態単位95%区間 | NLL差の95%区間 |
|---|---:|---:|---:|
| focus_descriptions | 31.15% → 81.64% | [+47.07, +53.91] pp | [-1.499752, -1.367445] |
| complement_descriptions | 1.46% → 6.45% | [+3.32, +6.64] pp | [-1.029804, -0.951149] |

両主条件で改善幅は正だが、両方とも最低属性Accuracy95%、条件ペア90%、ECEの許容差という3項目に不合格（計6項目）。対象明示ではconfidence平均59.58%に対しAccuracy81.64%で過小確信、除外ではconfidence40.44%に対しAccuracy6.45%で過大確信だった。単一の「過信」だけで両方を説明しない。評価後の温度調整は行っていない。

| 新512の主条件 | 属性 | Accuracy | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| focus_descriptions | color | 97.66% | 0.328873 | 0.115704 | 0.236837 |
| focus_descriptions | count | 44.92% | 1.735158 | 0.772314 | 0.141482 |
| focus_descriptions | shape | 92.19% | 0.562267 | 0.250975 | 0.307505 |
| focus_descriptions | texture | 91.80% | 0.475737 | 0.218796 | 0.245281 |
| complement_descriptions | color | 5.86% | 2.086256 | 0.962843 | 0.316520 |
| complement_descriptions | count | 1.56% | 2.585498 | 1.080376 | 0.412820 |
| complement_descriptions | shape | 0.00% | 2.914873 | 1.161428 | 0.448560 |
| complement_descriptions | texture | 18.36% | 1.655029 | 0.833529 | 0.200207 |

不確かさは10,000回、seed17、256状態をまとめて再標本化するpaired bootstrap、percentile95%・linear補間。4問を独立例として水増ししない。開発用の生成課題であり、単一seed・二つの未使用構文系列だけの転移である。構成選択後の評価でも、新しい製品用sealed auditとは呼ばない。

## 診断と次の判断

保存スコアだけの事後診断で、対象明示＋新説明／除外＋新説明の属性選択は82.71%／6.45%。要求された属性が分かっていると仮定した値の順位は98.34%／97.56%。これは正解属性を利用した原因切り分けであり、10候補への縮小による点数を40択の精度に置き換えない。主要課題は未知の指示と答える属性の対応に残る。

保持用original replayに、生成curriculum源のTRAIN例は0件だった。その利用可能なTRAIN集合は8,192問／1,024状態。源の欠落は確認したが、一般のoriginal例に似た概念が全くないとは言わず、忘却の唯一の原因とも断定しない。

次は条件意味・候補の意味対応を含む独立TRAIN契約と、維持すべき能力を含むリプレイの契約を作る。保持の源を広げる比較は、指示追加や容量変更と分離する。必要なら参照教師の能力をTRAIN側で先に実測する。今回の文面に合わせた再学習、同じ小集合での追加seed・係数・step探索は行わない。

## 再現性・安全性・費用

- 本学習UTC: `2026-10-15T05:18:03.092904+00:00` → `2026-10-15T05:25:47.668479+00:00`。固定512更新を完了、終了理由は予定予算到達。NaN/Inf・凍結更新・容量停止はなし。
- 選択時の4スコア配列をreloadで完全一致。別プロセスの選択再評価で12出力が一致。未使用構文の全再評価で65出力（スコア・確率・予測・指標・effective config）がbyte一致。候補反転の確率と選択ID、質問なしの4分布一致／Accuracy25%以下／ペア0も通過。
- CPU集計で全20学習曲線指標、全21転移指標、保持条件、7paired区間、入力対照、run間のhash連結を再構築。独立レビューでも学習341hash・転移415hash・21,504予測行を確認。
- RTX3080Laptop16GiB、FP32 SDPA、4CPU threads、TF32無効・決定論的Torch。Python3.11.9、Torch2.6.0+cu124、CUDA12.4、Transformers5.6.2、NumPy2.4.6、tokenizers0.22.2、safetensors0.8.0、psutil7.2.2。実行環境の更新なし。

| 処理 | 記録された秒 | peak allocated VRAM MiB | peak host MiB | candidate pair提示回数 |
|---|---:|---:|---:|---:|
| smoke | 11.991 | 1861.06 | 1515.67 | 353 |
| fit | 464.565 | 1987.74 | 1530.27 | 502,566 |
| selected_replay | 57.091 | 306.33 | 1238.57 | 68,919 |
| transfer | 366.537 | 332.34 | 1230.79 | 860,160 |
| transfer_replay | 370.584 | 332.34 | 1230.05 | 860,160 |

記録されたprocess bodyの合計は1270.767秒（21.18分）。実GPU busy時間や独立したlatency benchmarkではない。学習更新body合計143.441秒、定期評価256.445秒。ローカルのみ、cloud費用0、電力費未測定。レビュー・実装・CPU検証の時間はこの合計に含まない。

新binding行は切捨てなし。旧original源の既存only_second/512方針は変更せず、1779提示／163固有問で切捨て。旧runとの差は親cache推論の再実行を省いた分であり、入力制限緩和ではない。転移入力の最大長82tokens、切捨てなし。

勾配norm（clip前）範囲0.159157–51.051640、clip1。bestファイル42,620,884bytes（元親と同一SHA）、final42,620,986bytes。これらはdeltaだけで、全推論パッケージ容量ではない。optimizer／RNG stateの完全保存はなく、厳密resume対応とは主張しない。

## 成果物とhash

外部保存先は `C:/CodexArtifacts/pretrained-reuse-r2/`。重み・予測・大きなcheckpointはGitに追加しない。

| 成果物 | SHA-256 |
|---|---|
| binding-language-corpus-01/result.json | `142c4e540b67b65aea3c4803916e98cb3019009bd3ef6fefdeec6d9b403e971f` |
| config | `eef666ef40197baf82e0275c155186104cc9b55e3c916dcb5baf76ec920dceb0` |
| fit effective-config.json | `19e8640198f71fa7c59cfe24e5ec92f99f5d82373f6c9785432cb8bd4bdc34b5` |
| binding-language-smoke-01/result.json | `f6365c71c19d1c14db8e81a7a0e1a78844ee41a7edfbc245b308545648362e07` |
| binding-language-fit-01/result.json | `b954742214821de1923f4cb7179cf7ae4325e4cf7f25456fbe5bc6f009de3174` |
| binding-language-selected-replay-01/result.json | `e22b168e820161c9e6fd3b1ee12049699b27c6fdc82aca07f15a1c95e9d519da` |
| binding-language-transfer-01/result.json | `52b9b5ae11c80a69b4071d1fc4ed714adc9265ad1833d90a88faa4fbe5c3faa2` |
| binding-language-transfer-replay-01/result.json | `bb2b2cc33e288da3ed78f04a400723609912ffad5abf7fec743608fd4bae58d1` |
| fit best-delta.pt | `b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5` |
| fit final-delta.pt | `3d3a6d374e16bbd3dbc5907d426f6fcf5cb363bb18408105cdea66ec9dab07dd` |
| binding-language-publication-01.json | `c8ce2ae08aef90acc4b6f12e3102888d1875de4d83c6a3467c8126d426674e77` |
| binding-language-field-diagnostic-01.json | `83e6de29055c049c312ec128c248e8a034c0c7b610c554618efec5953190562d` |
| binding-language-retention-coverage-01.json | `107854855718db02f6f528129cf2c321dc8ee5612a7c735a53162ff2f34d9485` |

base: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`。base audit・全ファイルhash・旧corpus・新TRAIN・各heldout条件・sample orderはcompanion JSONの入力pinsで固定。既存v1/v2/QAT/旧研究成果物は変更していない。

実行引数は各result.jsonに保存。主要コマンド:

```powershell
python -m scripts.prepare_binding_language --output C:/CodexArtifacts/pretrained-reuse-r2/binding-language-corpus-01
python -m scripts.train_binding_language --corpus C:/CodexArtifacts/pretrained-reuse-r2/binding-language-corpus-01 --corpus-sha256 142c4e540b67b65aea3c4803916e98cb3019009bd3ef6fefdeec6d9b403e971f --output C:/CodexArtifacts/pretrained-reuse-r2/binding-language-fit-01
python -m scripts.evaluate_binding_language --corpus C:/CodexArtifacts/pretrained-reuse-r2/binding-language-corpus-01 --corpus-sha256 142c4e540b67b65aea3c4803916e98cb3019009bd3ef6fefdeec6d9b403e971f --fit C:/CodexArtifacts/pretrained-reuse-r2/binding-language-fit-01 --fit-sha256 b954742214821de1923f4cb7179cf7ae4325e4cf7f25456fbe5bc6f009de3174 --output C:/CodexArtifacts/pretrained-reuse-r2/binding-language-transfer-01
```

出力先は既存であれば拒否する。再実行は新しい出力先で行い、replayには完了済みresultのSHAを指定する。smokeの追加フラグは`--smoke`。実行済みreplay／publicationの全引数とhashは保存済みで、実験を繰り返すための指示ではない。

## 未実施・未達

追加seed・延長学習・別ハイパーパラメーター・評価後の温度校正は未実施。日本語、実画像／音声／動画、センサー、native時系列、両入力が必要な共同判断は今回未評価。新しい4モダリティ統合モデル、量子化、製品採用、sealed audit、main mergeは実施していない。今回の部分改善を汎用95–98%と報告しない。

Ruff pass。CPUテスト1,603 passed／1 skipped（22.82秒）、manifest／catalog 25コマンドpass。結果レポートcommitのGitHub Actionsはpush後に同じSHAで確認する。CI状態はこの文書自身のhashと循環させず、最終報告と外部CI記録に保存する。
