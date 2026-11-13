# 条件意味の教師候補：TRAIN側での実測

判定：**実行PASS、今回の教師方式は採用条件に不合格。蒸留は開始しない。**
これは固定した非thinking・次トークン選択ラベル方式の診断であり、Gemma全体の能力上限、
小型モデルの限界、四モダリティ製品の品質を示す結果ではない。

[事前固定プロトコル](R2_SEMANTIC_TEACHER_PROTOCOL_20261015.md)・
[全指標とハッシュのJSON](R2_SEMANTIC_TEACHER_RESULTS_20261015.json)・
[戦略](STRATEGY_20261015.md)。

## 実行したこと

- TRAIN側の既存2源だけから、seed17のハッシュ順位で各32状態を選択。
- 正しい構造化状態を与える40択の条件判断640問と、既存の4／5択の数値判断512問。
  計1,152提示／64状態。条件を消す対照や候補反転を含むため、1,152独立例ではない。
- Gemma4 E2B IT QAT-unquantizedの固定revision
  `6befbaca7398925921802abd1f277b495b78b738`を新規ロード。
  BF16、学習可能パラメーター0、全常駐パラメーター**5,104,297,504**。
  旧Teacherのadapterは使用していない。今回量子化も行っていない。
- 各promptの候補ラベルが単一トークンになることを検証。bindingは`A..Z,a..n`の40個、
  numericは先頭4／5個。最大643tokens、切捨て0。
  出力は候補内softmax、温度1。全語彙上の候補ラベル確率質量も保存。
- 別プロセスの36問smoke、本実行、その後の事前指定36問の再ロード確認を完了。
  再確認は全logit・確率・予測・許可ラベル質量を含め完全一致。全1,152問を再実行したわけではない。
- 保存済み全予測から別のCPU処理で指標を再集計。条件なし32組の各4問は、全logitが同一。
  正解やtask metadataがprompt経路に混入していないことも確認した。

## 実測

下表はTRAIN診断であり、他の小型モデルの検証値と差し引いて改善幅を計算しない。

| 条件 | 件数 | Accuracy | NLL | Brier | ECE | 平均confidence |
|---|---:|---:|---:|---:|---:|---:|
| 対象を明示 |128|98.4375%|0.055527|0.020230|0.045659|0.957711|
| 他3項目を除外 |128|81.2500%|0.722847|0.295613|0.059713|0.831578|
| 元の要求を取消し、新対象を指定 |128|98.4375%|0.048206|0.016631|0.039876|0.963215|
| 除外条件、候補順序を反転 |128|42.9688%|2.477375|0.910468|0.411050|0.831357|
| 条件文なし |128|25.0000%|5.632706|1.394967|0.679662|0.929662|
| 数値判断、元の候補順序 |256|54.6875%|1.354482|0.676280|0.236503|0.778582|
| 数値判断、候補順序を反転 |256|53.5156%|1.380629|0.681595|0.242764|0.774675|

各32問の属性別Accuracy：

| 条件 | color | texture | count | shape |
|---|---:|---:|---:|---:|
| 対象明示 |100%|100%|100%|93.75%|
| 除外 |100%|71.875%|78.125%|75%|
| 要求切替 |100%|100%|100%|93.75%|
| 除外＋候補反転 |53.125%|31.25%|6.25%|81.25%|

数値側の元順序：極値128問61.7188%、数値の時間順序64問54.6875%、
閾値以下の個数32問31.25%、閾値条件付き選択32問50%。
数値のtimeフィールドは、実動画の時間理解を意味しない。
全taskのNLL/Brier/ECE、macro値はJSONへ収録した。

条件変更ペアの両問正解率は、対象明示／要求切替96.875%、除外64.5833%、
除外＋反転15.625%、条件なし0%。binding各条件192ペア、独立状態は32。

### 候補順序と不確かさ

- 同じ意味の候補IDに合わせた回答一致率：binding **41.4063%**、numeric **51.9531%**。
  いずれも事前基準99%に不合格。最大確率差は0.992934／0.970029。
- 除外の反転によるAccuracy差−38.2813ポイント。
  状態単位paired bootstrapの95%区間は[−46.0938, −29.6875]ポイント。
- 数値の差−1.1719ポイント、同区間[−9.3750, +7.4219]ポイント。
  集計Accuracyが近くても、個々の回答の順序安定性は成立していない。
- 32状態単位、復元抽出10,000回、NumPy RNG seed17、percentile法。
  TRAIN側の固定方式の診断であり、盲検品質や未知構文への転移の証拠ではない。

許可ラベルへのvocabulary確率質量の平均は対象明示0.999450、除外0.996568、
反転除外0.998040、数値元順序0.969624だった。
したがって「指定ラベルをほとんど出さなかったので失敗した」という単純な説明は合わない。
ただし、位置・ラベル割当・条件理解のどれが原因かは今回だけでは区別できない。

## 再現性・環境・費用

推論source commit：`1c05f29ea10a8798a3b3ca879a4987759d86866d`。
実行作業ディレクトリ：`C:/CodexWorktrees/pretrained-reuse-omni`。
Python：`C:/Users/junny/AppData/Local/Programs/Python/Python311/python.exe`。

```powershell
python -m scripts.evaluate_semantic_teacher --stage prepare --output C:/CodexArtifacts/pretrained-reuse-r2/semantic-teacher-corpus-04
python -m scripts.evaluate_semantic_teacher --stage smoke --corpus C:/CodexArtifacts/pretrained-reuse-r2/semantic-teacher-corpus-04 --output C:/CodexArtifacts/pretrained-reuse-r2/semantic-teacher-smoke-01
python -m scripts.evaluate_semantic_teacher --stage full --corpus C:/CodexArtifacts/pretrained-reuse-r2/semantic-teacher-corpus-04 --output C:/CodexArtifacts/pretrained-reuse-r2/semantic-teacher-full-01
python -m scripts.evaluate_semantic_teacher --stage replay --corpus C:/CodexArtifacts/pretrained-reuse-r2/semantic-teacher-corpus-04 --reference C:/CodexArtifacts/pretrained-reuse-r2/semantic-teacher-full-01 --output C:/CodexArtifacts/pretrained-reuse-r2/semantic-teacher-replay-01
python -m scripts.publish_semantic_teacher_probe --root C:/CodexArtifacts/pretrained-reuse-r2 --output C:/CodexArtifacts/pretrained-reuse-r2/semantic-teacher-publication-01.json
```

実際のPython絶対パス、全引数、開始／終了時刻は各result.jsonに保存。
既存出力先は上書き拒否。再実行する場合は新しい出力先を指定する。

本実行UTC06:48:20.617198→06:52:00.071986、全件完了で終了。
RTX3080Laptop16GiB、Windows、driver616.92、Python3.11.9、PyTorch2.6.0+cu124、
CUDA12.4、Transformers5.6.2、safetensors0.8.0、tokenizers0.22.2、NumPy2.4.6。
SDPA math、BF16、process内だけでsafetensors preadを選択。依存更新なし。
重みのmissing/unexpected/mismatched/error keysはすべて0。

本実行219.46秒（ロード8.78秒、推論185.40秒）、ピークVRAM10,367,032,320bytes、
ピークhost11,186,249,728bytes。smoke41.48秒、replay41.29秒。
事前準備4回を含む記録済みprocess本体時間は470.37秒、推論部分合計197.93秒。
これはGPU占有時間や他プロセスと分離したベンチマークではない。クラウド費0。

推論前レビューで、部分予測の逐次保存、再現元のcode/model/corpus照合、失敗行の保存を補強した。
準備01〜04はその各版を残したもので、**rows.jsonは全てbyte一致**。結果を見た設定変更ではない。
GPU推論は上記最終版だけで行い、参照した既存重み・TRAINデータは前後ハッシュ一致。

| 成果物 | SHA256 |
|---|---|
| corpus04 result | `73ef542fccf9d82388a008a5b9c9402b3886ea7424d4dd3c4ed3585ff7b97cde` |
| corpus rows | `8b674f8cda7ac74e34aaf643bae2d40b6e534b2b1fd980fb3d8cacc1411229a2` |
| smoke result | `182587a9a45057393f863b8cee93553f00ca138b8c6fa6648fcf4a2cc98641f4` |
| full result | `94d28aec4f15a05a09275aa1c30ee540428c20fa8deda31769435029655016ac` |
| replay result | `9b77592d4ad03d0732474f4942d307836464b16d2e3d0f3996fde2d9d2d42c6f` |
| publication JSON | `37c1044b6754a35bd9c393720c3d471646ed6cd14392638a9af03eff682919c3` |

## 次の戦略と未実施

大きい教師を既定で蒸留する方針には進まない。教師を使えるかも実験対象とする。
先に、条件意味と候補の位置／ラベルを切り分ける一回の事前固定診断を置く。
例えば同じ意味の全候補を個別に採点する方式は別のインターフェース仮説になるが、
この観測済み集合で文面や温度を探索して成功を主張しない。思考有りの能力は今回未測定。

独立した条件構文・候補説明・状態組合せのデータ契約を優先し、正解ラベルを基準とする。
教師が実測で信頼できる能力に限り、同じデータ・初期化・予算で補助蒸留を比較する。
プログラムで検算した正解を、弱い教師の数値回答に置き換えることはしない。

小型studentの新規学習、実画像／音声／動画の今回の再評価、sealed audit、
追加seed、prompt探索、製品昇格、量子化、mainへのmergeは実施していない。
最終目標の全常駐200M以下・各必須課題95–98%・native四入力・共同判断・JSONは引き続き未達。

## コード検証

Ruff全体PASS、CPU pytest **1,618 passed /1 skipped（23.13秒）**、
manifest25コマンドすべてPASS。GPU／モデル実行は通常CIの外で上記の別processとして行った。
固定TRAIN選択、完全group、正解不整合拒否、promptからの教師情報隔離、候補数、反転対応、
指標、再現元の一致、再現対象36IDの不足・重複・順序違い拒否をテストした。
独立レビューでも保存結果・判定・不確かさ・報告範囲を確認した。
