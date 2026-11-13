# LSTMによる時間状態の追加：実装と実機接続の結果

2026-10-16。ユーザーの時系列メモリ提案に基づく別候補。
**接続・再現性PASS。判断精度の比較は未実施。製品採用は未判定。**
[事前設計と次の比較条件](R2_TEMPORAL_STATE_PROTOCOL_20261016.md) ·
[実測JSON](R2_TEMPORAL_STATE_RESULTS_20261016.json)。

## 何を追加したか

```text
凍結した動画特徴 → 空間位置ごとの時間列 → 小型LSTM → 元の全tokenへ残差を加える ┐
凍結した発話特徴 → 発話の時間列       → 小型LSTM → 元の全tokenへ残差を加える ├→ 条件・候補ごとの読出し → JSON
テキスト・画像・環境音のclip特徴 ─────────────────────────────────────────────┘
```

`native_temporal.py`の独立した構成で、既存`native_candidate.py`は変更していない。
各経路は192幅の投影・LayerNorm・1層の単方向LSTM・元の幅への投影。
初期の出力投影を0とし、既存readoutの初期出力と完全一致する。元の特徴を保持し、
最終状態一つだけへ情報を潰さない。映像は8時刻×256空間位置の全2,048tokenを維持する。
個々の空間位置に同じLSTMを使う構成なので、物体追跡を実装したとは扱わない。

低水準の状態は呼出し側へ明示して返す。入力間で自動的に持ち越さず、native wrapperは
blockごとに初期化する。欠損tokenは状態更新から除外する。時間順・空間対応・マスクの
検査を実装し、未来featureを変更しても過去のLSTM出力が変わらないことをテストした。
ただし既存Encoderはclip全体を参照できる。**モデル全体のオンライン因果性は未成立**。
異なる入力の相対座標を共通の実時計とはみなさない。

## サイズ

| 部品 | パラメーター |
|---|---:|
| 既存の全Encoder | 195,365,529 |
| 既存共通readout | 889,345 |
| 動画LSTM経路 | 591,936 |
| 発話LSTM経路 | 444,480 |
| **候補合計、量子化前** | **197,291,290** |

追加は1,036,416パラメーター／FP32 tensor 4,145,664 bytes。残余は2,708,710。
Encoderの数は以前の実ロード棚卸し、追加部分は今回の実インスタンスから数えた。
今回Encoderを再ロードした測定ではない。圧縮済みpackage容量は未測定。

## 実行したもの

既存TRAIN-only integration smokeの保存済み特徴と8問の候補vectorを使用した。
音声・映像は同じLombard発話1件、画像1件、著述text1件であり、品質検証に十分な母数でも
ラベル付き評価でもない。raw Encoder実行を繰り返さず、固定特徴からの接続を確認した。

- 初期値で8問すべて既存readoutと完全一致。候補順入替の対応も維持。
- 既存readoutを凍結し、動画＋発話を含む人工targetで**診断用2更新**。
  これは学習性能を測るrunではなく、実データの正解を使った学習回数は0。
- 最初の更新は0初期化した出力投影へ、2更新目はLSTMを含む全追加パラメーターへ
  非ゼロ・有限の勾配が到達。既存readoutの重みは変更なし。
- save/reloadと再推論の差0。別processで初期・診断後の全重みtensor、8問の出力、
  loss・勾配normが完全一致。各processの16入力pinと4保存artifact hashを再照合済み。
- CPUテスト22追加。全件 **2,029 passed / 1 skipped**、Ruff PASS、既存manifest検査25件PASS。
  依存を追加しない標準環境では2,009 passed / 11 skippedも記録。上記全件値は既存の
  optional audio依存パスを使った実行結果。

## 局所的な費用

RTX3080Laptop16GiB、PyTorch2.6系CUDA、FP32、TF32無効。
以下は最初の質問1つについて2回warmup後10回測ったp50。
**特徴保存後のheadだけの時間**で、Encoder・decode・転送時間を含めない。
先にbaseline、次にLSTMを測る短い診断であり、安定した製品latency benchmarkではない。

| 入力 | baseline | LSTM追加 | 別process baseline / LSTM |
|---|---:|---:|---:|
| 音声 | 2.20ms | 6.11ms | 2.12 / 6.01ms |
| 動画 | 2.09ms | 9.12ms | 1.94 / 8.97ms |

LSTMの追加費用は実際にある。パラメーターが小さいことから高速化を推定しない。
Pythonの入力検査とGPU同期もこの測定に入るためkernel単体時間ではない。
音声110token、動画2,048tokenというこの入力長以外の遅延は未測定。

診断逆伝播のpeak CUDA allocatedは160,061,952 bytes、reservedは234,881,024 bytes。
全Encoderが常駐した場合のVRAM値ではない。2回の実行内計測は2.701秒／2.754秒
（import・process起動を除外）。全作業・CPUテスト時間は別。クラウド費用0。

## 実行元と成果物

Source: `3f70918c63904a2b37470694ae395487e1baa5a5`。
設定: `configs/pretrained_reuse/native_temporal_probe_v1.json`。
Exact config hash・コマンド・開始終了時刻・環境・全artifact hashは実測JSONとreceiptに保存。

```powershell
python scripts/probe_native_temporal.py --output C:/CodexArtifacts/pretrained-reuse-r2/native-temporal-probe-01
python scripts/probe_native_temporal.py --output C:/CodexArtifacts/pretrained-reuse-r2/native-temporal-replay-01 --replay C:/CodexArtifacts/pretrained-reuse-r2/native-temporal-probe-01 --replay-sha256 9d8b383f2db414b4a35add1c6a8b6b4d8ec118ca44d0b58cc128d302f7169241
```

両方とも新しい出力先を要求する。再実行時は未使用のディレクトリ名にする。
既存ML依存・repo/srcをPYTHONPATHに設定し、OMP_NUM_THREADS=4で実行した。

- `C:/CodexArtifacts/pretrained-reuse-r2/native-temporal-probe-01/`
  receipt SHA256 `9d8b383f2db414b4a35add1c6a8b6b4d8ec118ca44d0b58cc128d302f7169241`
- `C:/CodexArtifacts/pretrained-reuse-r2/native-temporal-replay-01/`
  receipt SHA256 `27b835ccabf9d40caf784f94b07177e2f7254ec65f679d16cba1005a2db310b9`
- `C:/CodexArtifacts/pretrained-reuse-r2/native-temporal-verification-01/`
  CPUテスト・manifest・実行ログ。

`diagnostic-only.pt`は人工targetの勾配検査専用で、品質候補や初期化に流用しない。
旧corpus／Teacher／QAT／native特徴／checkpointを上書きしていない。
sealed audit・calibration・validationのpayloadは開いていない。

## 次の判断

同じdata・seed・sampler・選択規則のbaseline学習とLSTM比較が必要。
各モダリティの精度と保持、時間順変更、条件ペア、共同入力の取り違えを測る。
追加容量だけの効果を分けるため、同規模の非再帰residualも対照にする。
精度が上がらなければLSTMを採用しない。現状95–98%への到達根拠はない。

xLSTMの公式論文・実装を確認した。状態更新を強めるsLSTMと連想記憶を持つmLSTMが
次の候補だが、このPCでの実行・速度・精度はまだ測定していない。今回xLSTMは導入していない。
事前固定した小さな比較で必要性と利益が分かる場合に進め、アーキテクチャの広い総当たりはしない。
