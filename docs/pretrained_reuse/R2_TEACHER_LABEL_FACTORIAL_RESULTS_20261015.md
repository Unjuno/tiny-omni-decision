# 教師の表示順とラベル割当：固定二要因比較

2026-10-15。[事前固定仕様](R2_TEACHER_LABEL_FACTORIAL_PROTOCOL_20261015.md)に従い、
同じTRAIN64状態・384問を4条件、計1,536提示で測定した。追加学習・評価データ選別はない。
**実行PASS、教師の採用なし。製品品質は未達。**

## 実測

| source / arm | 表示順 | ID→文字 | 件数 | Accuracy | NLL | Brier | ECE | 平均confidence |
|---|---|---|---:|---:|---:|---:|---:|---:|
| binding / 00 | 元 | 元 | 128 | 81.25% | 0.722847 | 0.295613 | 0.059713 | 0.831578 |
| binding / 10 | 反転 | 元 | 128 | 59.38% | 1.812174 | 0.676543 | 0.250074 | 0.772492 |
| binding / 01 | 元 | 反転 | 128 | 74.22% | 0.848056 | 0.379452 | 0.095945 | 0.795730 |
| binding / 11 | 反転 | 反転 | 128 | 42.97% | 2.477375 | 0.910468 | 0.411050 | 0.831357 |
| numeric / 00 | 元 | 元 | 256 | 54.69% | 1.354482 | 0.676280 | 0.236503 | 0.778582 |
| numeric / 10 | 反転 | 元 | 256 | 52.73% | 1.359230 | 0.682873 | 0.221386 | 0.743250 |
| numeric / 01 | 元 | 反転 | 256 | 54.30% | 1.332929 | 0.666614 | 0.218331 | 0.750612 |
| numeric / 11 | 反転 | 反転 | 256 | 53.52% | 1.380629 | 0.681595 | 0.242764 | 0.774675 |

bindingは40択の属性除外、numericは4/5択の極値・数え上げ・条件選択・数値上の時間順序。
後者は動画の時間理解ではない。全タスク別指標、macro、条件ペア、合法ラベル確率質量、
全対比のNLL/Brierと区間は[保存JSON](R2_TEACHER_LABEL_FACTORIAL_RESULTS_20261015.json)に含む。

### 事前指定したAccuracy差

単位はポイント。各source32状態を対応付け、NumPy seed17、10,000回の状態単位bootstrap、
percentile95%区間。観測済みTRAIN内の診断であり盲検品質ではない。複数比較補正はしていない。

| source | 対比 | 差 | 95%区間 |
|---|---|---:|---|
| binding | order_with_original_mapping | -21.88 | [-30.47, -12.50] |
| binding | order_with_reversed_mapping | -31.25 | [-42.19, -18.75] |
| binding | mapping_with_original_order | -7.03 | [-14.06, -0.78] |
| binding | mapping_with_reversed_order | -16.41 | [-24.22, -8.59] |
| binding | interaction | -9.38 | [-19.53, +1.56] |
| numeric | order_with_original_mapping | -1.95 | [-8.20, +4.69] |
| numeric | order_with_reversed_mapping | -0.78 | [-4.69, +3.12] |
| numeric | mapping_with_original_order | -0.39 | [-7.03, +6.64] |
| numeric | mapping_with_reversed_order | +0.78 | [-4.69, +6.25] |
| numeric | interaction | +1.17 | [-5.08, +7.42] |

## 解釈と戦略

1. **表示順そのものに弱い。** bindingでは割当を固定して表示だけ反転しても−21.88ポイント。
   NLL差+1.089327、95%区間[+0.632110,+1.574040]、Brier差+0.380930、
   区間[+0.231335,+0.531237]。ラベル文字を付け替えただけの問題とは解釈できない。
2. **ラベル割当も影響する。** 元表示順で割当だけを反転すると−7.03ポイント。ただしそのNLL/Brierの
   区間は0を含む。反転表示下では−16.41ポイント。Accuracyの交互作用−9.38ポイントの区間も0を含む。
   一つの要因だけで全失敗を説明せず、条件依存の影響として記録する。
3. **数値判断の低さは残る。** 全条件52.73–54.69%。各対比のAccuracy/NLL/Brier区間は0を含む。
   ただし候補IDで対応付けた回答一致率は60.16–73.83%で、平均点が近くても安定しているわけではない。
4. これは固定した非thinking・一文字次トークン方式と小さなTRAIN集合の結果。非アルファベット順の
   提示の不慣れ、特定の文字／大小文字、一般的推論能力は別々に識別していない。
   世界知識不足、200Mの能力上限、全Gemmaモデルの限界とは結論しない。
5. **この教師のラベルlogitを既定の蒸留信号にしない。** ラベル・文面・seedの探索をここで閉じる。
   小型studentは既に候補ごとの採点を使うため、この診断だけでstudentの条件理解が直るわけではない。
   次は検算できる正解と条件対照を中心とした独立学習契約へ進む。教師を使う場合も実測で有効な能力に限る。

[次の三つの成果物と打ち切り規則](NEXT_STRATEGY_20261015.md)。

## 再現性・保護・実行環境

- 実験コード: `a49748a715b178e41fbb0c721c6862a5c6e4b60e`。全code/config/input/modelのSHA256は保存JSONに記録。
- prepareで過去768提示の文面・入力トークン・継続ラベルIDが一致。本実行で全768件のlogit差は0。
- 事前指定48問のsmokeと別process replayは、今回fullの全予測フィールドと完全一致。
  全1,536件の独立二回目実行とは呼ばない。
- 表示順・文字割当以外の状態、正解、問題文、候補説明、40/4/5択を維持。
- 推論前レビューで、旧集計関数のID規則と新factorial IDの不一致を修正。
  新しい局所集計viewだけで対応し、旧コードと保存IDは変更しない。CPU回帰を通してからprepare/GPUを実行した。
- 旧教師・旧TRAIN・過去出力は参照のみ。run前後のハッシュ確認と出版時の照合に合格。
- validationとsealed auditは未読、optimizerなし、trainable0、更新0。依存更新・モデル保存・クラウド費なし。
- 参照モデル: `google/gemma-4-E2B-it-qat-q4_0-unquantized`、revision `6befbaca7398925921802abd1f277b495b78b738`。
  実ロード5,104,297,504パラメーター。教師専用であり200Mの製品候補ではない。
- RTX3080Laptop16GiB、Windows、driver616.92、Python3.11.9、Torch2.6.0+cu124、CUDA12.4、
  Transformers5.6.2、safetensors0.8.0、tokenizers0.22.2、NumPy2.4.6。
  BF16、SDPA math、batch1、seed17、temperature1、process内Windows pread。
- 本実行UTC `2026-10-15T07:28:11.061667+00:00` → `2026-10-15T07:32:17.536599+00:00`、全件完了で終了。

| stage | process本体秒 | load秒 | inference秒 | peak VRAM bytes | peak host bytes |
|---|---:|---:|---:|---:|---:|
| teacher-label-factorial-corpus-01 | 41.643 | - | - | - | 1000517632 |
| teacher-label-factorial-smoke-01 | 42.077 | 9.808 | 6.845 | 10366917120 | 12012470272 |
| teacher-label-factorial-full-01 | 246.475 | 9.475 | 210.235 | 10366917120 | 12013780992 |
| teacher-label-factorial-replay-01 | 41.728 | 9.384 | 6.790 | 10366917120 | 12016992256 |

記録済み4process本体の合計371.922秒、推論部分223.870秒。CPU検証・出版時間や他processとの
干渉を含めた専用GPU占有ベンチマークではない。クラウド費0。

## コマンドと成果物

作業場所`C:/CodexWorktrees/pretrained-reuse-omni`。`python`は各runのcommandに記録されたPython311を指す。
出力先は全て新規。再実行時は別出力先を使う。

```powershell
python -m scripts.evaluate_teacher_label_factorial --stage prepare --output C:/CodexArtifacts/pretrained-reuse-r2/teacher-label-factorial-corpus-01
python -m scripts.evaluate_teacher_label_factorial --stage smoke --corpus C:/CodexArtifacts/pretrained-reuse-r2/teacher-label-factorial-corpus-01 --output C:/CodexArtifacts/pretrained-reuse-r2/teacher-label-factorial-smoke-01
python -m scripts.evaluate_teacher_label_factorial --stage full --corpus C:/CodexArtifacts/pretrained-reuse-r2/teacher-label-factorial-corpus-01 --output C:/CodexArtifacts/pretrained-reuse-r2/teacher-label-factorial-full-01
python -m scripts.evaluate_teacher_label_factorial --stage replay --corpus C:/CodexArtifacts/pretrained-reuse-r2/teacher-label-factorial-corpus-01 --reference C:/CodexArtifacts/pretrained-reuse-r2/teacher-label-factorial-full-01 --output C:/CodexArtifacts/pretrained-reuse-r2/teacher-label-factorial-replay-01
python -m scripts.publish_teacher_label_factorial --root C:/CodexArtifacts/pretrained-reuse-r2 --output C:/CodexArtifacts/pretrained-reuse-r2/teacher-label-factorial-publication-01.json
```

| artifact | SHA256 |
|---|---|
| `C:\CodexArtifacts\pretrained-reuse-r2\teacher-label-factorial-corpus-01\result.json` | `100fbfa614b814c068a329651c694fb45374a1902137f510e149effda431c0aa` |
| `C:\CodexArtifacts\pretrained-reuse-r2\teacher-label-factorial-smoke-01\result.json` | `094b1ab669b432bcd3beb24bca444238dcc7032826460b731aaacec02a84d37e` |
| `C:\CodexArtifacts\pretrained-reuse-r2\teacher-label-factorial-full-01\result.json` | `4586eaf78f8388090079ebba53dbbfc2d2c1e7fea4114339e30b171368008cd1` |
| `C:\CodexArtifacts\pretrained-reuse-r2\teacher-label-factorial-replay-01\result.json` | `22d4f04fd86037fedba173c8b1776750727c60cb484b94796e1a8e6b82d56f6c` |
| `C:/CodexArtifacts/pretrained-reuse-r2/teacher-label-factorial-corpus-01/rows.json` | `19875d126a2b100a59bfe4ba7fc5f6b26fe38cefbf5c7902ad25cc9f42020054` |
| `C:/CodexArtifacts/pretrained-reuse-r2/teacher-label-factorial-publication-01.json` | `b33f1dca70e1e8e5a6e894425600efa51cb074976a90f33741df8631fef315e0` |

## 検証と未実施

Ruff全体PASS、CPU pytest **1,625 passed /1 skipped（25.69秒）**、manifest検証25コマンドPASS。
manifest初回は検証ハーネスが存在しないpackage __main__を指定して全25件が実行前失敗。
正しい`tiny_omni_decision.cli`で再実行し全件PASS。両ログmanifests-01/02は保持し、ソースやmanifestは変更していない。
新規CPUテストは二要因独立性・過去入力一致・ラベル／target隔離・集計・対比・48件の不足／重複／順序違い・
過去出力改変拒否を対象とする。GPU実行は通常CIの外。

今回、新たなstudent学習、native四モダリティ再評価、最終監査、追加seed、量子化、製品昇格、main mergeは未実施。
全常駐200M以下・各必須課題95–98%のOmni製品は未完成。
