# 消費済みTRAIN全件の到達度

## 結論

初期モデルと4つの固定終点を、実際に消費した6,656問すべてで評価した。全5配列が独立processで完全一致。追加学習・checkpoint選択・auditアクセスは0。**TRAINにも高精度で適合しておらず、validationだけの崩れではない。** ただし容量・最適化・被覆不足のどれが原因かは、この測定だけでは確定しない。

同じ8,192入力回数の履歴を4runで照合。unique6,656、繰返し1,536、観測5,107、media asset3,114。初期モデルの学習量は0で、この集合で推論する比較用の参照。全91 TRAIN taskを含め、候補や問題を削除していない。

## 全消費TRAINのAccuracy（task/source macro）

| 固定モデル | Text | Image | Audio | Video |
| --- | ---: | ---: | ---: | ---: |
| initial | 20.39% | 0.00% | 15.34% | 38.28% |
| baseline | 25.87% | 21.92% | 57.40% | 42.19% |
| lstm | 27.01% | 19.58% | 57.01% | 41.21% |
| independent | 26.77% | 20.07% | 54.14% | 39.26% |
| matched | 26.08% | 21.39% | 56.58% | 40.43% |

Audioの集計に含むSTARSSの正例recallと全source/taskのAccuracy/NLL/Brier/ECEはJSONに保存。高い負例率を品質達成としない。occurrence加重の指標も別保存し、同一動画問の反復を独立事例と数えない。

## 共通課題の記述的gap

**共通43task、TRAINのみ48task、validationのみ28task**。全TRAIN91と全validation71の平均をそのまま汎化gapにしない。以下は共通taskだけの比較で、製品の全71task評価を置き換えない。task名が同じでも観測・正例率などの母集団は異なり、因果的な汎化効果ではない。

| モデル | modality | TRAIN Accuracy | validation Accuracy | TRAIN NLL | validation NLL |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | text | 30.21% | 26.99% | 1.39268 | 1.41435 |
| baseline | image | 21.92% | 19.73% | 2.11531 | 2.16229 |
| baseline | audio | 57.40% | 53.03% | 1.06137 | 1.18541 |
| baseline | video | 42.19% | 31.50% | 1.20467 | 1.34445 |
| lstm | text | 32.07% | 28.17% | 1.36824 | 1.42857 |
| lstm | image | 19.58% | 17.82% | 2.15744 | 2.20040 |
| lstm | audio | 57.01% | 53.17% | 1.07823 | 1.19003 |
| lstm | video | 41.21% | 33.25% | 1.20615 | 1.33748 |
| independent | text | 31.83% | 27.70% | 1.34682 | 1.38490 |
| independent | image | 20.07% | 19.82% | 2.12471 | 2.16112 |
| independent | audio | 54.14% | 51.11% | 1.08821 | 1.18959 |
| independent | video | 39.26% | 32.00% | 1.20690 | 1.32134 |
| matched | text | 30.67% | 27.56% | 1.36156 | 1.39789 |
| matched | image | 21.39% | 19.29% | 2.11772 | 2.16380 |
| matched | audio | 56.58% | 54.68% | 1.05407 | 1.15556 |
| matched | video | 40.43% | 34.25% | 1.20061 | 1.32878 |

## 次の変更を選ぶ根拠

既存の凍結TRAIN内で、同じsource/task/候補数のquotaを保ちつつ、Text・Image・Audioは各2,048問を未使用例へ置換できる。Videoは512問を全て消費済みで追加uniqueは0。この制約を隠してVideoのdata scalingと呼ばない。

次の限定比較は同じ初期readout・AdamW・2,048更新・8microbatches/updateで、元8,192回を2巡する対照と、2巡目の非Video6,144回を未使用TRAINへ置換する候補。新しい設定と出力を使い、両方とも新規初期化する。元runのresumeや単なる延長ではない。source/task/候補数の入力順序と総回数を揃え、唯一の介入を既存例の反復から未使用例への置換とする。learning curve、full validation、retention、unique assetsを記録し、改善しなくても保存する。この比較は本報告時点ではまだ未実施。

## 実行と再現

実行source `12b4d432f13ae30a17be3f73510ad7a68c075603`、config SHA `992ce174032fa9d1864f4d4ee86c64992910a78b68c2877add272df275032c57`。設定は`configs/pretrained_reuse/native_consumed_evaluation_v1.json`、プロトコルは[R2_NATIVE_CONSUMED_PROTOCOL](R2_NATIVE_CONSUMED_PROTOCOL_20261016.md)。実行形は`python scripts/evaluate_native_consumed.py --phase <smoke|probe|replay> --output <new-dir>`、probe/replayは直前receiptのpathとSHAを必須とする。正確な引数・時刻・hash・version・state digestは各receiptに保存。

| phase | process内秒 | peak CUDA bytes | receipt SHA256 |
| --- | ---: | ---: | --- |
| smoke | 12.71 | 127976960 | `8261ec122d60e1d7db0baaa1ebf7133c14b01db40b066e8de043a46e002c2b62` |
| probe | 106.63 | 177249280 | `fff31f2e563b577a14607c57146235041a034db379018e0a7c9b79442cd48964` |
| replay | 111.72 | 177249280 | `70041ebba3ce7669d84262d0d7b38eafb9e3eb81d40ee0df11ec4955abc02d9e` |

3process内wall計231.06秒。RTX3080 Laptop16GiB、FP32/TF32無効、Torch2.6.0+cu124、NumPy2.4.6。全Encoderの生入力処理を含む時間ではない。GPUはキャッシュ済み特徴の読出しだけを実行。クラウド費用0。

保存先：`C:/CodexArtifacts/pretrained-reuse-r2/native-consumed-{smoke,probe,replay}-01/`、分析`native-consumed-analysis-01/`。全5×6,656予測、完全なID会計、初期/固定終点の原本hash、全source/task指標を保持。全ての重みは読み取り専用、model state digest前後一致、旧artifact/config/corpus不変。既知の分割制約や欠けた正例は変えていない。

8件のmembership/共通task gapテストを追加。Ruff PASS、全CPU2,091 passed /1 skipped（30.15秒）。実行sourceのCI #1831 success（manifest検証を含む）。公開commitのCIもpush後に確認する。200M以下、native4モダリティ、任意条件、確率JSON、時間/共同/OOD、各必須課題95–98%という目標は未完了。
