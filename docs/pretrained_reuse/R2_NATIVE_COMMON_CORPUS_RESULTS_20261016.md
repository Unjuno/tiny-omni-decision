# 共通native教師信号と質問特徴：全件の対応付け完了

2026-10-16。**共通corpus構築・質問特徴の追加・全件再現PASS**。
[固定手順](R2_NATIVE_COMMON_CORPUS_PROTOCOL_20261016.md) ·
[実測JSONとhash](R2_NATIVE_COMMON_CORPUS_RESULTS_20261016.json)。
この単位ではoptimizer更新0、品質評価なし。学習結果は別のrunで測定する。

| 入力 | TRAIN質問 | 開発検証質問 | 設定 |
|---|---:|---:|---|
| Text | 18,432 | 10,732 | 元の6ファイル、質問・候補・正解を保持 |
| Image | 33,696 | 2,048 | 8,424 / 512画像、各4問、各40候補 |
| Speech | 24,480 | 5,334 | 4,080 / 889発話、6項目、25文字を含む全語彙 |
| Environmental audio | 23,460 | 5,832 | 1,955 / 486窓、全12クラスのbinary質問 |
| Video | 512 | 400 | 256 / 200scene、静的・時間的問題を保持 |
| **合計** | **100,580** | **24,346** | 観測単位17,341 / 3,843 |

観測単位の総数21,184は、text状態・画像・発話・音声window・video sceneの合計。
音声windowは独立録音数と同一ではない。検証の71課題区分のうちTextは既存47区分。
既知のgroup/media/normalized-text分割ゲートを維持し、元ファイルを上書きしていない。
これは観測済み開発データの統合であり、盲検評価ではない。

全観測は既存の完全native特徴へ対応する。特徴参照に正解・課題名・option IDを混入させず、
質問と候補からのみ言語vectorを指定する。元のcandidate順とtarget indexの対応を検査した。
Imageのみ元の固定semantic質問・40候補を使い、seed17の画像IDによる順序を固定した。
元の画像datasetラベル`val`と共通名`validation`の差で初回構築は停止した。読み込み境界に
明示的なaliasと回帰テストを追加し、元データを変えず別出力で再構築した。

## 言語特徴の整合性

共通の質問／候補は24,950種類。既存22,328種類のvectorはそのまま再利用し、
不足分2,622種類だけ、元のDeBERTa FP32/eager/batch1・切捨てなしの処理で追加した。
TRAIN32種類のsmokeと既存TRAIN1種類の完全一致を確認後、全件計算を実行した。
別processで新規2,622種類をすべて再計算し、保存vectorとの最大差は0。
旧runと同じencoder state digest・package version・runtime実装hashを確認し、
開始前後の凍結重みdigestも一致した。モデルや環境のupgradeはしていない。

| 実行 | 秒 | peak CUDA allocated | OS peak working set |
|---|---:|---:|---:|
| corpus構築 | 26.02 | GPU不使用 | 未測定 |
| TRAIN query smoke | 9.06 | 321,163,776 bytes | 2,029,666,304 bytes |
| 全query追加 | 60.06 | 321,213,952 bytes | 2,032,521,216 bytes |
| 全query fresh replay | 60.37 | 321,213,952 bytes | 1,849,716,736 bytes |

process内のwall時間で、起動前importを含めない。CPUテストと一部重なったため専有速度比較ではない。
初回失敗9.60秒も保存。クラウド費用0。新規query tensorは4,027,392 bytes、元の特徴は複製していない。

## 残る品質制約

STARSSのTRAIN bell正例0、validation water-tap正例0を保持した。負例だけの高Accuracyを
当該classの成功にしない。検証話者・scene・roomの独立数、同じ観測に複数質問があることを
考慮して解釈する。Genuine joint、独立OOD、日本語、自然画像、自由発話の不足は変わらない。
既存47区分の不採用判定や95–98%の製品要件も変えていない。

次は[固定した基準学習](R2_NATIVE_BASELINE_FIT_PROTOCOL_20261016.md)。同じcheckpointで
全24,346問を測定する。モダリティ・ソース・課題ごとの重みを明示し、STARSSの正例数も報告する。
LSTMの接続成功を精度向上とはみなさず、基準学習後の同条件比較で判断する。

## 再現情報

corpus成功source `4bbeb324754356c302cc015c9f308093d12fc5ce`。
query実行source `d462e436619012e02a9e40fd95fe10ebd911f194`。
corpus SHA `47a6d2f3495810bac58f57dec1a962f8b490fa409c75671b46bb2999fde87d49`。
query cache SHA `7f5ffd7362e84770b864a189d50bc29f43be3738f54a07f6dbdc21f1ef7d909e`。
query replay SHA `2a943cbbb5966ecee8c76b9d1cf6cd3fab108af4acfc95677beae51d3f7a0686`。

出力は`C:/CodexArtifacts/pretrained-reuse-r2/`以下の
`native-common-corpus-01`（保存した失敗）、`native-common-corpus-02`、
`native-common-queries-smoke-01`、`native-common-queries-cache-01`、
`native-common-queries-replay-01`。各receiptに実行引数・入力hash・出力hash・時刻を保存。
固定設定は`configs/pretrained_reuse/native_common_corpus_v1.json`と
`native_common_queries_v1.json`。再実行には未使用の出力先を使う。

corpus/query用25テスト追加後、全件2,054 passed / 1 skipped、Ruff PASS、manifest25件PASS。
続く学習runnerの検証結果は別に記録する。
