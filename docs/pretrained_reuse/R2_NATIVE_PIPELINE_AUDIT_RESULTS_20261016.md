# 学習データとパイプラインの実物監査

## 結論

追加学習の前に、実際のコーパス・特徴・消費履歴・損失・評価を監査した。
**固定学習集合の正例不足と、前処理による視野・時間情報の削減が確認された。**
ラベルの取り違えや勾配蓄積の不具合を今回の検査で裏付ける結果はない。
ただし、元ファイルとの一致を「データが良い」「正解が入力から判断できる」とは扱わない。
現段階で、低精度の原因を容量不足・LSTMの不適合・学習時間だけに帰属しない。

既存32,768-step run、best8192、最終checkpoint、コーパス、特徴は保存したまま。
今回のoptimizer更新は0、GPU使用なし、校正・sealed auditのpayload読取なし。
各必須課題95–98%、量子化前総パラメーター200M以下という目標は未達のまま維持する。

## 1. 学習に届いていない音声の正例

STARSSの同一12イベントについて、**利用可能なTRAIN全体**と**実際に消費したunique問**を分けた。
分母は各クラスの質問数。繰り返した回数をunique正例数に加えていない。

|イベント|TRAIN全体の正例/問|消費unique正例/問|validation正例/問|
|---|---:|---:|---:|
|speech|1631/1955|70/85|311/486|
|clapping|9/1955|**0/86**|10/486|
|telephone|27/1955|**0/85**|19/486|
|laughter|273/1955|6/85|56/486|
|domestic sounds|249/1955|10/86|267/486|
|footsteps|105/1955|3/85|19/486|
|door|60/1955|6/86|17/486|
|music|339/1955|14/85|229/486|
|instrument|73/1955|1/85|5/486|
|water tap|28/1955|**0/85**|0/486|
|bell|**0/1955**|**0/86**|32/486|
|knock|10/1955|**0/85**|4/486|

拍手・電話・水道・ノックは**既存TRAINに正例があるが、固定poolに入っていない**。
ベルはTRAIN全体にも正例がない。この2種類の不足を分けて対処する必要がある。
水道はvalidation正例もなく、現在のvalidationから陽性検出性能を測れない。
instrumentのunique正例1件を32回見せても、独立した32件の正例にはならない。

実消費STARSSは1024 unique問、うち正例110問。総32,768提示では正例3520提示。
単に全問「no」でもTRAIN消費集合89.26%、validation83.38%のAccuracyになる。
既存モデルのSTARSS validation Accuracyはbest8192で87.84%、終点で86.80%。
この比較は追加推論ではなく保存済み予測からの集計であり、非speechイベントの
低recallを隠す根拠にはしない。class別recall/specificityを引き続き併記する。

元アノテーションを独立に読み、2441窓・29,292問のhalf-open区間とラベルを再計算した。
不一致0。公式13クラス中の男女speechを統合する既存定義も照合した。
ただし「5秒内に100msの活動が一度でもある」正例を含む。
活動時間合計0.5秒以下はTRAIN151/2804正例、validation48/969正例。
全区間を1ベクトルに集約するevent経路の情報保持は、この一致検査では保証できない。
speech経路は時系列トークンを保持している。実際の音の再聴取による全件再注釈は未実施。

## 2. 画像・動画の視野と時間の削減

|経路|実物|前処理で残る領域|時間方向|
|---|---|---|---|
|Image|TRAIN8424 + validation512枚、640×480|中央480×480相当、**75%**|静止画1枚|
|Video|TRAIN256 + validation200本、480×320|中央320×320相当、**66.67%**|128フレーム→8フレーム|

画像の左右各80px、動画の左右各80pxがモデル入力から除かれる。
動画は全456本が128frames/25fpsで、抽出位置は
`[0, 18, 36, 54, 73, 91, 109, 127]`、最大間隔0.76秒。
8/128=6.25%は「使用するフレーム数の割合」であり、「保持した意味情報の割合」ではない。

実行経路は`_decode_video -> _preprocess -> VideoPrism -> token_grid -> FeatureStore`。
新しい特徴抽出で設定が変わったのではなく、以前からの前処理をそのまま追跡した結果である。
TRAIN manifest先頭sceneを予測に依存せず可視化したところ、元フレームにある右端の球や
左端へ移動した円柱がクロップ範囲の外になる時点を確認した。
[固定先頭sceneと保持領域](C:/CodexArtifacts/pretrained-reuse-r2/native-pipeline-audit-visual-01/video-input-geometry.png)。

コーパスにはcollision、before/after、enter/exit、beginning/endを問う問題がある。
これらに対して情報を捨てる前処理は優先的な検証対象になる。
**何問の正解がクロップ・間引きによって判別不能になるか、精度への因果効果はまだ不明。**
クロップされた面積をそのままラベル誤り率や性能上限へ換算しない。
画像についても、削った領域が背景だけなら正解情報は残り得る。

## 3. ラベル・特徴・勾配・評価で確認できたこと

|検査|対象|結果|
|---|---:|---|
|質問・候補・query hash・target index・split対応|124,926問|矛盾0、同一公開入力で異なるtargetも0|
|構造化textの独立正解計算|25,600問|不一致0|
|元STARSSアノテーションから区間ラベル再計算|29,292問|不一致0|
|特徴bankのhash・shape・dtype・finite|466bank|一致|
|保存sliceと実際のFeatureStore入力|21,184 observation|完全一致|
|query文字列→offset→実テンソル|24,950 vector|一致、異なるpairの完全同一vector0|
|実際の先頭8 microbatchでshuffle/蓄積勾配と平均損失を比較|4 modalityを含む|最大差2.19e-8、許容誤差内|
|train/eval modeと候補並べ替えのlogit対応|同じ実入力|最大差5.96e-8|
|保存予測からAccuracy/NLL/Brier/ECE等を独立再計算|全24,346 validation、10地点、5660数値|最大差8.88e-16|

勾配検査はCPU、コピーした初期readoutで実施し、optimizer.stepを呼んでいない。
全モデル重みが変化していないことも検査した。GPUカーネル全体の再検証ではない。
既存GPU実行については前回のbest/final完全replayの記録を保持する。

旧textの`original`計3564問（TRAIN1024、validation2540）は、元ラベルとの一致まで。
主観的なseverity、曖昧な自然言語判断を含み、独立正解計算は**UNKNOWN**。
構造化textも、生成ruleと目標の計算は検査したが、全表現の人手意味監査ではない。
Image/Videoの全件人手再注釈、全フレームのイベント再構成も未実施。

## 4. タスクと消費量の不足

利用可能TRAIN100,580問に対し、long runは**6656 unique問を262,144回提示**した。
Videoは256scene/512問を各128回。他のunique問は各32回。
既存の長期実験は「同じpoolへの追加最適化」の検証として保存する。
この結果を、十分な種類のデータを学習し尽くした証拠にはしない。

現在のobservationはtextのみ、imageのみ、videoのみ、または同じ音声の
speech/eventという2経路。**独立した音声と映像を組み合わせて初めて解ける例は0**。
自然言語の質問とmediaの条件付けはあるが、複数センサー間の統合判断の教師はない。
統合入力を受け付けられるコードと、統合判断を学習済みであることを分ける。

task文字列ではTRAIN専用48、validation専用28、共通43。
このうち一部は意図した未知の言い回しであり、「28の能力を全く教えていない」とは言えない。
validationをTRAINへ移す、難問を除く、taskを統合して精度を上げる変更はしない。
textの38長文状態は510-token窓と32-token overlapで保持されるが、
全token保持と全文脈の一括理解は同じではない。

## 5. 次の実験順序

1. **実際のsampler出力に対する正例・負例・独立assetの被覆を必須検査にする。**
   利用可能TRAINの正例が消費されるpoolを別versionで作る。ベルのように正例自体が
   ない場合は、許可済みの独立TRAINソースから補う。見つからない場合は未達のまま記録。
   各条件は同じ予算・readout・validationで比較し、重複をuniqueとして扱わない。
2. **視野を保持する前処理を別の一要因比較にする。** 最初は同じ8フレーム・モデルで
   中央cropとの比較を準備する。画素/座標のsmokeを先に行う。
   frame密度増加を同時に混ぜない。ImageとVideoも変更箇所を分ける。
3. その後、時間解像度・時系列処理を比較する。LSTM/xLSTMは候補として保持する。
   入力の不足が残ったまま短期runで棄却しない。長い予算と固定観測点を先に定義する。
4. 共同入力の教師例と未使用の検証を、公開・利用許可済みデータで設計する。
   実用タスクの正解が複数入力に依存することを、片方を隠す対照で検証する。
5. regularizationや容量の変更は上の要因と分ける。今回の監査から
   weight decay、学習率、encoder解凍の最適値を推測しない。

**上記の新しい学習・cache作成・モデル採用は今回未実施。**
既存runの設定を変更せず、次の限定実験の根拠を固めた。

## 再現・成果物

主監査の初回sourceは`c86524a49a1399a3e68824a14fe9c5f2aa878786`。
画像全件の実物geometryと独立評価を追加したsourceは
`69e6068c5cd38bc61a83238a3bfc6f6002ebe8f3`。
以前の監査も上書きせず残した。設定・入力・実行sourceのSHAは各receiptに記録。

```powershell
$env:PYTHONPATH='C:/CodexArtifacts/pretrained-reuse-r1/python-deps;C:/CodexArtifacts/pretrained-reuse-r2/native-speech-deps-01;C:/CodexWorktrees/pretrained-reuse-omni/src;C:/CodexWorktrees/pretrained-reuse-omni'
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='4'
python -u -m scripts.audit_native_training_pipeline --output C:/CodexArtifacts/pretrained-reuse-r2/native-pipeline-audit-02
python -u -m scripts.audit_native_saved_metrics --output C:/CodexArtifacts/pretrained-reuse-r2/native-pipeline-metrics-audit-02
```

再実行では未使用の出力先を指定する。既存出力への上書きは拒否される。
`native-pipeline-audit-02`にsource/label/coverage/feature/gradient/geometryの実測、
`native-pipeline-metrics-audit-02`に保存予測の独立再計算を保存。
`native-pipeline-audit-visual-01`にTRAIN先頭sceneの図と実行scriptを保存。
小さい結果とreceipt hashは[JSON](R2_NATIVE_PIPELINE_AUDIT_RESULTS_20261016.json)。
検証ログとCI記録は`C:/CodexArtifacts/pretrained-reuse-r2/native-pipeline-verification-01/`。

最終の全体監査60.71秒、独立評価集計2.73秒。初回監査51.16秒の記録も保持。
全体監査の検査対象9944ファイルは、終了時に再度hashを確認して変更0。
ローカルRuffはpass、CPU pytestは**2125 passed / 1 skipped（30.34秒）**、
manifest検証25コマンドは全てpass。新しい回帰テストは13件。
push先の対象commitに対するCIは別receiptで記録する。
