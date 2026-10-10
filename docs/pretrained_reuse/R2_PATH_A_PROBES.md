# R2 Path A probes — frozen-feature text, image, audio, and video runs

Date: 2026-10-11. These are bounded development probes over frozen features. The text labels are synthetic; CLEVR-4 and CLEVRER are synthetic visual benchmarks. Speech Commands measures a narrow fixed-vocabulary English keyword task. These results do not establish broad product capability or sealed-audit performance.

## Text: Typed Decisions Synth + MiniLM

| Item | Value |
|---|---|
| Dataset | `n4ze3m/typed-decisions-synth@5ece89a225b23c4cd5c4bab5735a0819d61dd7d5`, MIT; synthetic English decisions |
| Frozen encoder | `sentence-transformers/paraphrase-MiniLM-L3-v2@4ca70771034acceecb2e72475f72050fcdde4ddc`; weights SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433` |
| Source files | train SHA-256 `1f3e0f751b6fdd1115cf6af4ea3ebc13545a1148614eda4f9f56ae8492c80675`; validation `7be1d1624752938553e13e1eecef5363ca2c56dd264da2915b97ddae8d0b60a8` |
| Data size | 23,319 train decisions from 6,682 states; 2,540 validation decisions from 732 states; zero state and normalized-input overlap |
| Head | Shared candidate scorer over `[context, option, abs difference, product]`; 196,865 trainable parameters; AdamW, lr 0.001, weight decay 0.0001 |
| Budget | 8 epochs, seed 17; selected epoch 3 by minimum validation NLL |
| Local runtime | Feature extraction 31.3s; readout training 29.4s; CUDA peak allocated 384,730,112 bytes |
| Artifact | `C:\CodexArtifacts\pretrained-reuse-r2\text-probe-v0`; frozen feature cache 567,334,258 bytes, SHA-256 `6dbe639194d172d4223abeb768baadb6c1a89e0fa631907afcc49c962692d344` |

| Selected checkpoint | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|
| Train, 23,319 decisions | 0.5438 | 0.9518 | 0.5531 | 0.0259 |
| Validation, 2,540 decisions | 0.4937 | 1.0303 | 0.5946 | 0.0268 |
| Uniform-choice expected validation baseline | 0.3316 | 1.1882 | 0.6684 | 0.0000 |

Training CE fell from 1.0741 at epoch 1 to 0.7355 at epoch 8. The selected epoch was 3; by epoch 8 validation NLL rose to 1.1068 and ECE to 0.1160 while accuracy was 0.4787. This is clear overfitting after the selected point. It is an early low-capacity synthetic proxy, not an estimate of real-world English text accuracy. Open-Jev's pinned data endpoint returned HTTP 401 anonymously, so the Open-Jev task remains unmeasured.

### Text pooling ablation (same frozen encoder and split)

To check whether the baseline result depends on its token readout, three fixed pooling variants were compared on the exact same 23,319 train and 2,540 validation decisions, with seed 17 and the same 8-epoch, 196,865-parameter candidate scorer. Each variant selected its checkpoint by validation NLL. This is validation-only development evidence over synthetic labels.

| Pooling | Best epoch | Validation Accuracy | NLL | Brier | ECE (15 bins) | Train Accuracy at best |
|---|---:|---:|---:|---:|---:|---:|
| Final-layer masked mean (baseline control) | 3 | 0.4937 | 1.0303 | 0.5946 | 0.0268 | 0.5438 |
| Middle transformer layer masked mean (`hidden_states[-2]`) | 5 | 0.4972 | 1.0204 | 0.5909 | 0.0329 | 0.5837 |
| Final-layer first token | 3 | 0.4646 | 1.0658 | 0.6163 | 0.0387 | 0.5435 |

The middle-layer mean has a small numerical improvement over the baseline control (+0.0035 Accuracy, -0.0099 NLL, -0.0037 Brier) and slightly worse ECE (+0.0061). The first-token readout is worse on all four metrics. One seed and one synthetic validation set do not establish a general pooling advantage; the small difference is not a real-world text quality claim. No learned attention pooling was tested.

Reproduction: `scripts/compare_frozen_text_pooling.py` with `configs/pretrained_reuse/path_a_text_pooling.yaml`; output is outside Git at `C:\CodexArtifacts\pretrained-reuse-r2\text-pooling-probe-v0`. It contains all epoch train/validation metrics, selected readouts, per-example validation probabilities, and a serialized frozen feature cache. Run report SHA-256 `c71d4d483edc523906cb7186ccb0dbd69b2ec923f11b9efd598d3b4fa6061f75`; feature cache SHA-256 `21734e23ae51339e1c775b50a52e447824117ae242966d48aba0fd3c7ba3a06c`; config SHA-256 `8822361c43a22efc53fdd840574b30a394a5e007223aabfe804a24a09a70eb5a`; script SHA-256 `457b56d0a69c1dfd988ce70fa0349bcd77f025a1182770ed0deac6b9c6ca57dc`. Model and dataset hashes match the baseline above. No sealed audit or legacy final evaluation was loaded. The existing `text-probe-v0` artifact was not modified.

### Text state/query feature reuse probe

This separate seed-17 readout experiment split each text decision into a query-independent `State: ...` segment, a `Question: ...` segment, and candidate strings. It cached each frozen MiniLM masked-mean embedding by source revision, role, content hash, weights hash, tokenizer files and preprocessing hash. The readout receives the arithmetic mean of state and question vectors and the same four-way candidate features as the baseline scorer. Encoder weights stayed frozen; only the 196,865-parameter scorer was trained. This tests genuine state-feature reuse, while changing the readout inputs; it is **not** a cache-only parity ablation of the old scorer.

| Item | Value |
|---|---|
| Corpus and split | Same train/validation file hashes and exact 2,540 validation IDs as the baseline; 23,319 train decisions / 6,682 states and 2,540 validation decisions / 732 states; zero state or normalized-content overlap |
| Rights | Separate pinned train and validation manifests, both `ALLOW`; train usage `training`, validation usage `evaluation`, same dataset revision |
| Encoder | MiniLM revision and weights hash as above; tokenizer files SHA-256 `dd8aae41e429614ae2cce6385510055251c300d5ec3753fe4bfe89dc943bdd41`; 17,389,824 frozen parameters, 384 dimensions, max length 256 |
| Run | Seed 17; 8 epochs; AdamW 0.001 / 0.0001; selected epoch 3 by minimum validation NLL; peak CUDA allocation 691,373,056 bytes |
| Unique text cache | 94,636 entries; 145,360,896 payload bytes / 244,770,850 entry bytes; 94,636 cache entries verified on full reload |
| Reuse | 6,682 train and 732 validation state embeddings for 23,319 / 2,540 decisions: 16,637 / 1,808 state re-encodes avoided (71.35% / 71.18%) versus encoding each decision's state separately. Across state/question/candidate segment occurrences, content deduplication avoided 48,234 encodes |
| Runtime | Frozen feature extraction 70.56s; cache writes 421.98s; complete cache reload 18.84s; readout training 40.94s; post-save validation 0.22s. Cache writes of many small immutable files are the measured bottleneck; the research cache is not a deployment package |
| Artifact | `C:\CodexArtifacts\pretrained-reuse-r2\text-observation-cache-v0-retry1`; report SHA-256 `f3fc621e47008d997a07b45c5bfe7afba1a119d5379f61cf27fdf6068bab66d2`; runner SHA-256 `18904827108be068ce17384c7afe6f806ece1e3ab7c6afa5c97b18a91a7368ca`; validation predictions SHA-256 `ff8bc8a7f7306aebaf58381c0298a8bfb882922757ffa381d51437eb4eb61901`; reload verifier SHA-256 `d3fd9e4cad6dfe1ae4cf656fec4986f54b0ca76ef36968bc4414acb04f7836ab` |

Runtime was Python 3.11.9, PyTorch 2.6.0+cu124 / CUDA 12.4, Transformers 5.6.2, and NVIDIA GeForce RTX 3080 Laptop GPU (16 GiB, driver 616.92); MiniLM loaded as FP32. The successful process started at 2026-10-11 03:39:33 local time; its report was written at 03:49:00 and the independent reload verification at 03:50:16. No cloud compute was used. The run ledger records the effective arguments/config; the source commit was `049503b9212533ec2610a366fd8003faf80c76eb` and the executed runner bytes are pinned by the script hash above.

Reproduction command (from repository root; all input and output paths are explicit):

```powershell
$env:PYTHONPATH='src;.'
python -m scripts.train_frozen_text_observation_probe `
  --train C:\CodexArtifacts\pretrained-reuse-r2\text\train.jsonl `
  --validation C:\CodexArtifacts\pretrained-reuse-r2\text\validation.jsonl `
  --train-manifest manifests\dataset.example.yaml `
  --validation-manifest manifests\candidates\typed-synth-validation.yaml `
  --model C:\CodexArtifacts\pretrained-reuse-r2\models\text-minilm-en `
  --revision 4ca70771034acceecb2e72475f72050fcdde4ddc `
  --reference-run-dir C:\CodexArtifacts\pretrained-reuse-r2\text-probe-v0 `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\text-observation-cache-v0-retry1 `
  --seed 17 --epochs 8 --batch-questions 64 --feature-batch-size 128 --max-length 256

python -m scripts.verify_frozen_text_observation_probe `
  --run-dir C:\CodexArtifacts\pretrained-reuse-r2\text-observation-cache-v0-retry1 `
  --validation C:\CodexArtifacts\pretrained-reuse-r2\text\validation.jsonl `
  --validation-manifest manifests\candidates\typed-synth-validation.yaml `
  --model C:\CodexArtifacts\pretrained-reuse-r2\models\text-minilm-en `
  --output C:\CodexArtifacts\pretrained-reuse-r2\text-observation-cache-v0-retry1\readout-reload-verification.json `
  --device cuda
```

The earlier failed invocation used the validation-only manifest for both splits. It was stopped before readout training; its partial cache was not reused and remains isolated with an `aborted_before_readout_training` marker at `C:\CodexArtifacts\pretrained-reuse-r2\text-observation-cache-v0`.

| Selected readout | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|
| Original context scorer, epoch 3 | 0.4937 | 1.0303 | 0.5946 | 0.0268 |
| State/query factorized scorer, epoch 3 | 0.4752 | 1.0477 | 0.6053 | 0.0366 |

The factorized scorer is lower by 1.85 accuracy points and worse by 0.0174 NLL, 0.0107 Brier and 0.0098 ECE on these same development IDs. Its train Accuracy / NLL / Brier / ECE at the selected epoch was 0.5403 / 0.9656 / 0.5580 / 0.0301. Training CE kept falling while validation NLL and calibration worsened after epoch 3, showing the same overfitting pattern as the original probe. This one-seed synthetic result establishes that persistent text state features can be reused and exactly reloaded; it does **not** establish that arithmetic-mean state/query fusion preserves baseline decision quality.

The original direct-versus-cache validation logit delta was exactly 0 and all classes matched. A separate save/reload check loaded the saved readout and 9,996 validation cache entries, reproduced all 2,540 prediction classes/options/targets, and had max probability delta 0.0 on CUDA; verification JSON SHA-256 `b196b99a192f98c5800dc73de8ca6215f7ef9535008d00f3585f53bfb8ddfb62`. No sealed audit, legacy final evaluation, or model fine-tuning was used. A first invocation was aborted after passing the evaluation-only manifest for both splits; its partial cache remains isolated at `C:\CodexArtifacts\pretrained-reuse-r2\text-observation-cache-v0` with `run-aborted.json` (SHA-256 `a0ef726bcb3f7e629145b2365fe94b76014ea74d64b0024d230782dc9dafd1af`) and was not reused. The successful retry used the explicit training and evaluation manifests and a fresh output directory.

## Image: CLEVR-4 + V-JEPA 2.1 ViT-B

| Item | Value |
|---|---|
| Dataset | `sgvaze/clevr4@cddc78fb2a8359dc958987b2c750bfdd4bfd2c73`, CC BY 4.0; controlled synthetic image taxonomies |
| Sample | Deterministic seed 17: 1,024 train images and 512 validation images; all four image questions stay with their parent image; zero image-ID and SHA-256 overlap |
| Annotation | SHA-256 `fba922a00216bbc3641f7f03a117a9f5e6a0956edde1f66062ff399225219fd6`, matching the pinned manifest |
| Media | 544,327,790 selected image bytes. Each image SHA-256 and split ID order are in the external sample manifests. The 3,797,490,816-byte archive was range-read; its full SHA-512 was not downloaded or verified. ETag was stable across the range requests. |
| Frozen vision encoder | V-JEPA source commit `204698b45b3712590f06245fbfba32d3be539812`; checkpoint SHA-256 `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`; deterministic official 384px preprocessor; global mean of 576 image tokens |
| Candidate text encoder | MiniLM revision `4ca70771034acceecb2e72475f72050fcdde4ddc`, frozen label/taxonomy embeddings |
| Head | Shared image + candidate embedding scorer; 147,713 trainable parameters; AdamW, lr 0.001, weight decay 0.0001 |
| Budget | 8 epochs, seed 17; selected epoch 8 by minimum aggregate validation NLL |
| Local runtime | Image features: train 33.30s, validation 16.69s; readout training 6.45s; CUDA peak allocated 556,154,880 bytes |
| Artifact | `C:\CodexArtifacts\pretrained-reuse-r2\clevr4-sample-seed17-1024-512` and `C:\CodexArtifacts\pretrained-reuse-r2\image-probe-v0`; 4.87MB pooled feature cache, SHA-256 `d647605c707315839c5a81e0ac57c60df1092f051fdcb668482ae670f2a7001c` |

Validation contains 2,048 decisions, exactly 512 per taxonomy. Macro and sample-weighted Accuracy are both 0.5737.

| Taxonomy | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|
| Color | 0.6641 | 1.0429 | 0.5072 | 0.2053 |
| Texture | 0.3828 | 1.7620 | 0.7513 | 0.0559 |
| Count | 0.2988 | 1.7251 | 0.7799 | 0.0577 |
| Shape | 0.9492 | 0.2045 | 0.0902 | 0.0557 |
| Uniform 10-way expected baseline | 0.1000 | 2.3026 | 0.9000 | 0.0000 |

The selected checkpoint's train metrics were Accuracy 0.6453, NLL 1.0582, Brier 0.4823, ECE 0.1448. Train CE decreased every epoch; validation Accuracy rose from 0.1885 to 0.5737 and validation NLL fell from 2.2112 to 1.1836 over all eight epochs. The 7.15-point train/validation Accuracy gap is measurable, but the validation curve was still improving at the run boundary; this run does not establish a plateau or overfitting. Shape is solved much more easily than count and texture. Color Accuracy is moderate but poorly calibrated under the fixed ECE bins.

The upstream V-JEPA code repository is MIT-licensed, but no separate checkpoint license statement was found in the pinned release README. The weight file was obtained from the official release link; checkpoint redistribution/product-use rights remain **REVIEW**, pending explicit resolution. This local probe is technical evidence only, and the checkpoint is not approved for product packaging.

## Video: CLEVRER + V-JEPA 2.1 ViT-B

| Item | Value |
|---|---|
| Dataset | `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef`, CC0-1.0; official descriptive questions |
| Sample | Seed 17: 16 train scenes and 8 validation scenes; each scene has one temporal and one static descriptive question; previously used scenes 0–2 and 10000–10002 excluded |
| Question inputs | Train 32, validation 16; question type mix is exactly 50% temporal descriptive / 50% static descriptive |
| Split integrity | Zero scene overlap, zero media SHA-256 overlap, and zero normalized question/options overlap; each video and its two questions remain in one split |
| Video media | Train archive: 12,354,893,389 bytes; validation archive: 6,206,059,321 bytes. Range-read 20,610,458 train bytes and 10,633,031 validation bytes. ETags were pinned. Full archive hashes were not computed; each selected MP4 hash is recorded externally. |
| Frozen vision encoder | V-JEPA source commit `204698b45b3712590f06245fbfba32d3be539812`; checkpoint SHA-256 `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`; 8 uniformly sampled frames at 384px; global mean of spatiotemporal tokens |
| Candidate text encoder | MiniLM revision `4ca70771034acceecb2e72475f72050fcdde4ddc`, frozen question and answer embeddings |
| Readout | Shared candidate MLP; 295,169 trainable parameters; AdamW, lr 0.001, weight decay 0.001; 24 epochs, seed 17; selected epoch 1 by minimum validation NLL |
| Local runtime | NVIDIA GeForce RTX 3080 Laptop GPU 16 GiB; Python 3.11.9, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, timm 1.0.15, PyAV 18.1.0, Transformers 5.6.2; train features 4.99s, validation 2.39s, readout training 0.36s, validation readout 0.051ms/question; peak CUDA allocated 464,645,120 bytes |
| Cache reuse | 32 train questions required 16 video encoder calls (16 avoided); 16 validation questions required 8 calls (8 avoided). Pooled payload is 49,152 train bytes + 24,576 validation bytes; serialized video cache is 86,418 bytes, SHA-256 `7e97c562e10a5ff635dd1748349b16bc35bcd61399635f3e712f4c81bdeb623b`. |
| Reproduction | Config SHA-256 `1f171408ae4b32f0b31d70c9e8c897f909fa8447534776803dd2af8e40e2c935`; final training script SHA-256 `89943d01ba780671841d6b79ea85fd7c0078ca98ef95226207c1242ae33acaa0`; train/validation JSONL SHA-256 `25d70a899f934cd84f479ba19e52e109dbe64492446a47baf898e64ccb1824fc` / `8d227fb2c49449730e920e3efcb10aa377d64e3926cca10fddc6f37e561dae0e`. |
| Artifact | `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-video-probe-v0-retry3` and `C:\CodexArtifacts\pretrained-reuse-r2\video-probe-cache-v4`; best readout SHA-256 `77ff3dcc1067da138235fc3082fffc1746e8c5a14d644da5827c82813496df1f`; per-question validation probabilities SHA-256 `03402cbd380845d34d73d3efe05e4af630b5114bec1ee519821d3b990d5860bc`. |

| Validation subset | Count | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|
| Overall | 16 | 0.3750 | 1.1459 | 0.6475 | 0.1403 |
| Static descriptive | 8 | 0.2500 | 1.1338 | 0.6565 | 0.2766 |
| Temporal descriptive | 8 | 0.5000 | 1.1581 | 0.6385 | 0.3813 |

This is a tiny synthetic probe with two questions per scene. The sample is too small for a stable quality claim; temporal and static numbers are descriptive diagnostics only. The low accuracy and calibration do not establish a V-JEPA capability ceiling. The checkpoint's product-use rights remain **REVIEW**; no packaging or product adoption follows from this result.

### Video feature cache parity diagnostic

On one existing CLEVRER validation scene with its two original questions, the pinned 8-frame V-JEPA extractor was run once through the scene-keyed cache and once per question without reuse. Both paths produced exactly matching pooled features (max absolute delta `0.0`; allclose true). Reuse reduced encoder calls from 2 to 1. One timing sample measured 0.537 s cached and 0.639 s uncached (1.19x ratio); this is a smoke measurement, not a stable latency claim. The unique pooled feature occupied 3,072 bytes, while the question-expanded pair occupied 6,144 bytes. CUDA model allocation before inference was 347,332,608 bytes and peak allocation was 457,564,160 bytes on the RTX 3080 Laptop GPU. The diagnostic confirms basic per-scene feature parity and reuse; it does not test audio caching, persistent-cache invalidation, or broad multi-query scaling.

The reproduction script is `scripts/verify_video_cache_parity.py`. It used the existing sample `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-video-probe-v0-retry3` and wrote outputs only to `C:\CodexArtifacts\pretrained-reuse-r2\video-cache-parity-v1`. Report SHA-256 `668b74699f475973286e463ae41d18ed650316fd002c021670c51aa14b8144a2`; saved feature comparison SHA-256 `c19327cfef7591330e9eb4a4c5e4a28375738f37b0f9b0d115bd2cc457950bab`; script SHA-256 `b9e8f28148d59d22dc2a138be0b718958198578d81d41692b63a27e927f11fcc`. It ran with Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2, PyAV 18.1.0 and the already-pinned local `timm` dependency; no package was upgraded.

## Audio: Speech Commands + frozen Whisper Tiny

| Item | Value |
|---|---|
| Dataset | `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`, v0.02, CC-BY-4.0 per pinned candidate manifest; only official train and validation rows, labels yes/no/up/down/left/right/on/off/stop/go |
| Sample | Seed 17: 128 train and 64 validation clips per keyword (1,280 / 640 total); 829 train and 206 validation speaker groups; zero speaker and audio SHA overlap |
| Sample identity | Train ID-order SHA-256 `86faf73782e4270ab562d0ba402601ef4f9ac29f21b7fef8a4963712b2468923`; validation `20dca814ebf78e19ce0d81d77a2f87fec615d9496c20b7567347347ec9073b70`; fetch report SHA-256 `1ee42efcbc8e059aeb79112f65d7319f09669239dd99719313a6646bed4f1f4d` |
| Frozen encoder | `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; model safetensors SHA-256 `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`; mean pooled valid Whisper encoder tokens |
| Candidate text encoder / readout | MiniLM `paraphrase-MiniLM-L3-v2@4ca70771034acceecb2e72475f72050fcdde4ddc`; shared candidate scorer, 196,865 trainable parameters; AdamW lr 0.001, weight decay 0.0001 |
| Budget and selection | 8 epochs, seed 17; epoch 8 selected by minimum validation NLL. No sealed or test split loaded. |
| Reproduction | Config SHA-256 `6b5aea75da31c3d6cee334053b5ca6c9a8d135fc773bad28136789e94e29535c`; trainer SHA-256 `a20f0efa4c183454760c2ada27a1bee3ef1fd40e96069685391a133a4b512dec`; extractor SHA-256 `7ee939e0386e65ed303edea84602385b41cd99467c695594fa6a8732c410b3f9`. Extractor transport used `hyparquet@1.31.3` (npm tarball SHA-256 `1d448ce058c3817b4748ff43002fb644bcdca1382949a18c52f767e93eb3bd14`). |
| Local runtime | RTX 3080 Laptop GPU 16 GiB; Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.6.2; feature extraction 8.28s train / 3.94s validation; readout training 0.62s; peak CUDA allocation 208,517,120 bytes |
| Artifact | `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-sample-seed17-128-64-retry2` and `C:\CodexArtifacts\pretrained-reuse-r2\audio-probe-v0-retry3`; features SHA-256 `ace323458e7f443daf63382d2d41d99151d2a5801c671d30b0df68d52338ae64`; readout `969372c12dc742ede5e5b412b782e127f5cca50e57a07415c14a354f8693b23a`; validation predictions `0e2b9d3ffd18c67ed52c9c4cafec3a059ef28bd862f6fe8a4fb4fa471f9d31ae`; run report `864e81f838af6857b514c1a8e331361dab12a468e2c03f3f97ddfa4d9b4beef` |

| Selected checkpoint | Count | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|
| Train | 1,280 | 0.9797 | 0.1014 | 0.0372 | 0.0429 |
| Validation | 640 | 0.9531 | 0.1624 | 0.0661 | 0.0377 |

Validation accuracy ranged from 0.8906 (`down`) to 1.0000 (`yes`); each keyword has 64 examples. Training CE fell from 2.0884 to 0.1077 over 8 epochs, while validation NLL fell at every epoch from 1.7514 to 0.1624. Option-order permutations had zero aligned-logit delta. This is a comparatively easy, closed-set English keyword task; its high score does not establish open-vocabulary audio reasoning, speech comprehension, Japanese audio capability, or joint audio-video understanding. Validation was used for checkpoint selection and is not a blind result.

Reproduction command (from repository root, with the pinned local checkpoints present):

```powershell
$env:PYTHONPATH='src;.'
py -3.11 -m scripts.train_frozen_audio_probe `
  --sample-dir C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-sample-seed17-128-64-retry2 `
  --audio-model C:\CodexArtifacts\pretrained-reuse-r2\models\audio-whisper-tiny `
  --audio-revision 169d4a4341b33bc18d8881c4b69c2e104e1cc0af `
  --text-model C:\CodexArtifacts\pretrained-reuse-r2\models\text-minilm-en `
  --text-revision 4ca70771034acceecb2e72475f72050fcdde4ddc `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\audio-probe-v0-retry3
```

### Audio feature-cache parity diagnostic

On the first fixed validation clip, the saved Whisper feature and a fresh extraction with the same pinned encoder matched exactly (max absolute feature delta `0.0`). The downstream fixed 10-option readout logits and probabilities also matched exactly (max logit delta `0.0`). The cached feature is 1,536 bytes; re-encoding this clip took 0.289 s on the RTX 3080 Laptop GPU, peak allocated memory was 151,452,672 bytes, and the feature cache avoided one encoder execution for that one existing decision. These are parity and micro-timing diagnostics only: this does not create multiple independent questions for one audio event or measure repeated-query latency at scale.

Reproduction script: `scripts/verify_audio_cache_parity.py`. Inputs are the existing `C:\CodexArtifacts\pretrained-reuse-r2\speech-commands-sample-seed17-128-64-retry2`, Whisper and MiniLM checkpoints and readout from `audio-probe-v0-retry3`; output is external at `C:\CodexArtifacts\pretrained-reuse-r2\audio-cache-parity-v0`. Report SHA-256 `6d02a98c91dfd0a393b27d38dbca9afaf2b20ca750218c01d21c94392e1accdf`; source feature cache SHA-256 `ace323458e7f443daf63382d2d41d99151d2a5801c671d30b0df68d52338ae64`; script SHA-256 `270073d31fa1ed8419e8368deb1b3ec3ab266d0e7e25ce83af3b511d983b2bd0`. It used Python 3.11.9, PyTorch 2.6.0+cu124 and Transformers 5.6.2; no model was updated, and no sealed or test split was loaded.

## Japanese text-only diagnostic: Ruri v3-30m + JamC-QA-V2 dev

This one-time zero-shot diagnostic checks whether a small Japanese embedding
encoder can already rank multiple-choice answers without training a Decision
head. It is not a trained Path A result and does not establish broad Japanese
Decision quality.

| Item | Value |
|---|---|
| Dataset | [`sbintuitions/JamC-QA-V2`](https://huggingface.co/datasets/sbintuitions/JamC-QA-V2) at revision `cfb4b64d289acaf592237179f90dd23d3f2d5bdf`; CC-BY-SA-4.0; dev only |
| License status | **REVIEW** under the repository manifest policy; used only for this isolated evaluation diagnostic, not training or product distribution |
| Rows | 52; category counts: 4 each for culture, custom, regional identity, geography, history, government, law and healthcare; 20 JSDF. Duplicate answer strings were retained. |
| Data hashes | Source Parquet SHA-256 `acd98fd9a1cf3fd5c59bb321b4480141fae6d124b70779599280171741ed3198`; normalized dev JSONL SHA-256 `7c9eece543968f49816a17020d7ea286b43a7c689b383be3b795c4557a159380`; ordered question IDs SHA-256 `ca69c3443026fd44025b4c50d2e0dab5dfd5843f146beaf2fdc63d987af13e29` |
| Encoder | `cl-nagoya/ruri-v3-30m@24899e5de370b56d179604a007c0d727bf144504`; local weights SHA-256 `e94155e342cbfc33280c345c6e2912905fb5de6e1a3adeb938f395aafd2f8832`; 36,705,536 frozen parameters; masked-mean pooling and L2 normalization |
| Scoring | Query prefix `検索クエリ: `, candidate prefix `検索文書: `; cosine similarity, fixed temperature 1.0; no fitting, training or checkpoint selection |
| Split protection | Test split not loaded; sealed audit not loaded; zero training examples. Ruri training-data contamination on JamC-QA was not independently excluded. |
| Runtime | RTX 3080 Laptop GPU; CUDA; 0.330 s encoder time; peak allocated CUDA memory 213,029,888 bytes |
| Reproduction | Config `configs/pretrained_reuse/ruri_jamcqa_zero_shot.yaml`, SHA-256 `b2346f2e4f08d9c6c2629a7f68e05e7a6fe5c43b629a14e3a654e25108623b9f`; evaluator `scripts/evaluate_ruri_jamcqa_dev.py`, SHA-256 `21298059e7df0b6574cb4d1ac051d29d8d2242e221f9c778a72821919343f4dc`; external output `C:\CodexArtifacts\pretrained-reuse-r2\ruri-jamcqa-v2-dev-zero-shot-v1` |
| Saved predictions | SHA-256 `2de1b928fc1a177aef537c2359a0a5b7277df5495f3412fd6930cdcebdda5d74`; report SHA-256 `0f17d5cb26a53943930e5dcbc15fde81d4ede27a019f666d480c24979aa57dc4` |

| Split | Count | Accuracy | NLL | Brier | ECE (15 bins) | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|
| JamC-QA-V2 dev | 52 | 0.2308 | 1.3864 | 0.7500 | 0.0222 | 0.2530 |
| Uniform four-choice reference | — | 0.2500 expected | 1.3863 | 0.7500 | 0.0000 expected | 0.2500 |

The result is near the uniform four-choice reference and does not support using
this frozen embedding/cosine setup as a Japanese Decision path. ECE is low
because confidence is near 0.25, not because the ranking is useful. The dev set
is small and intended for development; the single run is a diagnostic, not a
blind estimate. No follow-up tuning or second evaluation was performed. The
result motivates a trained small readout or a different Japanese text encoder
as separately preregistered validation work; it does not justify selecting one
from this dev result. The fixed development evaluation was intentionally not
repeated after inspecting these metrics.

## Audio multi-query readout: four questions per Speech Commands event

To exercise observation reuse with a question-conditioned Decision interface,
the frozen Speech Commands audio features were reused to answer four
deterministic questions per clip: identify the keyword, detect a direction
word, detect a yes/no response, and detect an on/off control word. The last
three labels are mechanically derived from the original keyword, so this is a
closed-set compositional probe—not open-ended speech understanding or an
independent semantic benchmark. Every query variant for a clip stayed in its
original split.

| Item | Value |
|---|---|
| Source | `google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da`, 1,280 train / 640 validation clips; 829 / 206 speakers; zero speaker, clip ID and audio hash overlap |
| Frozen observation features | Existing Whisper Tiny validation/train feature cache SHA-256 `ace323458e7f443daf63382d2d41d99151d2a5801c671d30b0df68d52338ae64`; Whisper revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`; no audio encoder execution in this experiment |
| Question/candidate encoder | MiniLM-L3 English, revision `4ca70771034acceecb2e72475f72050fcdde4ddc`, model weights SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`; frozen masked-mean text embeddings |
| Readout | Shared candidate MLP; question, audio and candidate embeddings are input features; 344,321 trainable parameters; AdamW lr 0.001, weight decay 0.0001; seed 17; 8 epochs, 1,280 updates; minimum macro question-type validation NLL selected epoch 4 |
| Query counts | 5,120 train and 2,560 validation query examples from 1,280 / 640 unique audio events; each question type has 1,280 / 640 queries |
| Local runtime | RTX 3080 Laptop GPU, CUDA 12.4, PyTorch 2.6.0+cu124, Transformers 5.6.2; readout training 6.33s; evaluation 0.284s / 0.111ms per query; CUDA peak 27,896,832 bytes is readout-only and does not measure a simultaneous full-encoder resident runtime |
| Checkpoint and cache reuse | Best readout SHA-256 `4efa7939a686ad97b38f9eec0b8171e6e62d125ba4eb373e5d83e33ac4df42c7`; saved-checkpoint reload reproduced all 2,560 validation predictions with zero logit/probability/metric delta. One cached feature per each of 640 validation events serves 4 questions, avoiding 1,920 hypothetical per-query encoder executions; no encoder was actually run during this experiment. |
| Reproduction | Config `configs/pretrained_reuse/path_a_audio_multiquery.yaml`, SHA-256 `d5b130b8ba859a21091cd2fd6ff327301d408ae1cd0a4ae83e9a282708843396`; training script at source commit `685f9a413d78b16916d8662bfd9fb788ae01a0f0`, SHA-256 `1f52f0d4594771919043771a55bc9d0e9215bf16e4131f855ec8af263c2a9ffd`; output `C:\CodexArtifacts\pretrained-reuse-r2\audio-multiquery-v0` |
| Saved evidence | Run report SHA-256 `833258cbc584cbe5f2583c7cc9a5a6f115deb9a46b232e81b3eb7c7489fc081b`; predictions SHA-256 `e203cb22ba5e18d75b0a4a78afcd475993f556edefedc7cf12aeb5210a5b3932`; history SHA-256 `960a79406d3a292d8787ba63510cf8a8acd59ec81f77fa8e2f9d0af26323a274`; best reload verification SHA-256 `4e458403d31e7faec6a706512fe2d57304a5fa5a199415441331e42678bdd5f8` |

| Question type | Validation count | Accuracy | NLL | Brier | ECE (15 bins) | Positive rate |
|---|---:|---:|---:|---:|---:|---:|
| Keyword identity (10 choices) | 640 | 0.9594 | 0.1493 | 0.0610 | 0.0248 | — |
| Direction presence (2 choices) | 640 | 0.9797 | 0.0596 | 0.0319 | 0.0114 | 0.400 |
| Yes/no response presence (2 choices) | 640 | 0.9812 | 0.0422 | 0.0225 | 0.0093 | 0.200 |
| On/off control presence (2 choices) | 640 | 0.9937 | 0.0262 | 0.0139 | 0.0119 | 0.200 |
| Macro across question types | 2,560 | 0.9785 | 0.0693 | 0.0323 | 0.0143 | — |

Question-type Accuracy is not directly comparable because the three derived
binary tasks have different class balance and are much simpler than open-ended
speech questions. For reference, the earlier keyword-only checkpoint reached
0.9531 Accuracy / 0.1624 NLL on these same validation IDs, while the new
keyword-identity row reached 0.9594 / 0.1493. Candidate prompt text, scorer
architecture and training setup also changed, so this small difference is not
attributed to question conditioning. The validation split was already used for
checkpoint selection in the earlier audio probe and remains development data.

The original report's `model.total_resident_parameters` field was the sum of
Whisper (37,760,640), MiniLM (17,389,824) and the readout (344,321), not a
measurement of simultaneous residency in this cache-focused run. Its corrected
interpretation is a 55,494,785 full-component parameter sum; the 27.9 MB CUDA
peak applies only to cached-feature readout training/inference. Encoder model
files alone total 220,631,160 bytes and the readout is another 1,379,344 bytes,
before tokenizer/config files, so this component set exceeds the initial 200 MB
package target. The original report remains unchanged; the external
`interpretation-correction.json` records this scope correction (SHA-256
`48f06eec102529aba9c2e7f929384490cd5de04379cb1e4b7d5221ca68023c9e`). The
reload verifier is `scripts/verify_audio_multiquery_reload.py` (SHA-256
`4178f760481db40d1a825faa3faa931b2ed5ba8b3736d52deebcd70706a3e7cd`).

### Shared Path A candidate-scoring core

The frozen-feature Path A probes now use
`tiny_omni_decision.decision.FrozenFeatureCandidateScorer`. Modality encoders
and feature construction remain separate; the shared MLP scores each supplied
candidate feature and returns option logits, a temperature-scaled probability
distribution, the selected index, and confidence. It supports padded/batched
candidate tensors through `forward` and variable candidate counts per query
through `decide`. This is a shared readout implementation, not a jointly
trained multimodal encoder or fusion model.

The saved audio multi-query checkpoint was loaded through this package class
and re-evaluated on the same 2,560 validation questions. Logit, probability,
and metric deltas were all exactly zero. The separate verification artifact is
`C:\CodexArtifacts\pretrained-reuse-r2\audio-multiquery-v0\shared-decision-api-verification.json`
(SHA-256 `37bfea1155f742fd8fbde62e065609d45277c10cbae2f2157e08cb013a8f6f3b`);
it records the package module SHA-256
`43d26817150777a895f67a8992702a1cb722bbf07afc635027fbfaf01691a4ec` and
verifier SHA-256
`0766a703c1663544daf3a8b5319bbc65cad50d96fee9a43ab0524fa123af532d`. The
older reload sidecar and training report were not overwritten.

### CLEVRER visual multi-query/cache diagnostic

The frozen video readout was evaluated over every supported descriptive
question from the same eight CLEVRER development-validation scenes already
used by the original probe: 88 questions total (11 per scene), versus 16
questions in the original sampled validation file. The 16 original questions
are included in the 88 and reproduce after checkpoint reload with zero class
prediction mismatches and a maximum probability delta of `5.96e-8`. This is a
reuse/coverage diagnostic, **not independent validation**: all 88 decisions
come from only eight scenes, and the original 16 questions from those scenes
participated in readout checkpoint selection. No sealed audit was opened.

| Item | Value |
|---|---|
| Raw question source | `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef`; pinned validation JSON SHA-256 `fdf841678a476655b906165e6e4b0ed1516785c5ab927c8d0a4614e4e27e10d` |
| Frozen sample | 8 validation scenes; train manifest SHA-256 `25d70a899f934cd84f479ba19e52e109dbe64492446a47baf898e64ccb1824fc`; validation manifest SHA-256 `8d227fb2c49449730e920e3efcb10aa377d64e3926cca10fddc6f37e561dae0e`; fetch report SHA-256 `ea3918dc66ec8b355405faf36722a0774db6e921d885cf69fdb135578c27134f` |
| Feature cache | Existing 8 pooled V-JEPA features, 24,576 bytes; cache file SHA-256 `7e97c562e10a5ff635dd1748349b16bc35bcd61399635f3e712f4c81bdeb623b`; frame count 8; V-JEPA source commit `204698b45b3712590f06245fbfba32d3be539812`; checkpoint SHA-256 `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d` |
| Readout and text model | Existing readout SHA-256 `77ff3dcc1067da138235fc3082fffc1746e8c5a14d644da5827c82813496df1f`, selected at epoch 1 by validation NLL; MiniLM `paraphrase-MiniLM-L3-v2@4ca70771034acceecb2e72475f72050fcdde4ddc`, weights SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433` |
| Task mix | 8 static descriptive; 80 temporal descriptive. Taxonomies: count 32, exist 16, color 12, material 14, shape 14. |
| Runtime | RTX 3080 Laptop GPU; Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, Transformers 5.6.2. This query run loaded cached video features and did **zero** V-JEPA executions. MiniLM load 0.858s, text feature extraction 0.232s, readout load 0.005s, readout inference 0.006s; peak allocated CUDA memory 104,437,760 bytes. The source cache run separately encoded its 8 validation videos in 2.392s and reported a process peak of 464,645,120 bytes; this is not a simultaneous V-JEPA + MiniLM residency measure. Neither figure is full-pipeline latency/memory. |
| Cache accounting | 8 unique video features served 88 queries (11 per scene on average); 80 encoder calls would be avoided versus the hypothetical one-call-per-question pattern. The uncached 88-pass runtime was not measured; this is an execution-count comparison, not a measured speedup. |
| Reproduction | `scripts/diagnose_clevrer_video_multiquery.py`, SHA-256 `61f8aedd38ef371cb0499a03ffc4f7372f5b153b5fc357003aa1a87d2960be38`; source commit `e2461643adf47ae26096f4ccb04bc312d4e30839`; output `C:\CodexArtifacts\pretrained-reuse-r2\video-probe-multiquery-v3` |
| Saved evidence | Predictions SHA-256 `a35573e528949ca02e2b8fc29398358c097d3919c584b63c10e7e98b7e834cdd`; report SHA-256 `9fe6c2ee4db92c3027cacdb2ac14b84cfa8c1bca89fb061782a430349ba1da04` |

Exact invocation (PowerShell; all inputs are read-only and the output directory
must not already contain files):

```powershell
$env:PYTHONPATH = "src;."
python scripts/diagnose_clevrer_video_multiquery.py `
  --sample-dir C:\CodexArtifacts\pretrained-reuse-r2\clevrer-video-probe-v0-retry3 `
  --raw-validation-questions C:\CodexArtifacts\pretrained-reuse-r2\clevrer-sample-seed17\validation-questions.json `
  --cache-dir C:\CodexArtifacts\pretrained-reuse-r2\video-probe-cache-v4 `
  --readout-dir C:\CodexArtifacts\pretrained-reuse-r2\video-probe-v0 `
  --text-model C:\CodexArtifacts\pretrained-reuse-r2\models\text-minilm-en `
  --text-revision 4ca70771034acceecb2e72475f72050fcdde4ddc `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\video-probe-multiquery-v3
```

| Scope | Count | Accuracy | NLL | Brier | ECE (15 bins) |
|---|---:|---:|---:|---:|---:|
| All expanded validation questions | 88 | 0.3523 | 1.3390 | 0.6978 | 0.0441 |
| Static descriptive | 8 | 0.2500 | 1.1338 | 0.6565 | 0.2766 |
| Temporal descriptive | 80 | 0.3625 | 1.3596 | 0.7020 | 0.0534 |
| Count | 32 | 0.1563 | 1.7931 | 0.8338 | 0.0173 |
| Exist | 16 | 0.6875 | 0.6829 | 0.4898 | 0.1742 |
| Color | 12 | 0.2500 | 2.0092 | 0.8551 | 0.0633 |
| Material | 14 | 0.5000 | 0.7101 | 0.5168 | 0.0547 |
| Shape | 14 | 0.3571 | 1.1054 | 0.6710 | 0.0086 |

The taxonomy breakdown suggests that this probe's decision quality varies
substantially with the question/answer task; aggregate accuracy obscures very
weak count/color results. It does not identify whether the cause is the frozen
video representation, text-question embedding, or readout because those
components were not ablated here. The original v0 report lacked artifact hashes
and source commit metadata; the matching cache-v4 report records the identical
readout and prediction hashes and supplies provenance (source commit
`2124ea984e47cdeb431f436f681c9e454c5c4a8e`). Existing reports/checkpoints and
media were left untouched. The evaluator is a versioned repository script;
predictions and reports are external artifacts and were not added to Git.

#### Typed persistent observation-cache follow-up

The multi-query evaluator was then rerun through the new
`tiny_omni_decision.cache.ObservationFeatureCache`, using the same frozen
features and the same 88 validation questions. The cache key is observation
level and includes modality, source and source revision, media SHA-256, encoder
revision and weight SHA-256, preprocessing/code SHA-256, temporal frame policy,
feature dtype/shape, and optional time bounds. It contains no question or
candidate text, so all questions for one scene resolve to one entry. Entries
are immutable; exact repeats return the existing payload, identity changes
produce a different key, and read validates metadata, byte count and payload
SHA-256.

| Item | Value |
|---|---|
| Implementation | `src/tiny_omni_decision/cache.py`; CPU-only tests in `tests/test_feature_cache.py` cover exact roundtrip/idempotence, asset/source/model/preprocessor/frame-policy invalidation, malformed identities, and payload corruption. |
| Real cached records | 8 unique scene observations; 24,576 feature-payload bytes and 32,072 bytes including metadata. All eight persisted entries reloaded byte-for-byte and were then used to score the 88 questions. Each record was referenced by 11 questions. |
| Evaluation | Original 16 questions retained zero class mismatches and maximum probability delta `5.96e-8`; 88-question metrics are unchanged from the preceding diagnostic. No V-JEPA encoder executions occurred in this run. |
| Reproduction | Source commit `b2d285d0de93e22a3f2807761e65f80053cf28b3`; script SHA-256 `f8782e0ae33c819e41c9bec2651c1660104daddffc02d761db61b74dfa934cda`; output `C:\CodexArtifacts\pretrained-reuse-r2\video-probe-multiquery-v4` |
| Saved evidence | Cache manifest SHA-256 `289b3b435fc318c40f9e03ada0902645debd76f61cb1d6ac36c70e8fc35ed9d4`; prediction SHA-256 `0ccb8d1dc40502927d1a76717ebf09fa7e2075456ddac9506b8e9baf9647bf34`; report SHA-256 `2c676ab412fc41f4b13d3126e2fa396e06af1b005846692c461014bdfb85388b` |

This establishes content verification and key-based invalidation for the
video observation cache used by this diagnostic. The video-only run did not
add cache eviction/garbage collection, prove cross-process crash recovery
under power loss, or measure a live V-JEPA + text tower resident at once.
Audio and image follow-up checks below exercise the cache on their existing
frozen features; text integration and a shared end-to-end pipeline remain open.

#### Typed persistent audio-observation cache follow-up

The same immutable observation cache was exercised on the existing frozen
Speech Commands features. The run and its train/validation manifests,
checkpoint, saved predictions, and media were read-only. Every referenced WAV
was checked against its recorded SHA-256 before cache materialization. Cache
identity includes pinned dataset revision, audio asset hash, Whisper revision
and weight hash, feature-extractor config hash, and the extraction/pooling code
hash; question and candidate strings are absent from the key.

| Item | Value |
|---|---|
| Persisted records | 1,280 train and 640 validation observations; 1,920 cache entries total. Train/validation media overlap = 0 and speaker overlap = 0. Unique speakers: 829 train, 206 validation. |
| Payload and reuse | 384 float32 values per clip; 2,949,120 payload bytes and 5,172,480 bytes including entry metadata. Each of 640 validation observations is reused by four question types (2,560 queries). All restored feature tensors exactly equal the frozen source tensors. |
| Same-checkpoint re-evaluation | CPU replay reproduced all saved class predictions exactly. Relative to the original CUDA evaluation, max absolute logit delta was `1.1444e-5`, probability delta `2.2054e-6`, and metric delta `1.8627e-8`; these backend deltas are reported rather than described as bitwise output parity. Macro Accuracy/NLL/Brier/ECE remain `0.9785 / 0.069327 / 0.032318 / 0.014346`. |
| Runtime | Cache write plus reload: 10.025 s; CPU validation inference: 0.504 s. No audio encoder was loaded or executed, and the GPU was not used. |
| Reproduction | Source commit `e28ffdee727a046136ad5934baf46bec91394787`; script SHA-256 `6a224d1be61d0fca779766b9cda99f4b0508338d4b6758244745a0b43fe806a3`; output `C:\CodexArtifacts\pretrained-reuse-r2\audio-observation-cache-multiquery-v4`. |
| Saved evidence | Verification report SHA-256 `459871c66419122fcae96e3ed542d4e87dc7d0a30b73bfc850bcef998ec79f5a`; cache manifest SHA-256 `fe0d8b5717651d7e845e577902c6230f2f96516b654870310433bb2c648d3f39`; original multi-query run report SHA-256 `833258cbc584cbe5f2583c7cc9a5a6f115deb9a46b232e81b3eb7c7489fc081b`. Ordered train/validation ID hashes: `86faf73782e4270ab562d0ba402601ef4f9ac29f21b7fef8a4963712b2468923` / `20dca814ebf78e19ce0d81d77a2f87fec615d9496c20b7567347347ec9073b70`. |

The initial verification attempt used an overly strict cross-backend logit
threshold and stopped despite identical class predictions; that partial output
was retained under `audio-observation-cache-multiquery-v0`. The successful
follow-up reports exact feature equality and exact class-prediction equality,
while preserving measured CPU-vs-CUDA numeric deltas above. This is cache
correctness evidence on the same development validation set, not an independent
quality estimate. No sealed audit or test split was loaded.

#### Typed persistent image-observation cache follow-up

The same immutable observation cache was exercised on the existing frozen
CLEVR-4 image features, without re-running V-JEPA. Every selected PNG was
checked against the per-image SHA-256 manifest. The key binds the pinned
CLEVR-4 revision, image content hash, V-JEPA source/weight hashes, official
384-pixel preprocessor code, and the pooled spatial-token feature contract.
The original image probe's report omitted its source commit and script hash;
both remain **UNKNOWN** rather than being inferred from current files. The
source archive full SHA-512 is also still unverified.

| Item | Value |
|---|---|
| Persisted records | 1,024 train and 512 validation images; 1,536 cache entries. Image-ID and media-hash overlap = 0. All feature tensors reloaded exactly. |
| Payload and scoring | 768 float32 values per image; 4,718,592 feature-payload bytes and 6,099,456 bytes including entry metadata. The fixed 147,713-parameter readout and pinned MiniLM were replayed on 2,048 validation decisions. All prediction classes matched the saved CUDA run; maximum probability delta `1.3113e-6` and metric delta `1.1921e-7`. |
| Validation metrics | Macro Accuracy/NLL/Brier/ECE `0.5737 / 1.18360 / 0.53216 / 0.09024`. By taxonomy: color `0.6641 / 1.04290 / 0.50716 / 0.20532`; texture `0.3828 / 1.76196 / 0.75130 / 0.05588`; count `0.2988 / 1.72505 / 0.77995 / 0.05771`; shape `0.9492 / 0.20449 / 0.09025 / 0.05566`. This remains a synthetic CLEVR-4 task, not natural-image understanding. |
| Runtime | Cache write plus reload `8.362 s`; CPU validation inference `0.033 s`. No V-JEPA encoder execution or GPU use. The 86,833,152-parameter V-JEPA checkpoint was not rehashed or loaded during this cache check; its pinned hash is inherited from the frozen feature-cache metadata and original probe report. |
| Reproduction | Source commit `8a5e449cd8a6f490526ef1d448bbd795609af46f`; script SHA-256 `2ba4fb0aae83a9f61cc55da0fa3d3b9f0ed98b14ca7395f0f69bf5c87c68e77e`; output `C:\CodexArtifacts\pretrained-reuse-r2\image-observation-cache-v1`. The pinned upstream checkout had unrelated modified eval/train YAML files; those files were left untouched, and the preprocessor/transform source files were hashed into the cache identity. |
| Saved evidence | Verification report SHA-256 `6b27c717db997bc035afc9dfb128789c82405c9c9f3a20e7b023f3646de702ee`; cache manifest SHA-256 `1e4c1efcf352705aa05a8b0e431688bdcf8195c7e1614eccd8a218e122e56851`; original image feature cache SHA-256 `d647605c707315839c5a81e0ac57c60df1092f051fdcb668482ae670f2a7001c`; readout SHA-256 `e99708aba55159a7e1e50c02ebe7492d343082d8c3e0ea810406b594bd23d174`. |

The original validation is reused, not independent, and V-JEPA checkpoint rights
remain **REVIEW**. The successful replay validates persistent feature integrity
and readout reproducibility only; it does not raise the image-quality estimate.

#### Direct video-cache versus fresh-encoder parity (single scene)

To close the narrow question of whether video cache reuse reproduces a fresh
V-JEPA feature computation, I reloaded the pinned frozen encoder and processed
the same two existing validation questions both through `_video_features`
(which deduplicates their shared scene) and independently per question. The
sample, manifest, and media were read-only; the sealed audit was not loaded.

| Item | Result |
|---|---|
| Sample | CLEVRER validation scene `10498`; question IDs `clevrer-validation-10498-q000` and `clevrer-validation-10498-q008`; one underlying video SHA-256 `f5dc257fc4091cb2b2e6519a6ac1496c6399fdb16b2e60d897c37d9ddbd3c94f`. |
| Encoder | V-JEPA source commit `204698b45b3712590f06245fbfba32d3be539812`; checkpoint SHA-256 `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`; 8 frames; output `[2, 768]` float32. |
| Parity and reuse | Cached and fresh-encoder features are `allclose`; maximum absolute difference `0.0`. Deduplicated path made 1 encoder call for 2 questions (1 avoided). Feature payload was 3,072 bytes instead of 6,144 bytes for repeated rows. |
| Runtime and memory | RTX 3080 Laptop 16 GiB; CUDA 12.4, PyTorch 2.6.0+cu124, Transformers 5.6.2, PyAV 18.1.0. Cache path `0.5350 s`, uncached path `0.6527 s`, observed ratio `1.22x`; model allocation before inference `347,332,608` bytes and peak allocation `457,564,160` bytes. One scene is too small for a product latency claim. |
| Reproduction | `python -m scripts.verify_video_cache_parity --sample-dir C:\CodexArtifacts\pretrained-reuse-r2\clevrer-video-probe-v0-retry3 --vjepa-source C:\CodexArtifacts\pretrained-reuse-r1\vjepa2-source --vjepa-checkpoint C:\CodexArtifacts\pretrained-reuse-r1\vjepa2_1_vitb_dist_vitG_384.pt --vjepa-sha256 848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d --output-dir C:\CodexArtifacts\pretrained-reuse-r2\video-feature-cache-parity-v1`. Source commit `279e8d8692e41217d79df26abf1be46e6597fca6`; script SHA-256 `b9e8f28148d59d22dc2a138be0b718958198578d81d41692b63a27e927f11fcc`. |
| External artifacts | `C:\CodexArtifacts\pretrained-reuse-r2\video-feature-cache-parity-v1\report.json` SHA-256 `20f3573d325651e3671b38dc20deebe88b7708a659f317a8cd1239686a55ef41`; features SHA-256 `c19327cfef7591330e9eb4a4c5e4a28375738f37b0f9b0d115bd2cc457950bab`. |

This establishes exact feature parity for this one deterministic scene and
implementation path. It does not establish repeated-run determinism across
backends, prediction parity for a trained scorer, multi-scene latency, or broad
video quality. V-JEPA rights remain **REVIEW**.

## R2 status

#### Packed feature-cache experiment

The text observation cache contained 94,636 immutable entries across 94,636
files (244,770,850 bytes). Its earlier materialization recorded 421.98 seconds
for entry writes/fsync and 18.84 seconds for a complete reload. I tested two
read-only repacking formats against a new output directory; the source cache
was not modified.

| Format | Build time | Stored bytes | Result |
|---|---:|---:|---|
| SQLite rows with each full entry as a BLOB | 11.59 s | 443,899,904 | Rejected: 81.3% larger than source. |
| Sequential `features.pack` plus SQLite offset/hash index | 7.76 s in the standalone comparison; 11.49 s through the checked API | 260,311,357 | Retained as an optional immutable export format: 6.35% larger than source, one payload file and one index. |

The checked API snapshot contains all 94,636 entries. Full package SHA-256 and
SQLite integrity verification took 0.254 s; 1,000 seeded random entries were
loaded through the API and their decoded payloads matched exactly (0.218 s,
including key reconstruction and entry hashing). A lower-level matched raw-byte
read comparison measured 1,000 random reads at 0.077 s from individual files
and 0.018 s from the pack plus an already-open index/pack handle; this does not
include decoding the feature payload or running an encoder. These local warm-cache
measurements are a storage diagnostic, not a deployment latency claim.

The packer refuses to overwrite an existing destination, stages the new package
in a sibling directory, stores exact source entry bytes, hashes each entry and
the finished pack/index, and publishes a manifest after the files are complete.
The reader checks per-entry hashes and embedded payload hashes; `verify()` also
checks whole-file hashes and the SQLite index. CPU tests cover exact roundtrip,
overwrite refusal, and post-pack corruption detection. The storage remains
read-only and is not wired into a model-serving pipeline; cache eviction,
crash-recovery under power loss, and model-plus-readout residency remain open.

Reproduction:

```powershell
py -3.11 scripts/pack_observation_feature_cache.py `
  --source-cache C:\CodexArtifacts\pretrained-reuse-r2\text-observation-cache-v0-retry1\feature-cache `
  --output C:\CodexArtifacts\pretrained-reuse-r2\text-observation-cache-v0-packed-api
```

Source cache and packed output are outside Git. Packed manifest SHA-256 is
`1e6341df5a63708b343baefd10618b426e4dc59f295cf466db2bf0c662633781`; pack
SHA-256 is `f015c139dfc26d995fc6bf5dd4b891a9c526f58b4b751bc5f1e98b6f228867db`;
index SHA-256 is `91cd8dcd21ddacd35f1601b8e70a2076c70154441be336440c43648dfd97fecd`;
verification report SHA-256 is
`ecd8b905fd60d06c99150e48be1caa68bb98d20bb3e341a1f664cd0e2075b01f`.
The source cache report SHA-256 is `f3fc621e47008d997a07b45c5bfe7afba1a119d5379f61cf27fdf6068bab66d2`.
The experiment started from commit `06cd6c560e3bb8c6cc9ae4a7e03bbadd9f2fd8b8`
with the new implementation uncommitted; `packed_cache.py` SHA-256 was
`c1fbb3fd2ddd663cd6c50ff82b42ff3b801d8c53e07ffea4b00d340b8fec9ca2`, and the
external API validation driver SHA-256 was
`99d9875914f1548c9522b932b1a6e37efedd6eed691658f2433b7456ac715df9`. Runtime
was Python 3.11.9 on the local Windows machine; no GPU, model, media, or training
process was used. This was storage-only, so no validation predictions or model
quality metrics were regenerated. Implementation: `src/tiny_omni_decision/packed_cache.py`;
command wrapper: `scripts/pack_observation_feature_cache.py`.

- Implemented and exercised frozen-feature candidate readouts for text, image, and a small video task. Per-example validation probabilities, epoch metrics, source/media hashes, and selected heads are saved outside Git.
- Audio-only closed-set keyword evaluation is complete using frozen Whisper Tiny features and a small candidate readout; it does not establish broad speech understanding. Train/validation speaker and media gates passed, and artifacts are outside Git.
- A one-time Ruri Japanese zero-shot check on 52 JamC-QA-V2 dev items was near the uniform four-choice baseline. It is evaluation-only, has unresolved contamination risk, and the dataset license is `REVIEW`; it does not qualify as a Japanese Decision model or close the text-quality gate.
- Video cache reuse now covers 11 descriptive questions per each of 8 existing CLEVRER validation scenes; the original 16-query subset reloads with numerical parity. A separate direct V-JEPA check on one validation scene recomputed both questions independently and matched the deduplicated cache features exactly (max difference 0.0); this is a one-scene correctness check, not a latency claim. The 88-query metrics are not independent of the validation used for checkpoint selection. A typed immutable cache roundtrip was exercised on all 8 video features; separate CPU tests cover key invalidation and payload corruption. Audio cache roundtrip covers 1,920 frozen train/validation observations, with exact feature reload and the same class predictions for all 2,560 validation queries. Image cache roundtrip covers 1,536 frozen train/validation images, with exact feature reload and the same classes for all 2,048 validation decisions. Independent visual scene coverage, cache eviction, synchronized audio-video tasks, and full encoder-plus-readout deployment memory remain untested.
- No sealed audit, legacy final evaluation, or training checkpoint was loaded. No backbone was updated. No model weights or media are added to Git.
- This is not a selected release candidate. Open-Jev access and V-JEPA checkpoint rights remain unresolved; broader real text, audio, natural-image, joint-modality, calibration and deployment gates remain open.

#### Expanded CLEVRER frozen-video readout (2026-10-11)

This bounded data-coverage follow-up keeps the prior 8-frame V-JEPA + frozen
MiniLM encoders and the 295,169-parameter candidate scorer fixed, while
increasing unique CLEVRER scene coverage. It is a descriptive synthetic-video
probe, not general video understanding. The V-JEPA checkpoint's commercial
rights remain **REVIEW**; no product use is implied.

| Item | Result |
|---|---|
| Source/split | `MIT-IBM/CLEVRER@98b842082ba4f7c18b6b9e3f39145871782a65ef`, CC0-1.0. Seed 17; 256 train scenes / 512 questions and 200 validation scenes / 400 questions; 50% temporal-descriptive and 50% static-descriptive in each split. Existing scene IDs were excluded. Scene, media SHA-256, and normalized question overlap are all zero. The ordered validation ID hash is `eac798f4f3234f3aa9003bedd3721cb25597a923f7a7409d168fbc9226ea5874`. |
| Data provenance | Train manifest SHA-256 `5cd4e5155b0161e35b10aad2c98e14afba7d46407e635bd1c08d786e1e51765d`; validation manifest SHA-256 `53aa6f01447a43337a601279daf620c0df91d807773522cc92230fcab173db7c`; fetch report SHA-256 `75f0511bb94df2abd7e2734c0881af0282ed3797ec7b725cc07fa7dc8fd84a44`; excluded-scene config SHA-256 `a567348f615aa3f953c7fd95287e45ec4bfca0e55af9f6f4176305827efde138`. Only selected HTTP ranges were downloaded and per-video hashes were checked; the full source archives were not SHA-verified (pinned size/ETag only). |
| Configuration/models | Config `configs/pretrained_reuse/path_a_video_vjepa_expanded_seed17.yaml`, SHA-256 `3a81f75b80546f2484b8be0a8ad350aa81d3a79c8cc5c5e87081c6237edd8e3e`; training script SHA-256 `89943d01ba780671841d6b79ea85fd7c0078ca98ef95226207c1242ae33acaa0`; source commit `351136463af00ec5bad202b2d72823a37e229d27`. V-JEPA source commit `204698b45b3712590f06245fbfba32d3be539812`, checkpoint SHA-256 `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`, 8 frames at 384px; frozen MiniLM revision `4ca70771034acceecb2e72475f72050fcdde4ddc`, weights SHA-256 `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`. Readout: shared candidate MLP, AdamW lr `0.001`, weight decay `0.001`, 24 epochs, batch 8, seed 17; selected by minimum validation NLL. |
| Selected validation | 400 questions: Accuracy `0.3100`, NLL `1.3150`, Brier `0.6890`, ECE-15 `0.0573`; best epoch 2. Static descriptive (200): `0.3100 / 1.3437 / 0.6990 / 0.0628`. Temporal descriptive (200): `0.3100 / 1.2863 / 0.6791 / 0.0712`. |
| Learning curve / overfit | At selected epoch 2, train Accuracy/NLL/Brier/ECE was `0.4395 / 1.1819 / 0.6439 / 0.0584` (train CE `1.2189`), versus validation `0.3100 / 1.3150 / 0.6890 / 0.0573`. Train CE fell from `1.2736` at epoch 1 to `0.8566` at epoch 24. Validation NLL was best at epoch 2 (`1.3150`) and rose to `1.6378` at epoch 24; validation Brier rose from `0.6890` at epoch 2 to `0.8009` at epoch 24. Validation Accuracy fluctuated and reached `0.3675` at epoch 24, so the validation-NLL-selected checkpoint was not the final epoch. This is direct evidence of readout overfitting on this small synthetic task. |
| Matched held-out comparison | Both checkpoints were reloaded and scored on the exact same 400 expanded validation IDs with the same cached video and newly recomputed MiniLM features. The prior 16-scene v0 checkpoint (trained on disjoint earlier scenes) scored Accuracy/NLL/Brier/ECE `0.3325 / 1.3952 / 0.7095 / 0.0286`; the expanded-data checkpoint scored `0.3100 / 1.3150 / 0.6890 / 0.0573`. Thus expanded coverage improved NLL/Brier but reduced Accuracy by 2.25 points and worsened ECE; it is a mixed result, not a win. This is one-seed development validation and does not establish causality or generalization. Comparison artifact SHA-256 `104c3c5a4514508e4071bb538721a3661477ba299449fcd02fede6560687169`. |
| Runtime/artifacts | Local RTX 3080 Laptop 16 GiB; Python 3.11.9, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, CUDA runtime 12.4, pinned `timm` 1.0.15, PyAV 18.1.0, Transformers 5.6.2. Start/end: `2026-10-10 19:21:34`–`19:23:56 UTC` (`2026-10-11 04:21:34`–`04:23:56 JST`). Video feature extraction: train `74.62s`, validation `57.54s`; readout fit `4.34s`; peak CUDA allocation `464,645,120` bytes. The full run was about 142 seconds. External output: `C:\CodexArtifacts\pretrained-reuse-r2\clevrer-video-expanded-readout-v0-retry1`. Best readout SHA-256 `0db407b986c8a19e836883f3137d4ed99b684e962c531ba50907e3a6ebfcd25e`; predictions SHA-256 `78d7ea0ceeb247de363a7ada1b2d925aee9f9a2849ee9eac124849d67e6a8279`; video feature cache SHA-256 `2fa8935de933c1baa72f5562861c57ff7e482331f21316e30f1383ef1f3234d0`; run report SHA-256 `7a3b026b11a50e08e358fece496a363a8c831154c69eb3232533adfaf5ccf4f1`. The attempted original output path is an empty directory after a missing-`timm` initialization failure; it was preserved and the successful run used the separate `retry1` path. No package installation or upgrade occurred. |

Exact successful invocation (local pinned dependency path is needed for `timm`; no package was installed):

```powershell
$env:PYTHONPATH = "src;.;C:\CodexArtifacts\pretrained-reuse-r1\python-deps"
py -3.11 scripts/train_frozen_video_probe.py `
  --sample-dir C:\CodexArtifacts\pretrained-reuse-r2\clevrer-expanded-seed17-256train-200val `
  --vjepa-source C:\CodexArtifacts\pretrained-reuse-r1\vjepa2-source `
  --vjepa-checkpoint C:\CodexArtifacts\pretrained-reuse-r1\vjepa2_1_vitb_dist_vitG_384.pt `
  --vjepa-sha256 848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d `
  --text-model C:\CodexArtifacts\pretrained-reuse-r2\models\text-minilm-en `
  --text-revision 4ca70771034acceecb2e72475f72050fcdde4ddc `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\clevrer-video-expanded-readout-v0-retry1 `
  --config configs/pretrained_reuse/path_a_video_vjepa_expanded_seed17.yaml `
  --seed 17 --epochs 24 --batch-questions 8
```

This result shows that unique-scene expansion alone did not improve accuracy on
the held-out scenes, though probability losses improved. Training loss versus
validation NLL/Brier indicates early overfitting; the probe is too small to
separate representation limits from task ambiguity, sampling, or scorer
capacity. Do not increase epochs or treat this as a backbone ceiling. No sealed
audit or legacy final evaluation was loaded, and neither old outputs nor model
weights/media were added to Git.
