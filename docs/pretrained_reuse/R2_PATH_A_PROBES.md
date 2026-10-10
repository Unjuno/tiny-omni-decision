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

## R2 status

- Implemented and exercised frozen-feature candidate readouts for text, image, and a small video task. Per-example validation probabilities, epoch metrics, source/media hashes, and selected heads are saved outside Git.
- Audio-only closed-set keyword evaluation is complete using frozen Whisper Tiny features and a small candidate readout; it does not establish broad speech understanding. Train/validation speaker and media gates passed, and artifacts are outside Git.
- Video cache reuse is measured on two questions per scene: half of the would-be encoder calls were avoided. Persistent event updates, multi-observation state, and joint audio-video tasks remain untested.
- No sealed audit, legacy final evaluation, or training checkpoint was loaded. No backbone was updated. No model weights or media are added to Git.
- This is not a selected release candidate. Open-Jev access and V-JEPA checkpoint rights remain unresolved; broader real text, audio, natural-image, joint-modality, calibration and deployment gates remain open.
