# Small decoder trainability: measured feasibility

Source4adba68, execution PASS_FEASIBILITY_ONLY. Fresh pinned SmolLM2-135M-Instruct;
no inherited adapted weights. Last4blocks26–29 plus finalnorm enabled;
134,515,008resident /14,160,960trainable parameters. Other parameters frozen.

Two8-example optimizer updates from the fixed train order:

| Step | Training CE | Preclip gradient norm | Timed seconds |
| ---: | ---: | ---: | ---: |
| 1 | 1.02650 | 7.62396 | 0.47083 |
| 2 | 1.48160 | 9.11784 | 0.19600 |

Different examples at each update: these two losses are not a learning trend.
No validation quality was inspected. Full20,972planned train+validation prompts
preflighted without truncation: max1,029,median208,p90=434,p99=652tokens.
Eight longest training prompts1029,1007,945,916,913,908,902,900 were also evaluated
without gradients;0.55459s. This does not measure worst-case backward memory.

Training peak1,619,352,576bytes; long-batch no-grad peak982,538,752bytes.
CPU/GPU warmup affects the first update. Total model-load/smoke/reload3.85474s
excludes corpus parsing/tokenization. RTX3080Laptop,FP32,SDPA,cloud cost$0.
Final-position projection matches normal LM projection within7.6294e-6.
Frozen tensor hash before/after equal; saved delta reload predictions exact.

Smoke delta56,655,883bytes, SHA1a66bbc299d1fc5165eb2980e4cfd8c551a72d92e3fffe8dc562b00b41f1f654.
All trainable paths, versions, input/model hashes and timing records are in the
companion JSON. Full effective config/IDs remain local under
C:/CodexArtifacts/pretrained-reuse-r2/small-decoder-feasibility-01.
The smoke delta is preserved and will not initialize the learning-curve run.

134,515,008language +22,056,576DINOv2 +37,760,640Whisper =194,332,224parameters,
leaving5,667,776for added components below200M. No shared multimodal bridge has
been trained or validated; a static image encoder is not native video reasoning.
Arithmetic feasibility does not establish general quality or calibrated probabilities.

Decision: local limited adaptation is practical. The separately frozen learning
protocol measures2,048updates from the original base with the same independent
train/validation IDs; it does not repeat the already-observed condition probe or
read sealed audit. Existing model/config/data artifacts remain unchanged.
