# Fixed semantic adaptation on the existing image-state bridge

Execution **PASS**; fresh full replay and independent verification **PASS**.
No product adoption: the original semantic fit still selects step0 and its
trained1024 endpoint remains ineligible under its frozen retention contract.

## Scope and actual execution

Executed source `1d6282753dc475ab02dd4e18fb57fc278ab1202a`,
[fixed protocol](R2_SEMANTIC_NATIVE_COMPARISON_PROTOCOL_20261015.md) and
[config](../../configs/pretrained_reuse/semantic_native_comparison_v1.json).
The same pinned DeBERTa xsmall70830722 parameters is loaded in FP32/eager;
all parameters frozen, entailment-logit0, softmax across all40 supplied options.
The selected delta is byte-identical to step0 and is not run as a third model.

Compare pretrained step0 and the fixed1024 endpoint from the completed
[semantic fit](R2_DECISION_SEMANTIC_FIT_RESULTS_20261015.md), with no new optimizer
update, calibration, prompt search or checkpoint selection.
Preserve all256 TRAIN images /1024 questions /40 options from the old diagnostic.
Correct annotations and frozen-head Top1 predictions provide the same four-field
state interface. Only state/question/candidate content enters the public forward.

This is an observed TRAIN diagnostic. The image head was trained on these images.
The original image features and head outputs are reused and hash-verified;
there is **no fresh image-encoder forward** or independent native-quality claim.
The evidence identifies failures in this interface; it does not measure the full
Omni stack or prove a general model/encoder capacity ceiling.

## Main comparison

Each row has1024 questions. ECE is pooled ECE-15; macro ECE averages the four
question-type ECE values and is reported separately. NLL/Brier are lower-is-better.

| Model | Input facts | Accuracy | NLL | Brier | Pooled ECE | Macro ECE | Mean confidence |
|---|---|---:|---:|---:|---:|---:|---:|
|step0|Oracle|41.80%|2.47379|.85481|.29450|.36236|.12347|
|1024 endpoint|Oracle|73.83%|1.51596|.47179|.16969|.29823|.80456|
|step0|Image-predicted|41.11%|2.53682|.86038|.28782|.36350|.12332|
|1024 endpoint|Image-predicted|73.34%|1.78240|.48398|.17658|.30091|.80374|

Endpoint-minus-step0 Accuracy: oracle **+32.031pp**,95% interval[29.785,34.241]pp;
image-predicted **+32.227pp**,[30.116,34.350]pp. These are10000 seed17 paired
bootstrap draws of whole annotation-state groups:252 clusters,248 single-image
and4 double-image. Each draw uses the question-weighted ratio with its actual
denominator. These development intervals do not account for the broader adaptive
research history or make this a new blind test.

## The aggregate improvement leaves complete count-question failure

Every endpoint question type has256 examples:

| Question | Oracle Accuracy | Oracle NLL / Brier / ECE | Image-predicted Accuracy | Image-predicted NLL / Brier / ECE |
|---|---:|---|---:|---|
|Color|95.31%|.71069 / .28743 / .42714|95.31%|.71408 / .28919 / .42931|
|Texture|100%|.02022 / .00406 / .01737|98.05%|.21496 / .04302 / .02664|
|Count|**0%**|5.31294 / 1.59430 / .72902|**0%**|6.18057 / 1.60240 / .72826|
|Shape|100%|.01997 / .00136 / .01941|100%|.01999 / .00133 / .01944|

See the [60-row metrics CSV](R2_SEMANTIC_NATIVE_METRICS_20261015.csv) for both
models, all control conditions, per-type and pooled metrics.

For oracle endpoint, all268 errors choose a true but unrequested attribute:
160 count→texture,96 count→shape,12 color→texture. The learned model can rank the
correct value first **within the correct field** for every oracle question, but
that restricted attribution metric must not replace the real all40-option score.
The product receives no ground-truth field mask or task-routing label.

Oracle all-four-correct is0/256; changed-question pairs both correct are48.44%.
Predicted all-four is also0/256 and pair correctness47.79%. The count failures
prevent genuine four-question reuse despite gains on the other questions.

Image prediction is not thereby solved. Its separate ten-option native head is
94.24% overall on these TRAIN images: color100%,texture98.05%,count78.91%,shape100%.
Of59 native-fact errors,54 are count errors hidden behind the semantic count
failure and5 are texture errors. Only5 of the273 endpoint predicted-decision
errors are same-field wrong-value errors;268 choose true-but-unrequested facts.
The small0.49pp oracle/predicted aggregate gap must not be read as proof that
perception is sufficient. It is masked by the downstream count failure.

## Controls and coverage analysis

| Input control | step0 Accuracy | Endpoint Accuracy |
|---|---:|---:|
|Empty state|2.93%|2.83%|
|Empty question|24.22%|25.00%|
|Next-image state, original targets|6.74%|8.69%|
|Next-image state, donor-corrected targets|41.80%|73.83%|

The donor changes912/1024 targets;112 remain unchanged. Both empty-question
models have zero changed-question pairs both correct. These controls support
use of both facts and question on this fixed interface; they do not establish
general temporal or jointly multimodal understanding.

All40 candidates are retained. First-image permutation and singleton probes
preserve actual selected IDs. Full fresh replay matches all26 payload files
byte-for-byte, including effective config, scores, predictions, metrics and inputs.

Saved-error accounting additionally reads only the frozen TRAIN files:
16384 new questions include0 exact `How many objects are there?` and0 `how many`
questions;2048 replay questions include0 exact matches and129 `how many` questions
from other tasks. The new binding supervision uses property names and different
candidate descriptions. This is evidence of a coverage/interface difference,
**not a causal proof that adding this one phrase would solve transfer**.
No transfer or sealed-audit payload was opened.
[Machine-readable errors and input hashes](R2_SEMANTIC_NATIVE_ERRORS_20261015.json).

## Verification, platform correction and resources

The first smoke at source `04aa3c1` stopped before model import,0 forwards:
Windows error206 from a35615-character Git command containing941 filenames.
The corrected58-character `git ls-files -z` command preserves the exact same
941-file inventory and all hashes. Regression tests cover1000 long paths and
rejection of every missing required tracked file. The original failure directory
and log remain intact. No input, option, checkpoint or numeric threshold changed.

| Attempt | Status | Wall seconds | Candidate forwards | Output bytes |
|---|---|---:|---:|---:|
|smoke-01|INCOMPLETE before model load|1.321|0|66308|
|smoke-02|PASS|17.893|2160|634519|
|full primary|PASS|223.747|185040|29772491|
|fresh full replay|PASS|232.865|185040|29773631|

Primary UTC2026-10-15 12:19:42–12:23:25, PID7632;
replay12:24:48–12:28:40, PID24312. All terminal process exits are0 except failed
smoke01. Local RTX3080Laptop16GB only; cloud cost0. Successful GPU-process wall
time474.505s (~7.91min), not measured GPU-kernel duty time. Maximum CUDA allocation
448132096bytes (~.417GiB); maximum host working set including smoke1891344384bytes
(~1.761GiB). Max sequence length58; no truncation. Python3.11.9,
Torch2.6.0+cu124, CUDA12.4, Transformers5.6.2; no environment upgrade.

Ruff passes; pre-run CPU suite1698 passed/1 skipped;25 manifest commands pass.
Executed source CI runs38733656240/38733650295 both succeed. Publication checks
pass1705 CPU tests/1 skip and25 manifest commands after adding the independent
verifier fixtures. Publication CI is checked on its own commit rather than
attributed to the executed inference source.

The independent verifier imports no project scorer or torch. It recomputes all
20480 primary/replay predictions, per-type/pooled metrics, pair/routing measures,
error intersections and both bootstrap distributions, plus source/input/artifact
hashes and the exact original cohort. Seven numerical fixtures pass.
[Verification record](R2_SEMANTIC_NATIVE_VERIFICATION_20261015.json) ·
[verifier](../../scripts/verify_semantic_native_comparison.py) ·
[saved-product summary](../../scripts/summarize_semantic_native_comparison.py).

## Artifacts and reproducibility

Root `C:/CodexArtifacts/pretrained-reuse-r2/`; every attempt uses a separate path.
No prior Teacher, fit, corpus, model or checkpoint is overwritten. No weights/media
or full prediction payloads are committed to Git.

| Local artifact | SHA256 |
|---|---|
|`semantic-native-comparison-smoke-01/result.json`|`f828a81a272d1648d7d316721ea13b39e977b4cc3dfd4e0a55a00e08a07ccf31`|
|`semantic-native-comparison-smoke-02/result.json`|`46ee895e44752117af5926c5769e87271d4c132145f1b3ef165651fcf1a84a2d`|
|`semantic-native-comparison-01/result.json`|`e7901306ca31d6c32fee8d4e888d01f0eb4f78ec6854e8f71c0e72ea75ff289b`|
|`semantic-native-comparison-replay-01/result.json`|`4a28e6ad6d444af210c899de5114dad1bc5028433c58bc3dcc524be50fa9a1f8`|
|`semantic-native-independent-verification-01.json`|`ad95c9e47944d080921721ad448c4ea8804c443ce7a9de4f7132b9198e397007`|
|Executed `verify-semantic-native-v1.py`|`fd90b29cc69507653a32754515be3c990e5a5d9982fcd8ee93eb2de810ed966f`|
|Executed `summarize-semantic-native-v1.py`|`9a77503de76f6e2d73611de4f8821c179f73ad7f21f5a45b3d4ac874e30725a5`|
|`semantic-native-error-accounting-01/error-accounting.json`|`8ed484b8713161b468c58b9bf0725599716060027819b707d8c9a3c4eccedb0c`|
|`semantic-native-error-accounting-01/per-task-metrics.csv`|`2d3af2b7001194cd1d6de17fe2c8a936a8e293d054cb4fe4254331e5d9d2d12f`|

```powershell
python -u -m scripts.evaluate_semantic_native_comparison --smoke --output C:/CodexArtifacts/pretrained-reuse-r2/semantic-native-comparison-smoke-02
python -u -m scripts.evaluate_semantic_native_comparison --output C:/CodexArtifacts/pretrained-reuse-r2/semantic-native-comparison-01
python -u -m scripts.evaluate_semantic_native_comparison --output C:/CodexArtifacts/pretrained-reuse-r2/semantic-native-comparison-replay-01 --replay-reference C:/CodexArtifacts/pretrained-reuse-r2/semantic-native-comparison-01 --replay-sha256 e7901306ca31d6c32fee8d4e888d01f0eb4f78ec6854e8f71c0e72ea75ff289b
```

Use new output paths for reproduction. The exact inference code/config/protocol
are pinned to the executed source; later publication adds verifier/report files.

## Decision and next work

The fixed adaptation improves this observed bridge, but does not qualify the
selected model or meet95–98%. Close this comparison. Do not repair the result by
masking candidates to their known target field, changing wording on these256
images, selecting a different checkpoint or fitting temperature here.

Prioritize condition-to-property transfer in the shared decision core, with
natural questions, candidate descriptions and hard true-but-unrequested negatives.
Before another fit, define independent supervision and an interface-matched
retention/calibration/transfer contract. Distinguish wording and candidate
representation changes; do not repeat closed prompt/factorization sweeps without
a new discriminating training hypothesis. The known count failure remains a
regression test, not the new final benchmark. Perception count errors must remain
visible as a separate unresolved requirement.

Whole <=200M loaded parameters, native speech, temporal/joint benefit, sensor/screen,
OOD and all required95–98% tasks remain unmet. No quantization or product promotion.
