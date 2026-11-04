# NLI reference checkpoint audit

After the fixed MS MARCO scorer failed conditional numerical ranking, inspect
an existing checkpoint trained for entailment/contradiction rather than repeat
training or wording search on the observed diagnostic.

## Executed reference

`cross-encoder/nli-MiniLM2-L6-H768` revision
`b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`, Apache-2.0 in pinned card.
The publisher describes SNLI/MultiNLI training and labels contradiction,
entailment, neutral. This is English NLI; it is not a multimodal Decision model.
[Publisher card](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768/blob/b95119ce93d3e065de6214e38cd4a97b0f2f2c6d/README.md).

Downloaded only pinned model/tokenizer/card files, verified remote sizes and
weight LFS SHA-256 ec9c5b6a08b60d7a01b465c7ed44b496244dfd1bed3274c95875a74cea89ab35.
Loaded **82,120,707 parameters** (Hub total also includes514 integer buffer
entries). Weight file328,499,560bytes. Local CPU FP32, PyTorch2.6.0+cu124,
four threads: model-load/forward portion0.869sec. No missing, unexpected or
mismatched keys. Two publisher-card sentence pairs produced finite logits;
repeat inference was exactly equal. Outputs are in adjacent JSON. These two
examples establish runtime behavior only, not quality or calibration.

All artifacts: C:/CodexArtifacts/pretrained-reuse-r1/nli-minilm2-audit-01.
Weights were not added to Git. No training, GPU job, sealed evaluation, cloud
cost or legacy artifact modification.

## Parameter constraint

The existing modular reference has170,499,716 parameters. Replacing its
17,389,824 text encoder and196,865 readout with this82,120,707 reference would
produce approximately235,033,734 parameters, before any additional integration.
This is an accounting scenario, not an executed integrated model. It exceeds
the provisional200M underlying-parameter target, independently of precision.
Do not silently remove speech or another modality to make it qualify. It may
serve as a separately labeled research reference; product selection is open.

## Smaller candidate screened, not downloaded

`typeform/mobilebert-uncased-mnli` revision
`7bde1575aeaec3819a2ed901a35f114faf4a5318` has24,583,427 F32 elements in Hub
metadata (not a loaded count). Its pinned card says the license needs more
information. Rights remain REVIEW; no weights downloaded or executed.
[Publisher card](https://huggingface.co/typeform/mobilebert-uncased-mnli/blob/7bde1575aeaec3819a2ed901a35f114faf4a5318/README.md).

NLI requires premise/hypothesis semantics; arbitrary option softmax cannot be
called calibrated Decision probability merely because loading succeeds. Any
quality comparison must predeclare that transformation, preserve all cases,
and retain the current references. No such quality experiment ran in this audit.
