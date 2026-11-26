# D1 Omni pinned source and parameter audit

Revision `02b55d7076f15129e59ab3f94783f32c4b088674`. No tensor payload downloaded, no remote code executed,
no model instantiated, no quality evaluation. This is a source/header audit,
not a strict-load or performance pass.

| Tensor prefix | Serialized parameters |
|---|---:|
| encoder | 354,483,968 |
| head | 26,248,193 |
| vision (including projection) | 94,234,880 |
| audio (including projection) | 112,194,065 |
| total | **587,161,106** |

Counts sum pinned safetensors header shapes fetched using verified HTTP206
ranges. Header SHA and all source hashes are in the companion JSON. This is
serialized tensor accounting, not measured deduplicated loaded parameters.
Embedding table alone is65,536x1,024=67,108,864. Removing it would still leave
520,052,242 parameters and destroy the current text interface. Keeping just both
media prefixes already totals206,428,945 before the shared encoder/head.
Thus disabling choice output or quantizing storage cannot satisfy <=200M
prequantization with this unchanged architecture. No trimming/retraining done.

## Useful implementation evidence

`prompt.py:encode` puts the question and all options in one sequence. `encoder.py`
uses contextual option marker states and a shared scorer after two additional
Transformer head layers. All question types use these same parameters; binary
mode does not remove the head or trunk.

`modeling_d1.py` builds separate trunk rows for individual questions, even when
media prefix features have been extracted once. Batched execution is not proof
that the whole state is encoded once for every question. This distinction
matters for our many-questions-per-observation reuse objective.

The public path rejects simultaneous images and audio. Media prefixes precede
text; multiple images do not establish temporal video reasoning. Configured
temperature scaling is applied to text-only requests, not the media branches.
Do not inherit those temperatures or assume multimodal calibration is solved.

Our current CrossEncoder scores candidate pairs independently. Joint candidate
context is a concrete alternate mechanism worth testing with a compact existing
encoder. It is a hypothesis, not evidence that the missing feature caused our
failures. It requires a fixed task-matched comparison, option-order controls,
measured token cost, and retraining before any quality claim. No implementation
from this custom-license repository has been copied into product code.

## Rights and disposition

License metadata and file identify LFM Open License v1.0, not Apache/MIT.
Keep product incorporation/derivatives at REVIEW; this audit does not establish
redistribution eligibility. Full pinned license retained locally. D1 remains an
oversize research reference, with local quality and temporal capability unknown.

Artifacts: `C:/CodexArtifacts/pretrained-reuse-r1/d1-omni-source-audit-01`.

Primary sources:
- [Pinned config](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/config.json)
- [Pinned encoder](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/encoder.py)
- [Pinned inference](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/modeling_d1.py)
- [Pinned prompt](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/prompt.py)
- [License](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/LICENSE)
