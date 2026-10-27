# Open Images full train support audit

Freeze this protocol before streaming the pinned V6 human image-label train
object, generation 1580742923196487, ETag a0c8491396bfa736bec85cb172180bd3.
Require the full 2,474,671,021 bytes and SHA-256
`a97b079ccccf635a980bfe4b8180a0c844b4d9c887b2b7bc56535f539af044ea`.
The first execution failed closed because this protocol initially transcribed
the hash incorrectly, omitting `b7`; the corrected value matches the existing
train target support manifest and source preflight. No result was saved.
Do not save source rows or media. Do not use validation, test, or model outputs.

Use the frozen 11-class target scope, verification labels, and the existing
sorted image grouping and contradiction checks. For each class count positive
and negative image groups; count positive target/image pairs with at least
three distinct explicitly negative in-scope alternatives. For each eligible
target also count each available negative class as a possible distractor;
these are availability counts, not an actual sampled option distribution.
Report eligible unique images separately from target/image pairs and include
all classes, including zero support. Never infer absence from missing labels.

The audit does not establish image rights, asset availability, independent
splits, achievable balanced sampling, model quality, or a valid benchmark.
An image can support multiple targets. Any subsequent task requires a frozen
sampling design, provenance and group-integrity review before training.
