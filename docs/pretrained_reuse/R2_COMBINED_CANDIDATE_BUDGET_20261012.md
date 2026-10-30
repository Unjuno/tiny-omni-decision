# Combined candidate parameter budget audit

Accounting of separately measured components; this is not a loaded combined model
or an adopted architecture. All counts are before quantization. Preserve the
170,499,716-parameter historical Path A reference as a separate configuration.

| Component | Parameters |
|---|---:|
| videoprism | 114,365,184 |
| dinov2 | 22,056,576 |
| whisper | 37,760,640 |
| crossencoder | 22,713,601 |
| numeric_delta | 131,712 |
| minilm | 17,389,824 |
| image_readout | 884,992 |
| video_readout | 393,984 |

Independent stack: **215,696,513**, above 200M. Thus independently improved
modality probes cannot be assembled and called a qualifying compact product.

If one resident Cross-Encoder also supplies candidate/query embeddings in place
of MiniLM, the same-shape readout hypothesis is **198,306,689**, leaving
**1,693,311** parameters for additional fusion/calibration. The classifier
and numeric delta are included, as are all 393,984 video-head parameters, even
where one control condition leaves some unused. Native Whisper transcript
likelihood uses its own tokenizer/decoder and needs no separate MiniLM head.

## Evidence limits and required next gate

- Numeric embedding training is in progress; its inclusion is conservative budget accounting, not selection.
- Cross-Encoder hidden states are 384-dimensional, but matching dimensions do not imply matching semantic features.
- Existing image/video readout weights were trained on MiniLM embeddings. Removing that encoder without retraining/comparison is invalid.
- A future shared-text comparison must freeze a selected text checkpoint, keep media features/splits/budget unchanged, retrain readouts on training data only, and measure validation against the existing representation.
- Simultaneous fusion, arbitrary question reasoning and deployment byte/runtime gates remain unverified. Do not fill the remaining parameter allowance without measurement.
- This audit loads no model, changes no active run and evaluates no held-out data. Component counts and source-file hashes are in the adjacent JSON.
