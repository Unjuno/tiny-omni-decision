# GRID / Lombard cross-source interface smoke

Before audio from GRID is available, perform one **intentionally unmatched**
interface smoke: use the first row in the existing pinned Lombard train
manifest for audio and the first four preselected GRID `s1` smoke videos. The
two sources are not known to depict the same utterance. Never construct a
correct target, score Accuracy, use the output for selection or infer
multimodal quality. No sealed audit or new training is involved.

Verify the existing audio archive, train manifest, GRID feature/head receipts,
Whisper encoder, audio head, normalizers and original correspondence
calibration by pinned hashes. Decode the actual Lombard WAV at its native rate,
use the existing 16 kHz audio frontend and frozen Whisper encoder/audio head;
use the already extracted frozen GRID video logits and verified video bytes.
Connect the native correspondence runtime to those real media and obtain a
four-ID JSON probability distribution. Check finite logits, probability sum,
cache identity and unchanged input hashes. CPU only, distinct new external
output, no changes to existing media, data or weights. This does not replace
the later matched GRID audiovisual input smoke or the grouped evaluation.
