# Approved final-block Cross-Encoder adaptation

The user approved the bounded experiment on2026-10-11. Source is the pinned
22,713,601-parameter MS MARCO MiniLM-L6 model. Update exactly
bert.encoder.layer.5.* and classifier.*:1,774,849 parameters, no new model blocks.
All other parameters are frozen and hashed before/after every epoch.

Predefined run: seed17,3epochs,8questions per update, AdamW2e-5,
weight decay .01, constant LR, gradient clipping1.0. Dropout disabled in all
modules. Same frozen train23,319/validation2,540 IDs with binary criteria retained;
question/candidate first, state second,512tokens, state-only truncation. Reject
question/candidate overflow. Each question contributes equal CE regardless of
option count. Select earliest minimum validation NLL. Save best/final deltas
separately. Reload best and require exact saved-validation-logit parity. No
calibration fitting, new seeds, sealed audit or product promotion.

CPU suite316passed,1skipped; Ruff passed. Separate real-model smoke used16train
and8validation examples,2updates: frozen hash unchanged, nonzero selected
gradients, no frozen gradients, best reload/repeat parity PASS, peak allocated
VRAM410,522,624bytes. Smoke metrics are not quality evidence. Its result remains
C:/CodexArtifacts/pretrained-reuse-r2/text-crossencoder-adapt-smoke-01/result.json.

Main output: C:/CodexArtifacts/pretrained-reuse-r2/text-crossencoder-adapt-v1.
Runner: scripts/train_text_crossencoder.py. Inputs match the smoke command with
--smoke omitted. Main process is separately seeded; smoke state is not reused.
No exact optimizer resume is implemented. A stopped run would be incomplete.
