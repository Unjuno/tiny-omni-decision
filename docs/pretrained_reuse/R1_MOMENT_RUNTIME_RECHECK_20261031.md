# MOMENT-1-small local runtime recheck — 2026-10-31

## Result

The pinned MOMENT sensor checkpoint loaded from its local directory and completed a CUDA embedding forward on the RTX 3080 Laptop. This is an R1 runtime reproduction only; it is not a HARTH evaluation or a sensor-quality result.

- Model: `AutonLab/MOMENT-1-small`, Hub revision `411e288267f82cce86296dbe4d6c8bc533cc162f`
- Checkpoint: `model.safetensors`, 151,615,328 bytes, SHA-256 `785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877`
- Checkpoint tensor audit: 80 tensors, 37,901,512 FP32 parameters, 151,606,048 tensor bytes. The initialized `embedding` model contains 35,337,408 parameters (the task setup omits the checkpoint reconstruction head).
- Source: `moment-timeseries-foundation-model/moment`, commit `38f7310ad594100747ca2a8357e9c7ca7d323e0e` (MIT)
- Runtime: Python 3.11.9, PyTorch 2.6.0+cu124, CUDA 12.4, Transformers 5.6.2
- Input: all-zero `[1, 6, 128]` sensor tensor and all-valid `[1,128]` mask; task `embedding`; `reduction="none"`
- Output: finite `[1, 6, 16, 512]` tensor
- Forward time: 0.4865 s on the initial process and 0.2189 s on a separate immediate replay; peak allocated VRAM 162,730,496 bytes; peak reserved VRAM 184,549,376 bytes
- Cloud cost: $0

The source tree was clean at the pinned commit. The run used `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, loaded only the existing local checkpoint, and did not open any dataset. The environment lacks `python-dotenv`; the isolated process supplied a no-op `dotenv.load_dotenv` module shim because this local-only load has no `.env` configuration. No package was installed or updated. The initial failed import came from pointing at a separate older `moment` source tree; the audited snapshot is the `momentfm` package and imports successfully with the shim.

The checkpoint code warns that only its reconstruction head is pretrained and that task-specific classification heads require fine-tuning. The smoke used the pretrained encoder's embedding path; the warning is consistent with **not** claiming HARTH or other downstream quality.

## Artifacts

- Run record: `C:\CodexArtifacts\pretrained-reuse-r2\moment-harth-frozen-smoke-20261031-attempt-01\result.json` (SHA-256 `93CB4455C0303BE0E28587E785F1A96366702E95DDCA5DA6D014955D805F19DA`)
- Reproducer and config: `reproduce.py` (SHA-256 `0947885750BE1B0FD506A45ECA412AC7A80693E550D4F14A6259C18305D727CD`) and `config.json` (SHA-256 `8CE6C17007ACACC8D483A0E50267943B9D863A632DED7591407ED76D8BB9C1BF`)
- On-disk replay: `runtime-replay.json` (SHA-256 `A947FAFC314D32D8DAC7F45AFE354AFD0780B15B5E1E852224AE1411DC330E81`); contains the parameter count of the initialized embedding model.
- Existing weights and source remain read-only under `C:\CodexArtifacts\pretrained-reuse-r1\moment-small-sensor-r1-attempt-01\`
- Existing R1 audit record: `r1-sensor-audit-result.json`

The smoke output was written to a new attempt directory. Existing checkpoint, source, data, and prior R1 records were unchanged.

## R2 decision

Do not fit or select a new MOMENT readout on the existing HARTH validation subjects. The 2026-10-12 sensor report states that the same three participants were used for checkpoint selection and later metric reporting; those validation IDs are exposed development data. No untouched, approved sensor cohort is currently admitted for this candidate. Therefore R2 M1 sensor quality remains unmeasured for MOMENT, and no new training was justified by this recheck. Reopen this path only after a rights-cleared, leakage-audited cohort with independent groups and a fixed validation protocol is available.

## Verification

On the target branch, these existing CPU tests passed: 38 passed across `test_native_feature_decision.py`, `test_aligned_features.py`, `test_native_observations.py`, and `test_fusion_decision.py`. Ruff lint passed for the readout and its test. `ruff format --check` reports pre-existing formatting changes in those two committed files; neither file was modified in this recheck.

PR #15 remained open/draft at target head `505955942ac8c241f010190c886806a77994b7fa`; its two GitHub test checks were completed successfully when checked. No sealed audit data was opened.
