# Fixed VideoSAUR development validation protocol

Frozen before validation extraction or scoring. Evaluate the existing train-only
epoch50 endpoint; no further optimization, calibration, seed search or checkpoint
selection. This is previously observed development validation, not a blind audit.

- All257 existing validation videos /514 initial-final questions; preserve IDs,
  options, targets and order. No difficult-example filtering.
- Pair SHA256: `30846aaa87e2292e97ee3187c5f2f986ba3c3d57d46a249b361e5a14c0cc8ec2`.
- Media manifest SHA256: `213bd33994fb80d79ee05ee0580af2cc9915dfc61094aed00c847c2c67f1d546`.
- Train/validation video IDs and media hashes checked disjoint before extraction.
  Existing script/participant grouping limitations remain; not participant-disjoint.
- Identical frozen VideoSAUR checkpoint, runtime loader, full-frame preprocessing,
  seed17 per clip, all consecutive pre-cut frames, encoder batches6, predicted-state
  recurrence and224x129 slot/time features as the training protocol.
- The extraction harness is a copy of the completed train harness with only split,
  corpus hashes, expected count, status and new output path changed. Syntax checked.
  SHA256: `9d317cc4511ce193966e46dcc7c6e84ee84673f11d2762a0c69605f858f86534`.
- New output: `C:/CodexArtifacts/pretrained-reuse-r1/videosaur-valid-slots-01/`.
  Keep train extraction and all prior experiments intact. No annotation boxes or
  question labels enter the feature extractor. Require complete ledger, media and
  feature hashes, exact frame sampling and unchanged frozen weights before scoring.

## Fixed evaluation

Use existing MiniLM validation candidate features after their ledger/hash/ID checks.
Load `perception-occlusion-slot-train-readout-01/final.pt` with strict state loading,
SHA256 `3d12ea26148724ba0113e251b29c1e69e9aca9ff7c3bd3f95aed6d685c97b765`.
Use the same CPU prediction code and input_dim129. Evaluate normal, cyclic+1
mismatched-video and zero-video inputs; also evaluate the saved untrained head
as a reference, without selecting between heads. Save all per-example probabilities.

Report Accuracy/NLL/Brier/ECE, initial/final metrics, both-correct pairs,
changed-target/unchanged-target groups, condition switches and the train/validation
gap. Include paired video-level bootstrap95% percentile intervals for normal
minus mismatch Accuracy and NLL:10,000 replicates, seed17, resample entire pairs.
No multiple-candidate selection or new blind-evaluation claim follows from these
intervals. Keep all results, including negative ones. No product promotion.

## Execution

```powershell
python -c "import runpy;runpy.run_path('C:/CodexArtifacts/pretrained-reuse-r1/videosaur-source-audit-01/extract_valid_slots.py',run_name='__main__')"
```

The complete train extraction took80.5minutes for48,094frames. Validation duration
depends on its actual pre-cut frame count; record that count and measured time.
Local GPU only. Inspect existing process/output before launching; no duplicate run.
