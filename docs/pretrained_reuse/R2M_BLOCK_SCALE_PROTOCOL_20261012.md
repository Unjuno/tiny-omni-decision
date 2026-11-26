# Fixed VIDIMU block-scale diagnostic

Development-only follow-up to the negative early-concatenation result. Run one
seed17 joint readout with exactly the original data, initialization seed,
optimizer,100epochs and earliest minimum validation-NLL selector. Scale only
the standardized video block by sqrt(360/6144); sensor and placement unchanged.
The coefficient follows feature dimensions, not a validation search. No extra
seeds, ratios, epochs or architecture search in this experiment.

This intervention changes initial logit scales, gradient geometry and effective
regularization together. It tests whether this fixed parameterization helps;
it cannot isolate a unique cause. Both original arms and all frozen artifacts
remain intact. Output: `vidimu-joint-blockscale-arm-01` under the existing R2
artifact root. Record all101 predictions, curves, best/final hashes and reload.
Compare to the existing joint and sensor arms on identical39 validation IDs.
Even an improvement does not establish general fusion or justify promotion;
reused-development-validation and3-subject limitations remain. Sealed audit
is not accessed. No additional backbone parameters or encoder training.
