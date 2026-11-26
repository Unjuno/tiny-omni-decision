# Question wording / alias factorial results

All8192 rows per model completed. No training, calibration or checkpoint selection.
Same128 states, two numeric regimes, four question cells. All targets unchanged.
Generic/domain_alias reference cells reproduce prior probabilities exactly (maxdelta0).
Committed constructor also regenerates the complete prepared input file hash exactly.

| Model | Cell | Acc | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|
| parent | base-generic | 0.834961 | 0.416171 | 0.234453 | 0.043088 |
| parent | base-generic_alias | 0.517578 | 1.230788 | 0.652368 | 0.151378 |
| parent | base-domain | 0.493164 | 1.113316 | 0.609456 | 0.042947 |
| parent | base-domain_alias | 0.400391 | 1.275728 | 0.690228 | 0.072482 |
| parent | expanded-generic | 0.742188 | 0.632556 | 0.340084 | 0.041618 |
| parent | expanded-generic_alias | 0.447266 | 1.324180 | 0.698162 | 0.146689 |
| parent | expanded-domain | 0.429688 | 1.619024 | 0.740181 | 0.145191 |
| parent | expanded-domain_alias | 0.340820 | 1.710499 | 0.847008 | 0.262422 |
| candidate | base-generic | 0.843750 | 0.390302 | 0.220166 | 0.033666 |
| candidate | base-generic_alias | 0.530273 | 1.283418 | 0.658893 | 0.162220 |
| candidate | base-domain | 0.488281 | 1.083266 | 0.589124 | 0.055703 |
| candidate | base-domain_alias | 0.401367 | 1.321739 | 0.708043 | 0.112531 |
| candidate | expanded-generic | 0.746094 | 0.634908 | 0.340125 | 0.052818 |
| candidate | expanded-generic_alias | 0.499023 | 1.382572 | 0.690680 | 0.168658 |
| candidate | expanded-domain | 0.409180 | 1.654046 | 0.730929 | 0.134111 |
| candidate | expanded-domain_alias | 0.361328 | 2.061229 | 0.897738 | 0.320425 |

## Paired effects

Difference is second minus first. NegativeAccuracy is worse; positiveNLL/Brier is worse.
| Model | Contrast | Metric | Difference |95%interval|
|---|---|---|---:|---|
| parent | base-alias_generic | accuracy | -0.317383 | [-0.342773,-0.291992] |
| parent | base-alias_generic | nll | +0.814617 | [+0.755958,+0.872948] |
| parent | base-alias_generic | brier | +0.417915 | [+0.391198,+0.443829] |
| parent | base-alias_domain | accuracy | -0.092773 | [-0.123047,-0.062500] |
| parent | base-alias_domain | nll | +0.162412 | [+0.126500,+0.198003] |
| parent | base-alias_domain | brier | +0.080771 | [+0.062341,+0.099190] |
| parent | base-wording_without_alias | accuracy | -0.341797 | [-0.369141,-0.314453] |
| parent | base-wording_without_alias | nll | +0.697145 | [+0.664195,+0.730672] |
| parent | base-wording_without_alias | brier | +0.375004 | [+0.353646,+0.396600] |
| parent | base-wording_with_alias | accuracy | -0.117188 | [-0.143555,-0.089844] |
| parent | base-wording_with_alias | nll | +0.044940 | [-0.008936,+0.098309] |
| parent | base-wording_with_alias | brier | +0.037859 | [+0.015790,+0.060404] |
| parent | expanded-alias_generic | accuracy | -0.294922 | [-0.319336,-0.269531] |
| parent | expanded-alias_generic | nll | +0.691624 | [+0.632649,+0.751926] |
| parent | expanded-alias_generic | brier | +0.358078 | [+0.327459,+0.388223] |
| parent | expanded-alias_domain | accuracy | -0.088867 | [-0.109375,-0.068359] |
| parent | expanded-alias_domain | nll | +0.091475 | [+0.037091,+0.146852] |
| parent | expanded-alias_domain | brier | +0.106826 | [+0.087831,+0.126571] |
| parent | expanded-wording_without_alias | accuracy | -0.312500 | [-0.338867,-0.286133] |
| parent | expanded-wording_without_alias | nll | +0.986468 | [+0.913501,+1.058805] |
| parent | expanded-wording_without_alias | brier | +0.400097 | [+0.371826,+0.427394] |
| parent | expanded-wording_with_alias | accuracy | -0.106445 | [-0.129883,-0.083008] |
| parent | expanded-wording_with_alias | nll | +0.386320 | [+0.322186,+0.451752] |
| parent | expanded-wording_with_alias | brier | +0.148846 | [+0.123617,+0.174678] |
| candidate | base-alias_generic | accuracy | -0.313477 | [-0.339844,-0.287109] |
| candidate | base-alias_generic | nll | +0.893116 | [+0.827214,+0.960364] |
| candidate | base-alias_generic | brier | +0.438728 | [+0.408488,+0.469351] |
| candidate | base-alias_domain | accuracy | -0.086914 | [-0.117188,-0.056641] |
| candidate | base-alias_domain | nll | +0.238473 | [+0.205348,+0.272540] |
| candidate | base-alias_domain | brier | +0.118919 | [+0.102328,+0.136044] |
| candidate | base-wording_without_alias | accuracy | -0.355469 | [-0.381836,-0.329102] |
| candidate | base-wording_without_alias | nll | +0.692964 | [+0.663514,+0.722532] |
| candidate | base-wording_without_alias | brier | +0.368958 | [+0.350380,+0.387405] |
| candidate | base-wording_with_alias | accuracy | -0.128906 | [-0.155273,-0.103516] |
| candidate | base-wording_with_alias | nll | +0.038321 | [-0.028262,+0.103860] |
| candidate | base-wording_with_alias | brier | +0.049150 | [+0.022054,+0.076120] |
| candidate | expanded-alias_generic | accuracy | -0.247070 | [-0.274414,-0.219727] |
| candidate | expanded-alias_generic | nll | +0.747664 | [+0.672252,+0.826880] |
| candidate | expanded-alias_generic | brier | +0.350555 | [+0.317334,+0.383220] |
| candidate | expanded-alias_domain | accuracy | -0.047852 | [-0.069336,-0.027344] |
| candidate | expanded-alias_domain | nll | +0.407182 | [+0.315388,+0.500928] |
| candidate | expanded-alias_domain | brier | +0.166808 | [+0.142241,+0.191714] |
| candidate | expanded-wording_without_alias | accuracy | -0.336914 | [-0.361328,-0.312500] |
| candidate | expanded-wording_without_alias | nll | +1.019139 | [+0.946146,+1.093086] |
| candidate | expanded-wording_without_alias | brier | +0.390804 | [+0.365629,+0.416054] |
| candidate | expanded-wording_with_alias | accuracy | -0.137695 | [-0.165039,-0.111328] |
| candidate | expanded-wording_with_alias | nll | +0.678657 | [+0.560356,+0.800008] |
| candidate | expanded-wording_with_alias | brier | +0.207058 | [+0.172179,+0.242362] |

## Interpretation

For the coverage checkpoint, adding the exact alias preamble to the unchanged
generic question reduces baseAccuracy84.38% to53.03%:−31.35pp,
paired95%interval[−33.98,−28.71]pp. Expanded-range reduction is−24.71pp.
The old checkpoint shows the same failure direction. This is an actual
question-prefix intervention with states/options/targets fixed, not a state-key change.
It does not isolate added token length, distractor field mentions or alias semantics.

Removing the prefix from the domain family improves40.14% to48.83% in base
and36.13% to40.92% in expanded range; both remain weak. Domain wording without
prefix is still35.55pp below generic in base. Wording changes include syntax as
well as the adjective, so a single-word causal claim remains unjustified.

The two weaknesses coexist. Do not simply strip the prefix and claim general
reasoning is fixed, or tune a template-specific preprocessor on these labels.
A next learning hypothesis should address train-only compositional instruction
binding with independent templates and retain general original-text validation.
This diagnostic itself launches no such run. Existing models remain preserved.

## Accounting and limits

Count/conditional questions are identical retention controls in all four cells.
Full per-task metrics and opposite-condition paired correctness are in JSON.
Repeated states/cells are not independent samples. Paired state bootstrap:
10000draws,seed17,percentile95%, no multiplicity correction. Development data
were observed before this diagnostic; these are not blind audit results.
Timed inference72.104s, peak allocatedVRAM194713088bytes.
Cloud charges0; electricity unmeasured. GPU was idle before launch.
Execution source63b4226; input preparation preceded that commit, and its original
builder hash is retained. Exact reconstruction includes the original Windows CRLF bytes.
Artifacts `C:/CodexArtifacts/pretrained-reuse-r2/transfer-factorial-eval-01` and `C:/CodexArtifacts/pretrained-reuse-r2/transfer-factorial-inputs-01`.
Four modalities>=95%/<=200M objective remains unmet. No image/audio/video inference.
