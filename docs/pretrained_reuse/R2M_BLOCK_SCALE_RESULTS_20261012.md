# Fixed block-scale result

Protocol and source frozen at c082fca before execution. One seed17 run completed100epochs; selected26 by minimum validation NLL. Exact checkpoint reload passed. No additional coefficients or seeds tried.

| Epoch | Train NLL | Validation NLL | Validation Accuracy |
|---:|---:|---:|---:|
| 0 | 2.607851 | 2.607494 | 0.025641 |
| 7 | 0.424808 | 0.608831 | 0.820513 |
| 26 | 0.045475 | 0.350965 | 0.871795 |
| 50 | 0.014194 | 0.373681 | 0.871795 |
| 75 | 0.008781 | 0.390290 | 0.871795 |
| 100 | 0.006480 | 0.402890 | 0.871795 |

Selected Accuracy0.871795, NLL0.350965, Brier0.199967, ECE0.092742. Original joint:0.769231/0.572209/0.322131/0.119063. Sensor-only remains better (Accuracy1.0,NLL0.163704). Fusion adoption criteria remain unmet.

Paired subject-cluster NLL differences (scaled minus comparator),10000 replicates,seed17,percentile95:
- joint: -0.221243,95% interval [-0.2730724725498966, -0.16602880494324032].
- sensor: 0.187262,95% interval [0.09209469954266052, 0.35978904157449365].

Only3 development subjects and reused checkpoint-selection validation; not an independent audit. Scaling alters initialization contributions, gradients and regularization jointly. It supports sensitivity to this parameterization, not a unique causal explanation or general fusion gain.

Artifact: `C:/CodexArtifacts/pretrained-reuse-r2/vidimu-joint-blockscale-arm-01`; comparisons in `vidimu-blockscale-comparison-01`. All101 prediction snapshots, normalization, config, best/final weights and hashes preserved. Readout84591 parameters, total with frozen VideoPrism114449775. CPU fit/evaluation/save loop1.136s excluding startup/load; no paid compute. No sealed audit access, legacy changes or product promotion.
