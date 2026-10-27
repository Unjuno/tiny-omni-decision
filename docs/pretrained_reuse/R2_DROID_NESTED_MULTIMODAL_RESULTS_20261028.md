# DROID nested multimodal probe: result

The [fixed nested procedure](R2_DROID_NESTED_MULTIMODAL_PROTOCOL_20261028.md)
completed all five outer folds on the 275 train recordings. Each arm chose
update 1 in every inner selection. The pretrained encoders were frozen and the
linear heads used the same fold memberships; raw OOF predictions are saved.

| Inputs | Accuracy | NLL | Brier | ECE-10 |
|---|---:|---:|---:|---:|
| Sensor | 15.64% | 1.9765 | 0.8632 | 0.1017 |
| Sensor + video | 22.55% | 3.2896 | 1.1307 | 0.4401 |
| Sensor + text | 29.09% | 1.8064 | 0.8085 | 0.0396 |
| Sensor + video + text | 26.55% | 3.0046 | 1.0528 | 0.3615 |
| Outer-training class prior, post-hoc control | 37.09% | 1.6906 | 0.7750 | 0.0004 |

The class-prior control uses only the appropriate outer training labels and
was computed after the four readouts; its low ECE does not indicate useful
discrimination. None of these readouts beats it in aggregate accuracy or NLL.
Adding video worsens NLL and calibration in this setup. Text improves the
sensor-only readout but remains worse than the class prior. Choosing update 1
everywhere means this schedule/readout is poorly matched to the small sample;
the experiment does not establish that video or language lacks useful
information, nor demonstrate general multimodal complementarity. The data and
folds were previously inspected, and recording identity is not independent
operator or site identity. The earlier word-hash and learning-curve outputs
remain unchanged.

Output:
`C:/CodexArtifacts/pretrained-reuse-r2/droid-nested-multimodal-20261028-attempt-01/`;
`result.json` SHA-256
`b438c4aafe737148de41072f9ec437ff643be58ac6401602c4b35d51f106a064`.
The result stores fold-level metrics, selected updates and inner losses; the
OOF NPZ stores component assignment and predictions. The separate focused
test verified held labels do not affect fitting, and Ruff passed. No held-out
validation or sealed audit was scored, and no product model was promoted.
