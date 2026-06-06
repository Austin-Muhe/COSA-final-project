# Experiment Status

## Validated inputs currently available

- Dataset: `ETTh1`
- Model checkpoints:
  - `DLinear / ETTh1 / pred_len={96,192,336,720}`
  - `DLinear / weather / pred_len={96,192,336,720}`

The repository currently does **not** include:

- iTransformer checkpoints

`experiments/run_cosa_plus.py` now performs a preflight check so future runs fail
clearly when required data/checkpoints are missing instead of silently evaluating
randomly initialized models.

## Valid results

### ETTh1 / DLinear / horizon=96 ablation

| variant | MSE | MAE |
|---|---:|---:|
| original | 0.452734 | 0.452890 |
| vec_gate | 0.452959 | 0.453206 |
| rich_ctx | 0.452891 | 0.452978 |
| ctx_std_only | 0.452863 | 0.452958 |
| cosa_plus | 0.453148 | 0.453327 |

Interpretation: on the stable short-horizon ETTh1 setting, the original COSA
adapter remains slightly best. This supports the conditional narrative that
COSA+ is not expected to improve stable short-horizon settings.

### ETTh1 / DLinear / long-horizon ablation

| horizon | original MSE | vec_gate MSE | cosa_plus MSE |
|---:|---:|---:|---:|
| 192 | 0.501868 | 0.502159 | 0.502388 |
| 336 | 0.547112 | 0.546485 | 0.546639 |
| 720 | 0.673144 | 0.670848 | 0.670793 |

Interpretation: the vector-gate variants become useful at longer horizons. The
effect is negligible or negative at 192, small but positive at 336, and clearest
at 720, where COSA+ improves over original by about 0.35% relative MSE. This
matches the conditional hypothesis that step-wise correction capacity matters
more as forecast horizons grow.

### Black Swan / ETTh1 / DLinear / horizon=96 / magnitude=3.0

Metric shown below is `mse_after_50`.

| shift_type | no_tta | original | cosa_plus |
|---|---:|---:|---:|
| level | 2.794010 | 2.809918 | 2.801752 |
| variance | 1.998972 | 2.073147 | 2.064779 |
| trend | 0.392774 | 0.401253 | 0.398495 |
| spike | 0.369727 | 0.376349 | 0.374450 |

Interpretation: under the current short-horizon ETTh1 setup, adaptation does not
outperform no-TTA immediately after synthetic shifts. COSA+ is marginally better
than original COSA for all four shift types, but the effect is small. Stronger
claims require long-horizon checkpoints and/or Weather data.

### Black Swan / ETTh1 / DLinear / horizon=720 / magnitude=3.0

Metric shown below is `mse_after_50`.

| shift_type | no_tta | original | cosa_plus |
|---|---:|---:|---:|
| level | 1.5141 | 1.3900 | 1.3707 |
| variance | 1.4018 | 1.3126 | 1.3033 |
| trend | 1.0925 | 0.9965 | 0.9854 |
| spike | 1.0715 | 0.9798 | 0.9699 |

Interpretation: at the long horizon, both COSA variants improve the immediate
post-shift window over no-TTA, and COSA+ is consistently best among the three
methods across all four synthetic shifts. This is the strongest evidence so far
for the conditional robustness claim: COSA+ helps more when the prediction
horizon is long and shift effects propagate across many future steps.

### Weather / DLinear / clean-test main experiment

| horizon | original MSE | cosa_plus MSE | original MAE | cosa_plus MAE |
|---:|---:|---:|---:|---:|
| 96 | 0.190825 | 0.191509 | 0.238723 | 0.238533 |
| 192 | 0.232686 | 0.233276 | 0.275935 | 0.275405 |
| 336 | 0.280119 | 0.280689 | 0.310985 | 0.311692 |
| 720 | 0.344698 | 0.349111 | 0.358557 | 0.360986 |

Interpretation: on clean Weather test data, COSA+ does not improve MSE over the
original adapter. It slightly improves MAE at 96/192 but degrades at longer
horizons. This is a useful negative result: rich/volatility-aware context does
not automatically improve clean-test accuracy.

### Weather / DLinear / endpoint ablation

| horizon | original MSE | vec_gate MSE | rich_ctx MSE | ctx_std_only MSE | cosa_plus MSE |
|---:|---:|---:|---:|---:|---:|
| 96 | 0.190825 | 0.191642 | 0.190789 | 0.190763 | 0.191509 |
| 720 | 0.344698 | 0.349507 | 0.344772 | 0.344868 | 0.349111 |

Interpretation: context-only variants are essentially tied with original, while
vector-gate variants are worse on Weather clean-test MSE. This suggests that
the vector gate needs careful regularization/tuning on high-variance multivariate
data.

### Black Swan / Weather / DLinear / horizon=720 / magnitude=3.0

Metric shown below is `mse_after_50`.

| shift_type | no_tta | original | rich_ctx | cosa_plus |
|---|---:|---:|---:|---:|
| level | 0.5340 | 0.4633 | 0.4634 | 0.4665 |
| variance | 0.2244 | 0.2065 | 0.2062 | 0.2066 |
| trend | 0.2065 | 0.1614 | 0.1614 | 0.1626 |
| spike | 0.2021 | 0.1606 | 0.1606 | 0.1617 |

Interpretation: on Weather under abrupt shifts, adaptation is clearly useful
relative to no-TTA. Rich context is effectively tied with original and is
slightly best for variance shift, while COSA+ is marginally worse because the
vector gate hurts this dataset. This supports a nuanced conclusion: Mod B is
stable under volatility, but combining it with vector gating is not universally
beneficial.

## Invalidated earlier outputs

Earlier runs under `results/main_exp_dlinear_first/` failed before Weather data
was installed. Use `results/main_exp_weather_dlinear/` and
`results/ablation_weather_dlinear/` for Weather results.
