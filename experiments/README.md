# Experiments

This folder contains our custom experiment scripts.

Do not directly run the official full scripts unless necessary:

    external/COSA_ICLR2026/scripts/train.sh
    external/COSA_ICLR2026/scripts/cosa.sh

Reason:

The official scripts run many combinations of models, datasets, and prediction lengths. They are too large for our first test.

Current minimal experiment:

    Dataset: ETTh1
    Model: DLinear
    Prediction length: 96

Recommended order:

    bash experiments/run_00_check_env_data.sh
    bash experiments/run_01_train_etth1_dlinear_96.sh
    bash experiments/run_02_cosa_etth1_dlinear_96.sh

Black-swan stress experiment:

    Dataset: Exchange Rate
    Model: DLinear
    Prediction length: 96
    Shift: abrupt spike corruption on the second half of the test split
    Severities: 0 sigma, 5 sigma, 10 sigma

Run:

    bash experiments/test_run_black_swan_exchange_dlinear_96.sh

Latest overall results:

| Dataset | Model | Horizon | Shift | Baseline MSE | COSA MSE | Improvement | NAR |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| exchange_rate | DLinear | 96 | none, 0 sigma | 0.0827968419 | 0.0817446634 | 1.27% | 37.13% |
| exchange_rate | DLinear | 96 | abrupt spike, 5 sigma | 0.5509226918 | 0.5524317026 | -0.27% | 53.66% |
| exchange_rate | DLinear | 96 | abrupt spike, 10 sigma | 2.1627802849 | 2.1673173904 | -0.21% | 56.40% |

Latest transition-window results:

| Shift | Baseline transition MSE | COSA transition MSE | Improvement |
| --- | ---: | ---: | ---: |
| 0 sigma | 0.1459996402 | 0.1437337101 | 1.55% |
| 5 sigma | 6.6955876350 | 6.7212104797 | -0.38% |
| 10 sigma | 29.2314300537 | 29.2752418518 | -0.15% |

Summary file:

    results/test_black_swan/DLinear/exchange_rate_96/experiment_summary.csv

Run from the project root:

    cd /home/wwww/projects/COSA-final-project
