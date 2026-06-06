# Stress Testing COSA under Abrupt Test-Time Shifts in Time-Series Forecasting

## 1. Title
Stress Testing COSA under Abrupt Test-Time Shifts in Time-Series Forecasting

## 2. Abstract
This project studies whether COSA, a test-time adaptation method for time-series forecasting, remains effective when the test-time stream contains abrupt non-IID distribution shifts. We first reproduce a minimal COSA setting using DLinear on the ETTh1 dataset with prediction length 96. Then we extend the evaluation with an Exchange Rate black-swan stress test, where the first half of the test split keeps the original distribution and the second half receives a severity-scaled spike shift. The clean-setting results show that COSA slightly improves forecasting performance. However, under abrupt 5 sigma and 10 sigma shifts, COSA's overall improvement becomes negative and its window-level failure rate rises above 50%. These results suggest that COSA may be helpful under stable or mildly shifted test streams, but it is less reliable when the stream changes suddenly.

Group members: fill in names and student IDs here.

## 3. Introduction / Problem Definition
Time-series forecasting is commonly evaluated with fixed train, validation, and test splits. This setup is convenient, but it does not fully represent deployment scenarios where the test-time stream can change suddenly. Examples include financial shocks, abnormal energy demand, sensor drift, missing measurements, or other unexpected events. A forecasting method that performs well on a clean test split may behave differently when the input stream enters a new regime.

Test-time adaptation attempts to improve model behavior during inference by adapting to incoming test data. COSA, or Context-aware Output-Space Adapter, is designed for test-time adaptation in time-series forecasting. Instead of retraining the entire forecasting model, COSA adapts the output space using recent test-time context. This design is attractive because it is lightweight and can be applied after training.

The main research question of this project is:

Does COSA remain effective when the test-time data stream suddenly changes?

This question gives the project a direction beyond direct reproduction. We first verify that a minimal COSA experiment can run in our environment, and then we design a controlled abrupt-shift experiment to test whether COSA is robust under a black-swan-like test-time distribution shift.

## 4. Related Work / Background
COSA is a test-time adaptation method for time-series forecasting. Its key idea is to use an output-space adapter that adjusts predictions according to recent test-time context. This is different from methods that update all model parameters, and it is also different from standard forecasting evaluation where the trained model is fixed at test time.

The base model used in our experiments is DLinear. DLinear is a simple but strong time-series forecasting model that decomposes a time series and applies linear layers for forecasting. We use it because it is supported by the official COSA implementation and is computationally feasible for a course project.

This project also relates to distribution shift and robustness evaluation. In many real applications, the data distribution at test time may not match the training distribution. Instead of assuming a clean test split, our extension creates an abrupt synthetic shift in the test stream and studies how the no-adaptation baseline and COSA behave before, during, and after the shift boundary.

## 5. Methodology
We compare two methods:

- No test-time adaptation baseline: the trained DLinear model is evaluated directly on the test stream.
- COSA: the same trained DLinear model is evaluated with COSA test-time adaptation.

The project has two stages. The first stage is a minimal reproduction of the official COSA setting:

- Dataset: ETTh1
- Model: DLinear
- Prediction length: 96
- Methods: no-TTA baseline and COSA

The second stage is our black-swan stress test on the Exchange Rate dataset. We preserve temporal order and introduce an abrupt distribution shift at the midpoint of the test split:

```text
x'_t = x_t,                  if t < T_shift
x'_t = x_t + alpha * sigma,  if t >= T_shift
```

Here, `T_shift` is the midpoint of the test split, `alpha` is the shift severity, and `sigma` is the per-variable standard deviation of the test split. We evaluate three severities: 0 sigma, 5 sigma, and 10 sigma. The 0 sigma case is the clean control, while 5 sigma and 10 sigma represent stronger abrupt shifts.

Because forecasting uses sliding windows, we also divide test windows into three groups:

- Before shift: the prediction target is completely before the shift boundary.
- Transition: the prediction target crosses the shift boundary.
- After shift: the prediction target is completely after the shift boundary.

The transition group is especially important because it represents the moment when the model sees pre-shift context but must forecast values that partly enter the shifted regime.

## 6. Implementation Details
The official COSA implementation is kept under `external/COSA_ICLR2026/` as a Git submodule. We avoid directly modifying the official source unless necessary. Our project-specific scripts are stored under `experiments/` and `test_black_swan/`.

Main scripts:

- `experiments/run_00_check_env_data.sh`: checks Python packages, CUDA availability, and ETTh1 data.
- `experiments/run_01_train_etth1_dlinear_96.sh`: trains the minimal ETTh1 DLinear baseline.
- `experiments/run_02_cosa_etth1_dlinear_96.sh`: runs COSA on ETTh1.
- `experiments/test_run_black_swan_exchange_dlinear_96.sh`: runs the Exchange Rate abrupt-shift stress test.
- `test_black_swan/run_exchange_black_swan_test.py`: implements the test-time abrupt shift and writes summary results.

Environment:

- Python 3.10.20
- PyTorch 2.11.0+cu128
- NumPy 1.23.5
- SciPy 1.10.1
- GPU: NVIDIA GeForce RTX 5090

The main result tables are stored in `results/tables/experiment_summary.csv` and `results/test_black_swan/DLinear/exchange_rate_96/experiment_summary.csv`.

## 7. Experiments
We use the following metrics:

- MSE: mean squared error.
- MAE: mean absolute error.
- Improvement percentage: `(Baseline MSE - COSA MSE) / Baseline MSE * 100`.
- NAR percentage: the percentage of test windows where COSA has higher MSE than the no-TTA baseline.
- Segment MSE: before, transition, and after shift MSE.

### Experiment 1: Minimal Clean Reproduction on ETTh1

| Dataset | Model | Horizon | Shift | Baseline MSE | COSA MSE | Improvement |
| --- | --- | --- | --- | ---: | ---: | ---: |
| ETTh1 | DLinear | 96 | none | 0.4594808246 | 0.4527572393 | 1.46% |

This verifies that the official COSA pipeline can run in our environment. COSA gives a small improvement over the no-TTA baseline in the clean ETTh1 setting.

### Experiment 2: Exchange Rate Abrupt-Shift Stress Test

| Dataset | Model | Horizon | Shift | Baseline MSE | COSA MSE | Improvement | NAR |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| exchange_rate | DLinear | 96 | none, 0 sigma | 0.0827968419 | 0.0817446634 | 1.27% | 37.13% |
| exchange_rate | DLinear | 96 | abrupt spike, 5 sigma | 0.5509226918 | 0.5524317026 | -0.27% | 53.66% |
| exchange_rate | DLinear | 96 | abrupt spike, 10 sigma | 2.1627802849 | 2.1673173904 | -0.21% | 56.40% |

Transition-window analysis:

| Shift | Baseline transition MSE | COSA transition MSE | Improvement |
| --- | ---: | ---: | ---: |
| 0 sigma | 0.1459996402 | 0.1437337101 | 1.55% |
| 5 sigma | 6.6955876350 | 6.7212104797 | -0.38% |
| 10 sigma | 29.2314300537 | 29.2752418518 | -0.15% |

## 8. Results and Analysis
The clean experiments show that COSA can provide small positive improvements. On ETTh1, COSA improves MSE by 1.46%. On clean Exchange Rate, COSA improves MSE by 1.27%. These results are consistent with the idea that COSA can be useful when the test stream is stable enough for recent context to be informative.

The abrupt-shift results are different. Under the 5 sigma shift, COSA changes from a 1.27% clean improvement to a -0.27% overall degradation. Under the 10 sigma shift, COSA is also worse than the no-TTA baseline, with -0.21% improvement. The NAR percentage rises above 50% for both shifted settings, which means COSA has higher window-level MSE than the baseline on more than half of the test windows.

The transition-window results explain why the abrupt setting is difficult. When the prediction target crosses the shift boundary, the forecasting model receives context from the original regime but must predict values that partly belong to the shifted regime. The transition MSE increases from about 0.146 in the clean case to 6.696 under 5 sigma and 29.231 under 10 sigma. COSA does not reduce this transition error; it is slightly worse than the baseline in both shifted settings.

A possible explanation is that COSA's adapter relies on recent test-time context. This can help when the stream is smooth or mildly shifted, but near an abrupt boundary the recent context may be misleading. The adapter may adjust predictions based on patterns that no longer match the target regime. This does not mean COSA is ineffective in general. Instead, our results show a limitation: output-space adaptation may need additional shift detection, robust context filtering, or recovery mechanisms to handle sudden black-swan changes.

The project has several limitations. We only evaluate one base model, one prediction length, and one main black-swan dataset. The shift is synthetic and only tests an additive abrupt spike/level shock. Future work should test more shift types, such as variance shifts, trend changes, missing segments, and short local spikes. It would also be useful to compare additional adaptation methods or add a simple shift detector before applying COSA.

## 9. Conclusion
This project reproduces a minimal COSA experiment and extends it with an abrupt test-time distribution-shift evaluation. The results show that COSA gives small improvements in clean settings, but its benefit weakens or becomes negative under strong abrupt shifts. The transition-window analysis suggests that the most difficult part is the shift boundary, where the model must forecast into a new regime using old-regime context. Overall, the project supports the conclusion that test-time adaptation methods should be evaluated not only on clean test splits, but also under realistic non-IID stream changes.

## References
[1] BigBases. COSA_ICLR2026: Official implementation of COSA, Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting. GitHub repository. https://github.com/bigbases/COSA_ICLR2026

[2] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are Transformers Effective for Time Series Forecasting? Proceedings of the AAAI Conference on Artificial Intelligence, 2023. https://arxiv.org/abs/2205.13504

[3] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. NeurIPS, 2021. https://arxiv.org/abs/2106.13008

[4] Zhou Tian. ETDataset: Electricity Transformer Temperature benchmark datasets. GitHub repository. https://github.com/zhouhaoyi/ETDataset

[5] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks. SIGIR, 2018. https://arxiv.org/abs/1703.07015

## AI Usage Statement
We used AI tools to help understand project requirements, inspect and organize code, debug experiment scripts, and draft the report structure. All experimental design choices, implementation changes, numerical results, analysis, and final conclusions should be reviewed and verified by the project members before submission.

## Team Contribution Statement
Fill in each group member's contribution before submission.

Example format:

- Member A, student ID: environment setup, COSA reproduction experiments, result collection.
- Member B, student ID: abrupt-shift experiment design and implementation.
- Member C, student ID: report writing, slide preparation, result analysis.
