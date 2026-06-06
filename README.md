# COSA Final Project

## Project Goal

This project is based on the official implementation of COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting.

Official COSA repository: https://github.com/bigbases/COSA_ICLR2026

Our goal is to reproduce COSA on a selected time-series forecasting dataset and evaluate its behavior under non-IID abrupt test-time distribution shifts.

Main research question:

Does COSA remain effective when the test-time data stream suddenly changes?

## Project Direction

We first reproduce the original COSA setting on a selected benchmark dataset. Then we construct non-IID test-time streams by injecting abrupt distribution shifts into the test set.

Possible shift types:

- Level shift
- Variance shift
- Trend shift
- Spike or short-term shock
- Missing segment or sensor dropout

Methods to compare:

- No test-time adaptation baseline
- Original COSA
- Possible COSA variants or simple correction baselines

## Repository Structure

    COSA-final-project/
      README.md
      .gitignore
      .gitmodules

      external/
        COSA_ICLR2026/          Official COSA implementation as a Git submodule

      experiments/              Our custom experiment scripts and notes
      results/                  Our result tables and figures
      notebooks/                Our analysis notebooks
      report/                   Proposal, final report, and presentation materials

## Important Distinction

The following folder contains the official COSA code:

    external/COSA_ICLR2026/

We should avoid directly modifying the official code unless necessary.

Our own project work should mainly go into:

    experiments/
    results/
    notebooks/
    report/
    README.md

If we need to modify an official script, we should copy it into `experiments/` first and then edit our own copy.

Example:

    external/COSA_ICLR2026/scripts/train.sh       Official script
    experiments/train_etth1_96.sh                 Our modified experiment script

## Server Information

Project root on server:

    /home/wwww/projects/COSA-final-project

Official COSA code path:

    /home/wwww/projects/COSA-final-project/external/COSA_ICLR2026

## Conda Environment

Use the `cosa` environment:

    conda activate cosa

Current tested setup:

    Python 3.10.20
    torch 2.11.0+cu128
    numpy 1.23.5
    scipy 1.10.1
    GPU: NVIDIA GeForce RTX 5090

Check basic packages:

    python -c "import torch, numpy, pandas, sklearn, scipy, matplotlib; print('basic packages ok'); print('torch:', torch.__version__); print('numpy:', numpy.__version__); print('scipy:', scipy.__version__)"

Check GPU:

    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

## Important Warning

Do not run the original command below inside `external/COSA_ICLR2026/`:

    pip install -r requirements.txt

Reason:

The original `requirements.txt` contains an old PyTorch requirement such as `torch==1.7.1`, which is incompatible with the current RTX 5090 environment.

The working PyTorch version is:

    torch 2.11.0+cu128

## Basic Usage for Group Members

Login to the server:

    ssh wwww@10.33.104.26

Enter the project:

    conda activate cosa
    cd /home/wwww/projects/COSA-final-project
    git pull

Enter the official COSA code:

    cd external/COSA_ICLR2026

Check GPU before running experiments:

    nvidia-smi

Check Python and GPU environment:

    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

## Dataset Location

The official COSA README asks us to place datasets under:

    external/COSA_ICLR2026/datasets/

For the minimal reproduction, we plan to start with:

    ETTh1.csv

Expected path:

    /home/wwww/projects/COSA-final-project/external/COSA_ICLR2026/datasets/ETTh1.csv

## Minimal Reproduction Experiment

Completed minimal setting:

- Dataset: ETTh1
- Model: DLinear
- Prediction length: 96
- Methods:
  - No-TTA baseline
  - COSA

Result:

- Baseline MSE: 0.4594808246
- COSA MSE: 0.4527572393
- Improvement: 1.46%

This verifies that the official COSA pipeline can run in our server environment.

## Non-IID Abrupt Shift Experiments

We preserve the temporal order of the test set and inject an abrupt distribution shift into the test-time stream. This is the main extension beyond direct reproduction and is used to study whether COSA remains effective when the data stream suddenly changes.

Current implemented setting:

- Dataset: Exchange Rate
- Model: DLinear
- Prediction length: 96
- Shift type: abrupt spike/level shock on the second half of the test split
- Severities: 0 sigma, 5 sigma, 10 sigma

Implemented shift:

    x_t' = x_t,                       if t < T_shift
    x_t' = x_t + alpha * sigma,       if t >= T_shift

where `T_shift` is the midpoint of the test split, `alpha` is the severity, and `sigma` is the per-variable test-set standard deviation.

Results:

| Dataset | Model | Horizon | Shift | Baseline MSE | COSA MSE | Improvement | NAR |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| exchange_rate | DLinear | 96 | none, 0 sigma | 0.0827968419 | 0.0817446634 | 1.27% | 37.13% |
| exchange_rate | DLinear | 96 | abrupt spike, 5 sigma | 0.5509226918 | 0.5524317026 | -0.27% | 53.66% |
| exchange_rate | DLinear | 96 | abrupt spike, 10 sigma | 2.1627802849 | 2.1673173904 | -0.21% | 56.40% |

Segment analysis:

| Shift | Baseline before | COSA before | Baseline transition | COSA transition | Baseline after | COSA after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 sigma | 0.0692245141 | 0.0691839233 | 0.1459996402 | 0.1437337101 | 0.0860871598 | 0.0843231827 |
| 5 sigma | 0.0692245141 | 0.0690561831 | 6.6955876350 | 6.7212104797 | 0.1471323967 | 0.1468728036 |
| 10 sigma | 0.0692245141 | 0.0690915585 | 29.2314300537 | 29.2752418518 | 0.3594304919 | 0.3627609611 |

The transition segment contains forecasting windows whose prediction targets cross the abrupt shift boundary. This segment shows the largest error increase, which matches the black-swan setting: the model sees pre-shift context but must forecast values that partly enter the shifted regime.

Initial interpretation:

COSA gives a small improvement in the clean Exchange Rate setting. Under abrupt 5 sigma and 10 sigma shifts, the overall improvement becomes negative and NAR rises above 50%, meaning COSA is worse than the no-TTA baseline on more than half of the test windows. This suggests that COSA may be less reliable when the test-time stream contains strong abrupt shifts.

Evaluation metrics:

- MSE
- MAE
- Improvement percentage over the no-TTA baseline
- NAR percentage, the percentage of test windows where COSA has higher MSE than the baseline
- Before/transition/after shift MSE based on whether each prediction target is before, crossing, or after the shift boundary

## Current Status

Completed:

- GitHub repository created
- COSA official code added as a submodule
- Server clone completed
- Conda environment `cosa` created
- GPU environment tested successfully
- Basic Python packages tested successfully
- ETTh1 DLinear 96 clean baseline training completed
- ETTh1 DLinear 96 COSA test-time adaptation completed
- Exchange Rate DLinear 96 spike-shift black-swan experiment completed
- Main result tables generated

Not completed yet:

- Result analysis and final report
- Presentation slides
- AI usage statement and team contribution statement for the report

## Team Workflow

Before editing code on the server:

    cd /home/wwww/projects/COSA-final-project
    git pull

After editing project files:

    git add .
    git commit -m "Describe the change"
    git push

Do not commit large files such as datasets, checkpoints, logs, or model weights.