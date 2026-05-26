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

Run from the project root:

    cd /home/wwww/projects/COSA-final-project
