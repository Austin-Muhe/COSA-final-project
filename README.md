# Dynamic Regime Adapter for Zero-Shot Time-Series Adaptation

## Abstract

This project extends COSA/COSA+ for time-series forecasting under non-IID test-time distribution shifts. The original COSA-style adapter relies on online test-time optimization: it updates a static output-space gate by backpropagating on each test stream segment. That can improve MSE, but it is slower at inference time and does not explicitly reveal which hidden market or sensor regimes the model has discovered.

Our contribution is a **Dynamic Regime Adapter**: a lightweight hypernetwork that learns to map recent historical context windows into latent regime factors `z`, then generates a horizon-wise correction gate `g` in one forward pass. Instead of tuning a static gate with test-time backpropagation, we meta-train the adapter offline on simulated validation shifts. At test time, the adapter performs **zero-shot adaptation** with `torch.no_grad()`.

The key innovation is moving from static MSE-driven gate tuning to **unsupervised latent pattern discovery**. The `RegimeEncoder` compresses recent time-series context into low-dimensional learned factors, and the `GateGenerator` converts those factors into dynamic output corrections. The learned latent space can be visualized with t-SNE/PCA to show whether the model separates normal, level-shifted, variance-shifted, and spike-shifted regimes.

## Method Overview

The dynamic adapter receives a historical context window:

```text
context_window: (batch_size, context_length, feature_dim)
```

It produces:

```text
z: (batch_size, latent_dim)
g: (batch_size, horizon)
```

The robust forecast is computed as:

```python
g, z = dynamic_adapter(context_window)
correction = torch.tanh(g).unsqueeze(-1) * base_forecast
robust_forecast = base_forecast + correction
```

At test time this is executed under `torch.no_grad()`, so there is no online backpropagation cost.

## Repository Structure

```text
COSA-final-project/
  experiments/
    train_dynamic_regime_adapter.py      # Offline meta-training
    evaluate_dynamic_regime_adapter.py   # Zero-shot inference/evaluation
    visualize_latent_regimes.py          # t-SNE/PCA visualization of z
    run_cosa_plus.py                     # Existing COSA+ ablation runner

  external/COSA_ICLR2026/
    tta/cosa.py                          # DynamicRegimeAdapter implementation
    config.py                            # COSA config extensions
    data/                                # Dataset files
    checkpoints/                         # Frozen forecasting backbone checkpoints

  results/
    regime_adapter_mixed_loss/           # Meta-trained adapter checkpoints
    regime_adapter_eval_mixed_loss/      # Zero-shot evaluation metrics
    regime_adapter_latents_mixed_loss/   # Latent-regime plots and arrays

  test_black_swan/
    test_dynamic_regime_adapter.py       # Unit/smoke tests
```

## Environment

Use the existing conda environment:

```bash
conda activate cosa
cd /home/wwww/projects/COSA-final-project
```

Check the basic environment:

```bash
python -c "import torch, numpy, pandas, sklearn, scipy, matplotlib; print('packages ok'); print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Tested setup:

```text
Python 3.10.20
torch 2.11.0+cu128
GPU: NVIDIA GeForce RTX 5090
```

Do not install the original COSA `requirements.txt` directly, because it may pin an old PyTorch version incompatible with the current GPU stack.

## Data and Checkpoints

Expected data location:

```text
external/COSA_ICLR2026/data/<dataset>/<dataset>.csv
```

Expected frozen backbone checkpoint location:

```text
external/COSA_ICLR2026/checkpoints/<model>/<dataset>_<pred_len>/checkpoint_best.pth
```

Example used in the final experiments:

```text
Dataset: weather
Model: DLinear
Prediction horizon: 96
Data: external/COSA_ICLR2026/data/weather/weather.csv
Backbone checkpoint: external/COSA_ICLR2026/checkpoints/DLinear/weather_96/checkpoint_best.pth
```

## 1. Meta-Train the Dynamic Regime Adapter

The offline phase builds validation meta-training batches:

```text
(context_window, base_forecast, y_true)
```

The forecasting backbone is frozen. Only `DynamicRegimeAdapter` is updated. During training, validation context windows are exposed to simulated black-swan shifts, and the adapter learns to generate robust dynamic gates.

Run the mixed-loss meta-training used in the final experiment:

```bash
python experiments/train_dynamic_regime_adapter.py \
  --dataset weather \
  --model DLinear \
  --pred_len 96 \
  --batch_size 64 \
  --epochs 10 \
  --lr 0.001 \
  --latent_dim 16 \
  --hidden_dim 64 \
  --dropout 0.1 \
  --mse_weight 0.7 \
  --mae_weight 0.3 \
  --output_dir ./results/regime_adapter_mixed_loss
```

The training loss is:

```python
mse_loss = F.mse_loss(robust_forecast, y_true)
mae_loss = F.l1_loss(robust_forecast, y_true)
loss = mse_weight * mse_loss + mae_weight * mae_loss
```

This mixed loss reduces overreaction to spikes compared with pure MSE training.

Main output:

```text
results/regime_adapter_mixed_loss/DLinear/weather/96/dynamic_regime_adapter.pt
results/regime_adapter_mixed_loss/DLinear/weather/96/meta_training_metrics.json
```

Final mixed-loss validation result:

```text
base_val_mse:    0.481351
adapter_val_mse: 0.432192
```

## 2. Run Zero-Shot Test-Time Inference

The online phase evaluates the frozen backbone plus the meta-trained dynamic adapter. No test-time optimization is performed.

Run zero-shot inference with the trained adapter:

```bash
python experiments/evaluate_dynamic_regime_adapter.py \
  --dataset weather \
  --model DLinear \
  --pred_len 96 \
  --split test \
  --batch_size 64 \
  --dropout 0.0 \
  --regime_checkpoint ./results/regime_adapter_mixed_loss/DLinear/weather/96/dynamic_regime_adapter.pt \
  --output_dir ./results/regime_adapter_eval_mixed_loss \
  --save_latents
```

Run the untrained dynamic-adapter baseline:

```bash
python experiments/evaluate_dynamic_regime_adapter.py \
  --dataset weather \
  --model DLinear \
  --pred_len 96 \
  --split test \
  --batch_size 64 \
  --latent_dim 16 \
  --hidden_dim 64 \
  --dropout 0.0 \
  --output_dir ./results/regime_adapter_eval \
  --save_latents
```

Main output:

```text
results/regime_adapter_eval_mixed_loss/meta_trained/DLinear/weather/96/metrics_test.json
```

Final zero-shot result on `weather / DLinear / horizon=96`:

| Method | Test MSE | Test MAE | MSE Improve vs Base | Notes |
| --- | ---: | ---: | ---: | --- |
| Base frozen backbone | 0.195218 | 0.234493 | 0.00% | No adapter |
| DynamicRegimeAdapter untrained | 0.195219 | 0.234494 | -0.00% | No test-time backprop |
| DynamicRegimeAdapter meta-trained, mixed loss | 0.190373 | 0.240629 | 2.48% | Zero-shot adaptation |
| COSA original | 0.191272 | 0.237397 | 2.02% | Test-time backprop, 492 adaptation steps |
| COSA+ | 0.191641 | 0.237735 | 1.83% | Test-time backprop, 492 adaptation steps |

The dynamic adapter matches or exceeds COSA/COSA+ MSE while requiring no online gradient updates.

## 3. Generate Latent Regime t-SNE Visualization

To prove that the adapter discovers hidden regimes, run inference on validation contexts with four explicit regime types:

```text
Normal
Level
Variance
Spike
```

The script extracts the intermediate latent vector `z` from the `RegimeEncoder`, projects it to 2D with t-SNE or PCA, and colors each point by the known shift type.

Run t-SNE visualization:

```bash
python experiments/visualize_latent_regimes.py \
  --dataset weather \
  --model DLinear \
  --pred_len 96 \
  --regime_checkpoint ./results/regime_adapter_mixed_loss/DLinear/weather/96/dynamic_regime_adapter.pt \
  --projection tsne \
  --max_batches 30 \
  --max_points_per_regime 2000 \
  --output_dir ./results/regime_adapter_latents_mixed_loss
```

Run PCA instead of t-SNE:

```bash
python experiments/visualize_latent_regimes.py \
  --dataset weather \
  --model DLinear \
  --pred_len 96 \
  --regime_checkpoint ./results/regime_adapter_mixed_loss/DLinear/weather/96/dynamic_regime_adapter.pt \
  --projection pca \
  --output_dir ./results/regime_adapter_latents_mixed_loss
```

Main t-SNE output:

```text
results/regime_adapter_latents_mixed_loss/tsne/DLinear/weather/96/latent_regimes.png
```

The script also saves raw analysis artifacts:

```text
latent_z.npy
gate_g.npy
projection_2d.npy
projection_2d.csv
labels.npy
metadata.json
```

Final t-SNE run collected:

```text
Normal:   1920 points
Level:    1920 points
Variance: 1920 points
Spike:    1920 points
Total:    7680 points
```

If the latent plot shows separable clusters, it is evidence that the adapter is learning regime-specific representations rather than only fitting an output-space correction.

## 4. Compare Against COSA/COSA+

Run existing COSA+ ablations:

```bash
python experiments/run_cosa_plus.py \
  --dataset weather \
  --model DLinear \
  --pred_len 96 \
  --variant original \
  --batch_size 64 \
  --steps 3 \
  --output_dir ./results/ablation_dynamic_regime \
  --visible_devices 0
```

```bash
python experiments/run_cosa_plus.py \
  --dataset weather \
  --model DLinear \
  --pred_len 96 \
  --variant cosa_plus \
  --batch_size 64 \
  --steps 3 \
  --output_dir ./results/ablation_dynamic_regime \
  --visible_devices 0
```

## 5. Tests

Run the dynamic adapter tests:

```bash
python test_black_swan/test_dynamic_regime_adapter.py
```

Expected output:

```text
DynamicRegimeAdapter tests passed.
```

## Key Files

| File | Purpose |
| --- | --- |
| `external/COSA_ICLR2026/tta/cosa.py` | `DynamicRegimeAdapter`, mixed-loss training helper, no-backprop inference helper |
| `experiments/train_dynamic_regime_adapter.py` | Offline meta-training on validation black-swan shifts |
| `experiments/evaluate_dynamic_regime_adapter.py` | Zero-shot online inference and metric export |
| `experiments/visualize_latent_regimes.py` | t-SNE/PCA plot of latent regime factors `z` |
| `test_black_swan/test_dynamic_regime_adapter.py` | Shape, initialization, training, and no-grad inference tests |

## Notes for Future Work

- Tune the mixed loss or try Huber/SmoothL1 loss to improve MAE further.
- Add explicit gate magnitude regularization to prevent overcorrection.
- Visualize gate heatmaps alongside latent t-SNE clusters.
- Repeat the protocol on ETTh1 and Exchange Rate black-swan settings.
- Add quantitative cluster metrics such as silhouette score or Davies-Bouldin index.
