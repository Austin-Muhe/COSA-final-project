# Black Swan Experiment Task Brief
## Context

This is a follow-up task for the COSA course project. The first task brief
(`TASK_BRIEF.md`) already handles implementing `experiments/tta/cosa_plus.py`
and the ablation scripts. **Assume that work is done or in progress. Do not touch it.**

This brief covers one additional experiment module:
**Abrupt distribution shift (black swan) robustness analysis.**

---

## Research Question

> Does COSA remain effective when the test-time data stream suddenly changes?
> Does COSA+ (vector gate + rich context) recover faster than original COSA?

The paper's Appendix D / Figure 5 shows that COSA's gating value drops toward 0
when residual magnitude spikes — meaning the adapter already has a self-protection
mechanism. We want to stress-test this under controlled, severe shifts and compare
original COSA vs COSA+.

---

## What a "Black Swan" Shift Is

We take the **original, unmodified test set** and inject an artificial shift at a
chosen time step `T_shift`. Before `T_shift`, data is clean. After it, data is
perturbed. The model never knows the shift happened — it just keeps predicting.

Four shift types (all operate on the test split only, not train/val):

```python
# 1. Level shift — sudden jump in mean
x[t:] = x[t:] + c                          # c = k * std(x_train)

# 2. Variance shift — sudden change in scale
x[t:] = mean + alpha * (x[t:] - mean)      # alpha in {0.3, 2.0, 5.0}

# 3. Trend shift — gradual drift after T_shift
x[t + i] = x[t + i] + beta * i             # beta = small slope

# 4. Spike — single extreme value, then back to normal
x[T_shift] = x[T_shift] + k * std(x_train) # k in {3, 5, 10}
```

---

## Files to Create

### 1. `experiments/blackswan/shift_injector.py`

A single Python module with one function per shift type plus a dispatcher:

```python
import numpy as np

def inject_level_shift(data: np.ndarray, t_shift: int, magnitude: float) -> np.ndarray:
    """
    data: shape (T, n_vars) or (T,)
    t_shift: index where shift starts
    magnitude: how many train-std units to shift by
    Returns perturbed copy of data (original not modified).
    """
    out = data.copy()
    out[t_shift:] += magnitude
    return out


def inject_variance_shift(data: np.ndarray, t_shift: int, alpha: float) -> np.ndarray:
    """
    alpha > 1 → higher variance; alpha < 1 → compressed variance.
    """
    out = data.copy()
    mean = data[:t_shift].mean()
    out[t_shift:] = mean + alpha * (out[t_shift:] - mean)
    return out


def inject_trend_shift(data: np.ndarray, t_shift: int, slope: float) -> np.ndarray:
    """
    Adds a linear drift starting at t_shift.
    slope: value added per time step (e.g. 0.01 * std / step)
    """
    out = data.copy()
    n = len(data) - t_shift
    drift = np.arange(n) * slope
    if out.ndim == 2:
        out[t_shift:] += drift[:, None]
    else:
        out[t_shift:] += drift
    return out


def inject_spike(data: np.ndarray, t_shift: int, magnitude: float) -> np.ndarray:
    """
    Single extreme value at t_shift, data returns to normal after.
    magnitude: k * std
    """
    out = data.copy()
    out[t_shift] += magnitude
    return out


def inject_shift(data: np.ndarray, shift_type: str, t_shift: int,
                 magnitude: float, train_std: float) -> np.ndarray:
    """
    Dispatcher. magnitude is in units of train_std.
    shift_type: 'level' | 'variance' | 'trend' | 'spike'
    """
    m = magnitude * train_std
    if shift_type == 'level':
        return inject_level_shift(data, t_shift, m)
    elif shift_type == 'variance':
        # magnitude here is the alpha multiplier, not offset
        return inject_variance_shift(data, t_shift, alpha=magnitude)
    elif shift_type == 'trend':
        slope = magnitude * train_std / 100   # gentle slope
        return inject_trend_shift(data, t_shift, slope)
    elif shift_type == 'spike':
        return inject_spike(data, t_shift, m)
    else:
        raise ValueError(f"Unknown shift_type: {shift_type}")
```

### 2. `experiments/blackswan/run_blackswan.py`

Main experiment runner. For each combination of (shift_type, magnitude, method),
it runs inference on the perturbed test set and records:
- MSE over the full test set
- MSE in the 50 steps **before** T_shift (should be identical for all methods)
- MSE in the 50 steps **after** T_shift (this is where differences appear)
- Recovery time: how many steps until rolling MSE returns to within 10% of pre-shift level

```python
import argparse
import sys
import os
import numpy as np
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../external/COSA_ICLR2026'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from experiments.blackswan.shift_injector import inject_shift

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    default='ETTh1')
    p.add_argument('--model',      default='DLinear')
    p.add_argument('--pred_len',   type=int, default=96)
    p.add_argument('--variant',    default='original',
                   choices=['original', 'vec_gate', 'rich_ctx', 'cosa_plus', 'no_tta'])
    p.add_argument('--shift_type', default='level',
                   choices=['level', 'variance', 'trend', 'spike'])
    p.add_argument('--magnitude',  type=float, default=3.0)
    p.add_argument('--t_shift_frac', type=float, default=0.4,
                   help='Fraction of test set where shift is injected (default: 40%)')
    p.add_argument('--output_dir', default='./results/blackswan/')
    return p.parse_args()


def rolling_mse(errors: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute rolling MSE with given window size."""
    squared = errors ** 2
    result = np.convolve(squared, np.ones(window) / window, mode='valid')
    return result


def compute_recovery_steps(rolling: np.ndarray, t_shift: int, baseline_mse: float,
                            threshold: float = 0.1) -> int:
    """
    Returns number of steps after t_shift until rolling MSE is within
    threshold fraction of baseline_mse. Returns -1 if never recovers.
    """
    after = rolling[t_shift:]
    for i, v in enumerate(after):
        if abs(v - baseline_mse) / (baseline_mse + 1e-8) <= threshold:
            return i
    return -1


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data and model (use official helpers) ---
    # TODO: Replace the lines below with the actual data loading from
    #       external/COSA_ICLR2026's data_provider, same as in run_cosa_plus.py
    # test_data shape: (T, n_vars)
    # train_data is needed to compute train_std for shift magnitude calibration

    # train_data = load_train_split(args.dataset, ...)
    # test_data  = load_test_split(args.dataset, ...)
    # base_model = load_checkpoint(args.model, args.dataset, args.pred_len)

    # --- Compute train std for magnitude calibration ---
    # train_std = train_data.std()

    # --- Inject shift ---
    # t_shift = int(args.t_shift_frac * len(test_data))
    # perturbed_test = inject_shift(test_data, args.shift_type,
    #                               t_shift, args.magnitude, train_std)

    # --- Run inference loop ---
    # Instantiate adapter based on args.variant (same as run_cosa_plus.py)
    # Run prediction + adaptation on perturbed_test
    # Collect per-step prediction errors: errors shape (T,)

    # --- Compute metrics ---
    # mse_full   = (errors**2).mean()
    # mse_before = (errors[:t_shift]**2).mean()
    # mse_after  = (errors[t_shift:]**2).mean()
    # rolling    = rolling_mse(errors, window=20)
    # recovery   = compute_recovery_steps(rolling, t_shift, mse_before)

    # --- Save results ---
    # results = {
    #     'variant':    args.variant,
    #     'shift_type': args.shift_type,
    #     'magnitude':  args.magnitude,
    #     'mse_full':   float(mse_full),
    #     'mse_before': float(mse_before),
    #     'mse_after':  float(mse_after),
    #     'recovery_steps': recovery,
    #     'rolling_mse': rolling.tolist(),
    #     't_shift':    t_shift,
    # }
    # fname = f"{args.variant}_{args.shift_type}_m{args.magnitude}_{args.pred_len}.json"
    # with open(os.path.join(args.output_dir, fname), 'w') as f:
    #     json.dump(results, f, indent=2)
    # print(f"MSE full={mse_full:.4f}  before={mse_before:.4f}  after={mse_after:.4f}  recovery={recovery} steps")

if __name__ == '__main__':
    main()
```

**Your job:** Fill in the TODO sections by copying the data loading and inference
loop pattern from `experiments/run_cosa_plus.py` (already written in the first task brief).
The adapter instantiation is identical — just pass the perturbed test data instead of clean data.

### 3. `experiments/run_06_blackswan.sh`

```bash
#!/bin/bash
# run_06_blackswan.sh
# Black swan robustness experiment.
# Runs all combinations of shift type, magnitude, and method on ETTh1/DLinear/96.

cd /home/wwww/projects/COSA-final-project
conda activate cosa

VARIANTS=("no_tta" "original" "cosa_plus")
SHIFT_TYPES=("level" "variance" "trend" "spike")
MAGNITUDES=(1.0 3.0 5.0)   # in units of train_std (except variance: alpha multiplier)

for VARIANT in "${VARIANTS[@]}"; do
  for SHIFT_TYPE in "${SHIFT_TYPES[@]}"; do
    for MAG in "${MAGNITUDES[@]}"; do
      echo "=== variant=${VARIANT}  shift=${SHIFT_TYPE}  mag=${MAG} ==="
      python experiments/blackswan/run_blackswan.py \
        --dataset ETTh1 \
        --model DLinear \
        --pred_len 96 \
        --variant ${VARIANT} \
        --shift_type ${SHIFT_TYPE} \
        --magnitude ${MAG} \
        --output_dir ./results/blackswan/ \
        >> ./results/blackswan/run_log.txt 2>&1
    done
  done
done

echo "Done. Results in results/blackswan/"
```

### 4. `notebooks/blackswan_analysis.ipynb`

Notebook that reads all `results/blackswan/*.json` and produces two figures:

**Figure A — Recovery curve (main visual for the report):**
- One subplot per shift type (2×2 grid)
- x-axis: time steps in test set
- y-axis: rolling MSE (window=20)
- Three lines: no_tta (gray), original COSA (blue), COSA+ (orange)
- Vertical dashed line at `t_shift`
- This shows visually which method recovers fastest after the shift

```python
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt

results = []
for fpath in glob.glob('results/blackswan/*.json'):
    with open(fpath) as f:
        results.append(json.load(f))

# Filter: mag=3.0 only for the main figure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
shift_types = ['level', 'variance', 'trend', 'spike']
colors = {'no_tta': 'gray', 'original': '#378ADD', 'cosa_plus': '#E85D24'}

for ax, stype in zip(axes.flat, shift_types):
    for variant in ['no_tta', 'original', 'cosa_plus']:
        match = [r for r in results
                 if r['shift_type'] == stype
                 and r['variant'] == variant
                 and r['magnitude'] == 3.0]
        if not match:
            continue
        r = match[0]
        rolling = np.array(r['rolling_mse'])
        ax.plot(rolling, label=variant, color=colors[variant], linewidth=1.5)
        ax.axvline(r['t_shift'], color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title(stype)
    ax.set_xlabel('Test step')
    ax.set_ylabel('Rolling MSE (w=20)')
    ax.legend(fontsize=9)

plt.suptitle('COSA vs COSA+ under abrupt distribution shifts (ETTh1, DLinear, L=96, mag=3σ)')
plt.tight_layout()
plt.savefig('results/blackswan/recovery_curves.png', dpi=150)
plt.show()
```

**Figure B — Recovery steps bar chart (for ablation table):**
- x-axis: shift type
- y-axis: recovery_steps value
- Grouped bars by variant
- Lower is better

```python
# Summary table: recovery steps
import pandas as pd

rows = []
for r in results:
    if r['magnitude'] == 3.0:
        rows.append({
            'shift_type': r['shift_type'],
            'variant':    r['variant'],
            'mse_after':  r['mse_after'],
            'recovery':   r['recovery_steps'],
        })

df = pd.DataFrame(rows)
pivot = df.pivot_table(index='shift_type', columns='variant',
                       values='recovery_steps', aggfunc='mean')
print(pivot.to_string())

# Bar chart
pivot.plot(kind='bar', figsize=(8, 4), color=['gray', '#378ADD', '#E85D24'])
plt.ylabel('Recovery steps (lower = better)')
plt.title('Recovery speed after abrupt shift')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('results/blackswan/recovery_steps.png', dpi=150)
plt.show()
```

---

## Integration With the Main Experiment

This black swan module is **standalone** — it reuses the same adapter code from
`experiments/tta/cosa_plus.py` but passes perturbed data. No changes to any
previously written files are needed.

In the final report, this becomes **Section 8.3** (a subsection of Results and Analysis):

> "To further evaluate robustness under extreme conditions, we inject four types
> of abrupt distribution shifts into the test set and measure recovery behavior.
> As shown in Figure X, COSA+ recovers significantly faster under variance shifts
> and spikes, consistent with our hypothesis that volatility-aware context (Mod B)
> provides earlier warning of distribution change."

---

## Step-by-Step for Copilot

1. Create `experiments/blackswan/__init__.py` (empty).
2. Create `experiments/blackswan/shift_injector.py` — copy the code above exactly.
3. Create `experiments/blackswan/run_blackswan.py` — fill in the TODO sections
   by reusing the data loading and inference loop from `experiments/run_cosa_plus.py`.
   The only difference is: **apply `inject_shift()` to the test data before running**.
4. Create `experiments/run_06_blackswan.sh` as shown.
5. Create `notebooks/blackswan_analysis.ipynb` with both figures.
6. Test with one run:
   ```bash
   python experiments/blackswan/run_blackswan.py \
     --variant original --shift_type level --magnitude 3.0
   ```
   Expected output: `MSE full=X.XXXX  before=X.XXXX  after=X.XXXX  recovery=NN steps`
   The `mse_after` should be noticeably higher than `mse_before` for magnitude=3.0.

---

## What Not to Do

- Do not modify `external/COSA_ICLR2026/` for any reason.
- Do not re-implement the COSA adapter — import it from `experiments/tta/cosa_plus.py`.
- Do not apply shifts to the train or validation split — test split only.
- Do not use stochastic shifts (random noise injected every step) — the shifts
  defined here are **deterministic** given (shift_type, magnitude, t_shift).

---

*Black swan module for CPS3830 Final Project — COSA+ study.*
