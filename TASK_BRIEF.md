# COSA+ Implementation Task Brief
## Background

This is a course project based on the ICLR 2026 paper:
**"COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting"**
(Im & Kwon, 2026). Official code: https://github.com/bigbases/COSA_ICLR2026

The official repo is already included as a Git submodule at:
```
external/COSA_ICLR2026/
```
**Do not modify files inside `external/`. Copy what you need into `experiments/` first.**

---

## What COSA Does (Original)

COSA is a plug-and-play adapter that corrects the predictions of a *frozen* base forecasting model
at test time. The math is as follows:

| Symbol | Meaning |
|--------|---------|
| `L` | prediction horizon length (e.g. 96, 192, 336, 720) |
| `K` | context length (default: 10) |
| `B` | batch size for adaptation (default: 48) |
| `Y_base` | base model prediction, shape `(L,)` |
| `C_t` | context vector, shape `(K,)` — stack of recent K batch-mean ground truths |
| `X_adapter` | adapter input = concat([Y_base, C_t]), shape `(L+K,)` |
| `W, b` | linear layer weights, `W` ∈ `R^(L × (L+K))`, `b` ∈ `R^L` |
| `g` | gate parameter, scalar `R` (original) |
| `H` | residual = `W @ X_adapter + b`, shape `(L,)` |
| `Y_hat` | corrected output = `Y_base + tanh(g) * H` |

**Streaming adaptation protocol (leakage-free):**
1. Predict with frozen base model → get `Y_base`
2. Concat with context → feed into adapter → get `Y_hat`
3. Wait for ground truth `Y_true` to be revealed
4. Collect B (prediction, ground-truth) pairs
5. Minimise MSE loss + L2 weight decay over `S` gradient steps (default S=3)
6. Repeat

**Loss:**
```
L = sum_i ||Y_hat_{t-i} - Y_true_{t-i}||^2  +  lambda*(||W||_F^2 + ||b||^2 + ||g||^2)
```

**Learning rate schedule (CALR):** Cosine annealing within each batch's S steps, with online
adjustment: cut lr by 50% on loss spike, multiply by 1.1 on stable decrease.

---

## What We Are Adding (COSA+)

We implement two modifications that the original paper's Appendix H identifies as directions for
future work / alternative designs:

### Modification A — Vector Gate (per-step gating)

**Original:** `g` is a single scalar → `tanh(g)` applies the same correction strength to all L
output steps.

**Proposed:** `g` becomes a vector of length L → each forecast step has its own correction
strength:
```python
# Original
self.g = nn.Parameter(torch.zeros(1))          # scalar
Y_hat = Y_base + tanh(g) * H                   # broadcast over L

# Modified (COSA-VecGate)
self.g = nn.Parameter(torch.zeros(pred_len))   # vector R^L
Y_hat = Y_base + torch.tanh(self.g) * H        # element-wise ⊙
```

**Research question:** Does per-step gating help more at longer horizons (720) than shorter
ones (96)? The paper's Figure 4 already shows long-horizon gains are disproportionately large —
this motivates the hypothesis that different forecast steps need different correction intensities.

**Note:** The paper's Appendix H states they explored vector gating and found the original is
"most consistent on average". Our goal is NOT to beat the original globally, but to characterise
*when* vector gating helps (expected: long horizon, high-variance datasets).

### Modification B — Rich Context (mean + std)

**Original:** Context vector = stack of K batch-wise *means* of recent ground truth → shape `(K,)`.
This captures level/scale drift.

**Proposed:** Context vector = stack of K batch-wise *means* AND *standard deviations* →
shape `(2K,)`. This adds volatility information:
```python
# Original
mu_k   = batch_gt.mean(dim=0)                       # (L,) or scalar
C_t    = stack([mu_{t-K}, ..., mu_{t-1}])            # (K,)

# Modified (COSA-RichCtx)
mu_k   = batch_gt.mean(dim=0)
std_k  = batch_gt.std(dim=0, unbiased=False)
C_t    = stack([mu_{t-K},...,mu_{t-1}, std_{t-K},...,std_{t-1}])  # (2K,)
```

Because the adapter input grows from `(L+K,)` to `(L+2K,)`, the linear layer W changes shape:
```
Original:  W ∈ R^(L × (L+K))
Modified:  W ∈ R^(L × (L+2K))
```
Everything else (loss, CALR, gating) stays the same.

**Research question:** Does volatility information in the context help datasets with heteroscedastic
shifts (e.g. Weather) more than stable datasets (e.g. ETTh1)?

**Note:** The paper's Appendix G.1 compares *single* aggregation functions (mean, median, WA)
but never *combines* multiple statistics. This is a gap we fill.

### COSA+ = Modification A + B combined

Both changes can be applied simultaneously:
- `g ∈ R^L` (vector gate)
- `C_t ∈ R^{2K}` (rich context)
- `W ∈ R^(L × (L+2K))`
- `Y_hat = Y_base + tanh(g) ⊙ H` (element-wise)

---

## Files to Create

All new files go in `experiments/`. Do NOT touch `external/`.

### 1. `experiments/tta/cosa_plus.py`

A self-contained Python file implementing COSA+. It should:
- Read the original `external/COSA_ICLR2026/tta/` implementation for reference
- Subclass or replicate the original COSA adapter class
- Accept a `variant` argument: `"original"`, `"vec_gate"`, `"rich_ctx"`, `"cosa_plus"`
- All variants share the same training loop / CALR / loss

Skeleton (fill in details by reading the original):
```python
import torch
import torch.nn as nn
import copy

class COSAPlus(nn.Module):
    """
    COSA+ adapter with optional vector gating and/or rich context.

    Args:
        pred_len   (int):  L — forecast horizon
        ctx_len    (int):  K — context length (default 10)
        vec_gate   (bool): if True, use per-step gate g ∈ R^L (Mod A)
        rich_ctx   (bool): if True, concatenate mean+std in context (Mod B)
    """

    def __init__(self, pred_len: int, ctx_len: int = 10,
                 vec_gate: bool = False, rich_ctx: bool = False):
        super().__init__()
        self.pred_len = pred_len
        self.ctx_len  = ctx_len
        self.vec_gate = vec_gate
        self.rich_ctx = rich_ctx

        ctx_dim = 2 * ctx_len if rich_ctx else ctx_len
        adapter_input_dim = pred_len + ctx_dim

        # Linear correction layer
        self.linear = nn.Linear(adapter_input_dim, pred_len, bias=True)

        # Gate: scalar (original) or vector (Mod A)
        gate_dim = pred_len if vec_gate else 1
        self.g = nn.Parameter(torch.zeros(gate_dim))

        # Xavier init for linear weight, bias and gate start at 0
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

        # Context buffer: stores recent batch statistics
        # Each entry is a tuple (mu, std) of shape (pred_len,) or scalar
        self._ctx_buffer: list = []

    def build_context(self, gt_buffer: list) -> torch.Tensor:
        """
        Build context vector from a list of recent ground-truth batches.
        gt_buffer: list of tensors, each shape (B, L) or (L,)
        Returns C_t of shape (ctx_dim,)
        """
        stats = []
        for batch in gt_buffer[-self.ctx_len:]:
            mu = batch.mean()   # scalar for univariate
            if self.rich_ctx:
                std = batch.std(unbiased=False)
                stats.append(torch.stack([mu, std]))
            else:
                stats.append(mu.unsqueeze(0))

        # Pad with zeros if buffer not full yet
        ctx_step_dim = 2 if self.rich_ctx else 1
        while len(stats) < self.ctx_len:
            stats.insert(0, torch.zeros(ctx_step_dim, device=self.g.device))

        C_t = torch.cat(stats, dim=0)  # (ctx_dim,)
        return C_t

    def forward(self, y_base: torch.Tensor, C_t: torch.Tensor) -> torch.Tensor:
        """
        y_base: (L,)  — frozen base model prediction
        C_t:    (ctx_dim,) — context vector
        Returns y_hat: (L,)
        """
        x_adapter = torch.cat([y_base, C_t], dim=-1)   # (L+ctx_dim,)
        H = self.linear(x_adapter)                      # (L,)
        alpha = torch.tanh(self.g)                      # (1,) or (L,)
        y_hat = y_base + alpha * H                      # broadcast or element-wise
        return y_hat
```

### 2. `experiments/tta/calr.py`

Copy the CALR (cosine adaptive learning rate) scheduler from the original `tta/` folder. If the
original uses a standalone class, copy it as-is. If it's inline in the training loop, extract it into a
`CALR` class with `step(loss)` → returns updated lr.

### 3. `experiments/run_03_baseline_expand.sh`

Extend the existing `run_02` to all 4 horizons for ETTh1 + DLinear (variant="original").

```bash
#!/bin/bash
# run_03_baseline_expand.sh
# Extend baseline to 4 horizons. Records COSA-F numbers for comparison.

cd /home/wwww/projects/COSA-final-project   # adjust path

PRED_LENS=(96 192 336 720)

for PRED_LEN in "${PRED_LENS[@]}"; do
    echo "=== Running COSA-F  dataset=ETTh1  model=DLinear  horizon=${PRED_LEN} ==="
    python external/COSA_ICLR2026/main.py \
        DATA.NAME ETTh1 \
        DATA.PRED_LEN ${PRED_LEN} \
        MODEL.NAME DLinear \
        MODEL.pred_len ${PRED_LEN} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR "./checkpoints/DLinear/ETTh1_${PRED_LEN}/" \
        TTA.ENABLE True \
        TTA.SIMPLE.BATCH_SIZE 48 \
        TTA.SIMPLE.STEPS 3 \
        TTA.SIMPLE.BUFFER_CONTEXT_SIZE 10 \
        TTA.SIMPLE.FAST_ADAPTATION True \
        TTA.SIMPLE.ADAPTIVE_LR True \
        TTA.SIMPLE.PER_BATCH_LR_RESET True \
        TTA.SIMPLE.PAAS False \
        RESULT_DIR "./results/baseline/" \
        > "./results/baseline/DLinear_ETTh1_${PRED_LEN}.txt" 2>&1
done

echo "Done. Check results/baseline/"
```

### 4. `experiments/run_04_ablation.sh`

Runs the 5-variant ablation study on ETTh1 + DLinear × 4 horizons.

The script should call a wrapper that imports `COSAPlus` from `experiments/tta/cosa_plus.py`,
replaces the original adapter in the inference loop, and records MSE for each variant.

```bash
#!/bin/bash
# run_04_ablation.sh
VARIANTS=("original" "vec_gate" "rich_ctx" "ctx_std_only" "cosa_plus")
PRED_LENS=(96 192 336 720)

for VARIANT in "${VARIANTS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
        echo "=== variant=${VARIANT}  horizon=${PRED_LEN} ==="
        python experiments/run_cosa_plus.py \
            --dataset ETTh1 \
            --model DLinear \
            --pred_len ${PRED_LEN} \
            --variant ${VARIANT} \
            --output_dir ./results/ablation/ \
            > ./results/ablation/${VARIANT}_ETTh1_DLinear_${PRED_LEN}.txt 2>&1
    done
done
```

### 5. `experiments/run_cosa_plus.py`

The Python wrapper that:
1. Loads base model checkpoint from `external/COSA_ICLR2026/checkpoints/`
2. Loads test dataset
3. Instantiates `COSAPlus` with the right flags
4. Runs inference + adaptation loop
5. Prints MSE and MAE at the end

This file should **import helpers from the external submodule** (dataset loader, base model loader)
to avoid code duplication. Example:
```python
import sys
sys.path.insert(0, 'external/COSA_ICLR2026')

from datasets.data_factory import data_provider    # use official loader
from models import DLinear, iTransformer            # use official models
from experiments.tta.cosa_plus import COSAPlus
```

### 6. `experiments/run_05_main_exp.sh`

Main experiment: 2 datasets × 2 models × 4 horizons × {COSA-F, COSA+} = 32 runs.

```bash
DATASETS=("ETTh1" "weather")
MODELS=("DLinear" "iTransformer")
VARIANTS=("original" "cosa_plus")
PRED_LENS=(96 192 336 720)
```

### 7. `notebooks/analysis.ipynb`

Jupyter notebook that:
1. Reads all `results/ablation/*.txt` and `results/main_exp/*.txt`
2. Parses MSE values (grep/regex)
3. Creates two figures:
   - **Figure 1 (ablation bar chart):** x=variant, y=MSE, faceted by horizon. Shows contribution of each modification.
   - **Figure 2 (gate heatmap):** For COSA+ on ETTh1, load the saved gate parameter `tanh(g)` (shape L=720) and plot it as a bar chart or heatmap — shows which future steps get more/less correction.
4. Computes % improvement of COSA+ over COSA-F per horizon.

---

## Key Implementation Notes

1. **Dimension consistency:** When `rich_ctx=True`, the context dimension doubles. Make sure
   `nn.Linear(adapter_input_dim, pred_len)` uses `adapter_input_dim = pred_len + ctx_dim`
   where `ctx_dim = 2*ctx_len` (not `ctx_len`).

2. **Gate initialisation:** Both scalar and vector gates should be initialised to 0, so `tanh(0)=0`
   and the adapter starts as an identity (no correction). This is identical to the original.

3. **Saving gate values for visualisation:** After adaptation on the test set, save
   `torch.tanh(adapter.g).detach().cpu().numpy()` to a file so the notebook can plot it.

4. **Numerical stability for std:** Use `unbiased=False` (population std) to avoid NaN when
   batch size is 1. Also add a small epsilon when needed: `std = std + 1e-8`.

5. **`ctx_std_only` variant:** This is for the ablation. Same as `rich_ctx=True` but context
   is `[std_{t-K}, ..., std_{t-1}]` only (no means). Requires a `ctx_mode` flag in `COSAPlus`.

6. **CALR compatibility:** The original CALR adjusts lr based on loss trends. It should work
   identically for all variants — no changes needed to the scheduler itself.

7. **Check the original source carefully:** Before writing code, read:
   - `external/COSA_ICLR2026/tta/` — find the main adapter class (likely `SimpleTTA` or similar)
   - `external/COSA_ICLR2026/predictor.py` — find the inference loop
   - `external/COSA_ICLR2026/config.py` — find all config keys

   The exact class names may differ. Adapt accordingly.

---

## Step-by-Step Instructions for Copilot

1. **First:** Open `external/COSA_ICLR2026/tta/` and read the existing adapter implementation.
   Identify: (a) where `self.g` is defined, (b) where context is built, (c) the forward pass.

2. **Create** `experiments/tta/__init__.py` (empty).

3. **Create** `experiments/tta/cosa_plus.py` following the skeleton above, filling in the
   details to match the original's patterns (same loss function, same CALR logic, same
   initialisation). The class should be a drop-in replacement for the original adapter.

4. **Create** `experiments/run_cosa_plus.py` — a thin wrapper that runs one experiment
   given `--dataset`, `--model`, `--pred_len`, `--variant`, and prints `MSE: X.XXXX` at the end.

5. **Create** the shell scripts `run_03`, `run_04`, `run_05` as shown above.

6. **Create** `notebooks/analysis.ipynb` with the two figures described.

7. **Test** by running:
   ```bash
   python experiments/run_cosa_plus.py \
       --dataset ETTh1 --model DLinear --pred_len 96 --variant original
   ```
   Verify the output MSE is close to the paper's Table 2 value for ETTh1/DLinear/96
   (approximately 0.4363–0.4695 depending on whether TTA is on).

---

## What Success Looks Like

After all scripts run successfully, `results/ablation/` should contain MSE values for 5 variants
× 4 horizons = 20 entries. The notebook should produce:

- **Ablation table** (for report Table 2): shows that `COSA-VecGate` and `COSA-RichCtx`
  each contribute, with `COSA+` being best (or at least competitive) — *especially at longer
  horizons*.
- **Gate visualisation** (for report Figure): shows that learned gate weights are non-uniform
  across the L output steps, with a visible pattern (e.g. larger weights at steps 300–720).

Even if COSA+ does not outperform COSA-F on average (the paper says original is most robust),
the analysis of *when* it helps is the actual scientific contribution.

---

*Generated for CPS3830 Final Project — COSA+ study.*
*All modifications should be placed in `experiments/`, never in `external/`.*
