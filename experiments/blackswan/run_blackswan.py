#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "COSA_ICLR2026"
sys.path.insert(0, str(EXTERNAL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_norm_module_cfg
from datasets.loader import get_test_dataloader
from experiments.blackswan.shift_injector import inject_shift
from experiments.run_cosa_plus import build_cfg, preflight_inputs
from experiments.tta.cosa_plus import COSAPlusAdapter
from models.build import build_model, build_norm_module, load_best_model
from models.forecast import forecast
from utils.misc import prepare_inputs, set_devices, set_seeds


def parse_args():
    parser = argparse.ArgumentParser(description="Black swan robustness experiment.")
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--model", default="DLinear")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument(
        "--variant",
        default="original",
        choices=["original", "vec_gate", "rich_ctx", "ctx_std_only", "cosa_plus", "no_tta"],
    )
    parser.add_argument("--shift_type", default="level", choices=["level", "variance", "trend", "spike"])
    parser.add_argument("--magnitude", type=float, default=3.0)
    parser.add_argument(
        "--t_shift_frac",
        type=float,
        default=0.4,
        help="Fraction of test set where shift is injected (default: 40%).",
    )
    parser.add_argument("--output_dir", default="./results/blackswan/")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ctx_len", type=int, default=10)
    parser.add_argument("--base_lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--visible_devices", default="0")
    return parser.parse_args()


def rolling_mse(errors: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute rolling MSE with given window size."""
    squared = errors ** 2
    if len(squared) == 0:
        return np.array([], dtype=np.float64)
    window = max(1, min(window, len(squared)))
    return np.convolve(squared, np.ones(window) / window, mode="valid")


def compute_recovery_steps(
    rolling: np.ndarray,
    t_shift: int,
    baseline_mse: float,
    threshold: float = 0.1,
) -> int:
    """
    Returns number of steps after t_shift until rolling MSE is within
    threshold fraction of baseline_mse. Returns -1 if never recovers.
    """
    if len(rolling) == 0:
        return -1
    start = min(max(t_shift, 0), len(rolling) - 1)
    after = rolling[start:]
    for i, value in enumerate(after):
        if abs(value - baseline_mse) / (baseline_mse + 1e-8) <= threshold:
            return i
    return -1


def compute_half_recovery_steps(rolling: np.ndarray, t_shift: int, baseline_mse: float) -> int:
    """
    Relative recovery metric: return steps after t_shift until rolling MSE
    drops halfway from the post-shift peak back toward the pre-shift baseline.

    This is less brittle than requiring return to within 10% of the pre-shift
    baseline under severe synthetic shifts.
    """
    if len(rolling) == 0:
        return -1
    start = min(max(t_shift, 0), len(rolling) - 1)
    after = rolling[start:]
    if len(after) == 0:
        return -1

    peak_mse = float(after.max())
    target = peak_mse - 0.5 * (peak_mse - baseline_mse)
    peak_idx = int(after.argmax())
    for i, value in enumerate(after[peak_idx:], start=peak_idx):
        if value <= target:
            return i
    return -1


def _resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_cfg_model(args):
    cfg = build_cfg(args)
    preflight_inputs(cfg)
    set_devices(cfg.VISIBLE_DEVICES)
    set_seeds(cfg.SEED)

    model = build_model(cfg)
    norm_module = build_norm_module(cfg) if cfg.NORM_MODULE.ENABLE else None
    model = load_best_model(cfg, model)
    if cfg.NORM_MODULE.ENABLE:
        norm_module = load_best_model(get_norm_module_cfg(cfg), norm_module)
    return cfg, model, norm_module


def _load_train_and_test_data(cfg) -> tuple[np.ndarray, np.ndarray]:
    loader = get_test_dataloader(cfg)
    dataset = loader.dataset
    return dataset.train.copy(), dataset.test.copy()


def _patch_loader_test_data(loader, perturbed_test: np.ndarray):
    loader.dataset.test = perturbed_test.copy()
    return loader


@torch.no_grad()
def run_no_tta(cfg, model, norm_module, perturbed_test: np.ndarray) -> np.ndarray:
    cfg.TEST.BATCH_SIZE = len(get_test_dataloader(cfg).dataset)
    loader = _patch_loader_test_data(get_test_dataloader(cfg), perturbed_test)
    model.eval()
    if norm_module is not None:
        norm_module.eval()

    mse_all = []
    cur_step = cfg.DATA.SEQ_LEN - 2

    for inputs in loader:
        enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
        batch_start = 0
        batch_end = 0

        while batch_end < len(enc_window_all):
            batch_size = getattr(cfg.TTA.COSA, "BATCH_SIZE", 48)
            batch_end = min(batch_start + batch_size, len(enc_window_all))
            cur_step += batch_end - batch_start
            batch_inputs = (
                enc_window_all[batch_start:batch_end],
                enc_window_stamp_all[batch_start:batch_end],
                dec_window_all[batch_start:batch_end],
                dec_window_stamp_all[batch_start:batch_end],
            )
            pred, ground_truth = forecast(cfg, batch_inputs, model, norm_module)
            mse = F.mse_loss(pred, ground_truth, reduction="none").mean(dim=(-2, -1)).detach().cpu().numpy()
            mse_all.append(mse)
            batch_start = batch_end

    assert cur_step == len(perturbed_test) - cfg.DATA.PRED_LEN - 1
    return np.concatenate(mse_all)


def run_tta(cfg, model, norm_module, variant: str, perturbed_test: np.ndarray) -> tuple[np.ndarray, str]:
    adapter = COSAPlusAdapter(cfg, model, norm_module=norm_module, variant=variant)
    adapter.test_loader = _patch_loader_test_data(adapter.test_loader, perturbed_test)
    adapter.test_data = perturbed_test
    adapter.adapt()
    gate_path = adapter.save_gate(cfg.RESULT_DIR, cfg.DATA.NAME, cfg.MODEL.NAME, cfg.DATA.PRED_LEN)
    return adapter.mse_all.copy(), gate_path


def _window_mean(values: np.ndarray, start: int, end: int) -> float:
    start = max(0, start)
    end = min(len(values), end)
    if start >= end:
        return float("nan")
    return float(values[start:end].mean())


def main():
    args = parse_args()
    if not 0.0 < args.t_shift_frac < 1.0:
        raise ValueError("--t_shift_frac must be between 0 and 1.")

    output_dir = _resolve_output_dir(args.output_dir)
    cfg, model, norm_module = _load_cfg_model(args)

    train_data, test_data = _load_train_and_test_data(cfg)
    train_std = float(train_data.std())
    raw_t_shift = int(args.t_shift_frac * len(test_data))
    raw_t_shift = min(max(raw_t_shift, 0), len(test_data) - 1)
    perturbed_test = inject_shift(test_data, args.shift_type, raw_t_shift, args.magnitude, train_std)

    if args.variant == "no_tta":
        mse_series = run_no_tta(cfg, model, norm_module, perturbed_test)
        gate_path = None
    else:
        mse_series, gate_path = run_tta(cfg, model, norm_module, args.variant, perturbed_test)

    # Dataset sample i predicts raw test indices
    # [i + seq_len, i + seq_len + pred_len - 1]. The first affected
    # prediction is therefore where that horizon first overlaps raw_t_shift.
    t_shift = raw_t_shift - cfg.DATA.SEQ_LEN - cfg.DATA.PRED_LEN + 1
    t_shift = min(max(t_shift, 0), len(mse_series) - 1)
    mse_full = float(mse_series.mean())
    mse_before = _window_mean(mse_series, t_shift - 50, t_shift)
    mse_after_50 = _window_mean(mse_series, t_shift, t_shift + 50)
    mse_after_100 = _window_mean(mse_series, t_shift, t_shift + 100)
    # The runner stores per-sample MSE, so sqrt converts it to an error magnitude
    # before applying the prompt's rolling_mse(errors) helper.
    errors = np.sqrt(np.maximum(mse_series, 0.0))
    rolling = rolling_mse(errors, window=20)
    baseline_recovery = compute_recovery_steps(rolling, t_shift, mse_before)
    half_recovery = compute_half_recovery_steps(rolling, t_shift, mse_before)

    results = {
        "dataset": args.dataset,
        "model": args.model,
        "pred_len": args.pred_len,
        "variant": args.variant,
        "shift_type": args.shift_type,
        "magnitude": args.magnitude,
        "train_std": train_std,
        "raw_t_shift": raw_t_shift,
        "t_shift": t_shift,
        "mse_full": mse_full,
        "mse_before": mse_before,
        "mse_after": mse_after_50,
        "mse_after_50": mse_after_50,
        "mse_after_100": mse_after_100,
        "recovery_steps": int(half_recovery),
        "half_recovery_steps": int(half_recovery),
        "baseline_recovery_steps": int(baseline_recovery),
        "recovery_metric": "halfway_down_from_post_shift_peak",
        "rolling_mse": rolling.tolist(),
        "gate_path": gate_path,
    }

    fname = f"{args.variant}_{args.shift_type}_m{args.magnitude:g}_{args.pred_len}.json"
    output_path = output_dir / fname
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(
        f"MSE full={mse_full:.4f}  before={mse_before:.4f}  "
        f"after50={mse_after_50:.4f}  after100={mse_after_100:.4f}  "
        f"half_recovery={half_recovery} steps"
    )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
