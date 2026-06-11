#!/usr/bin/env python
import argparse
import json
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
from experiments.blackswan.run_blackswan import (
    compute_half_recovery_steps,
    compute_recovery_steps,
    rolling_mse,
)
from experiments.blackswan.shift_injector import inject_shift
from experiments.evaluate_dynamic_regime_adapter import (
    build_cfg,
    build_dynamic_adapter,
    preflight_inputs,
)
from models.build import build_model, build_norm_module, load_best_model
from models.forecast import forecast
from tta.cosa import dynamic_regime_inference
from utils.misc import prepare_inputs, set_devices, set_seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Black Swan evaluation for DynamicRegimeAdapter."
    )
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--model", default="DLinear")
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--shift_type", default="level", choices=["level", "variance", "trend", "spike"])
    parser.add_argument("--magnitude", type=float, default=3.0)
    parser.add_argument("--t_shift_frac", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--regime_checkpoint", default="")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--output_dir", default="./results/blackswan_dynamic_regime_h720_mag3")
    parser.add_argument("--visible_devices", default="0")
    return parser.parse_args()


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

    for param in model.parameters():
        param.requires_grad_(False)
    if norm_module is not None:
        for param in norm_module.parameters():
            param.requires_grad_(False)
    return cfg, model, norm_module


def _load_train_and_test_data(cfg) -> tuple[np.ndarray, np.ndarray]:
    loader = get_test_dataloader(cfg)
    dataset = loader.dataset
    return dataset.train.copy(), dataset.test.copy()


def _patch_loader_test_data(loader, perturbed_test: np.ndarray):
    loader.dataset.test = perturbed_test.copy()
    return loader


@torch.no_grad()
def evaluate_series(cfg, model, norm_module, dynamic_adapter, perturbed_test: np.ndarray):
    model.eval()
    dynamic_adapter.eval()
    if norm_module is not None:
        norm_module.eval()

    loader = _patch_loader_test_data(get_test_dataloader(cfg), perturbed_test)
    base_mse_all = []
    base_mae_all = []
    dyn_mse_all = []
    dyn_mae_all = []

    for inputs in loader:
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        base_forecast, y_true = forecast(
            cfg,
            (enc_window, enc_window_stamp, dec_window, dec_window_stamp),
            model,
            norm_module,
        )
        robust_forecast, _, _ = dynamic_regime_inference(
            dynamic_adapter, base_forecast, enc_window
        )

        base_mse = F.mse_loss(base_forecast, y_true, reduction="none").mean(dim=(-2, -1))
        base_mae = F.l1_loss(base_forecast, y_true, reduction="none").mean(dim=(-2, -1))
        dyn_mse = F.mse_loss(robust_forecast, y_true, reduction="none").mean(dim=(-2, -1))
        dyn_mae = F.l1_loss(robust_forecast, y_true, reduction="none").mean(dim=(-2, -1))

        base_mse_all.append(base_mse.detach().cpu().numpy())
        base_mae_all.append(base_mae.detach().cpu().numpy())
        dyn_mse_all.append(dyn_mse.detach().cpu().numpy())
        dyn_mae_all.append(dyn_mae.detach().cpu().numpy())

    return {
        "base_mse": np.concatenate(base_mse_all),
        "base_mae": np.concatenate(base_mae_all),
        "dynamic_mse": np.concatenate(dyn_mse_all),
        "dynamic_mae": np.concatenate(dyn_mae_all),
    }


def _window_mean(values: np.ndarray, start: int, end: int) -> float:
    start = max(0, start)
    end = min(len(values), end)
    if start >= end:
        return float("nan")
    return float(values[start:end].mean())


def summarize_series(values: np.ndarray, t_shift: int) -> dict:
    mse_full = float(values.mean())
    mse_before = _window_mean(values, t_shift - 50, t_shift)
    mse_after_50 = _window_mean(values, t_shift, t_shift + 50)
    mse_after_100 = _window_mean(values, t_shift, t_shift + 100)
    errors = np.sqrt(np.maximum(values, 0.0))
    rolling = rolling_mse(errors, window=20)
    return {
        "mse_full": mse_full,
        "mse_before": mse_before,
        "mse_after": mse_after_50,
        "mse_after_50": mse_after_50,
        "mse_after_100": mse_after_100,
        "recovery_steps": int(compute_half_recovery_steps(rolling, t_shift, mse_before)),
        "half_recovery_steps": int(compute_half_recovery_steps(rolling, t_shift, mse_before)),
        "baseline_recovery_steps": int(compute_recovery_steps(rolling, t_shift, mse_before)),
        "recovery_metric": "halfway_down_from_post_shift_peak",
        "rolling_mse": rolling.tolist(),
    }


def main():
    args = parse_args()
    if not 0.0 < args.t_shift_frac < 1.0:
        raise ValueError("--t_shift_frac must be between 0 and 1.")

    output_dir = _resolve_output_dir(args.output_dir)
    cfg, model, norm_module = _load_cfg_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dynamic_adapter = build_dynamic_adapter(cfg, args, device)

    train_data, test_data = _load_train_and_test_data(cfg)
    train_std = float(train_data.std())
    raw_t_shift = int(args.t_shift_frac * len(test_data))
    raw_t_shift = min(max(raw_t_shift, 0), len(test_data) - 1)
    perturbed_test = inject_shift(test_data, args.shift_type, raw_t_shift, args.magnitude, train_std)

    series = evaluate_series(cfg, model, norm_module, dynamic_adapter, perturbed_test)
    t_shift = raw_t_shift - cfg.DATA.SEQ_LEN - cfg.DATA.PRED_LEN + 1
    t_shift = min(max(t_shift, 0), len(series["dynamic_mse"]) - 1)

    run_name = "dynamic_meta_trained" if args.regime_checkpoint else "dynamic_untrained"
    base_summary = summarize_series(series["base_mse"], t_shift)
    dynamic_summary = summarize_series(series["dynamic_mse"], t_shift)

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "pred_len": args.pred_len,
        "variant": run_name,
        "shift_type": args.shift_type,
        "magnitude": args.magnitude,
        "train_std": train_std,
        "raw_t_shift": raw_t_shift,
        "t_shift": t_shift,
        "regime_checkpoint": args.regime_checkpoint,
        "base": base_summary,
        "dynamic": dynamic_summary,
        "base_mae_full": float(series["base_mae"].mean()),
        "dynamic_mae_full": float(series["dynamic_mae"].mean()),
    }

    fname = f"{run_name}_{args.shift_type}_m{args.magnitude:g}_{args.pred_len}.json"
    output_path = output_dir / fname
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(
        f"{run_name} {args.shift_type}: "
        f"base_after50={base_summary['mse_after_50']:.6f} "
        f"dynamic_after50={dynamic_summary['mse_after_50']:.6f} "
        f"dynamic_before={dynamic_summary['mse_before']:.6f}"
    )
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
