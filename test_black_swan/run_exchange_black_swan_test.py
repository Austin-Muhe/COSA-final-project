#!/usr/bin/env python3
"""Test-only Exchange Rate black swan stress runner.

This file intentionally lives outside external/COSA_ICLR2026 so the original
COSA source stays untouched. It reuses COSA modules by importing them from the
submodule and writes all artifacts under results/test_black_swan/.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COSA_DIR = PROJECT_ROOT / "external" / "COSA_ICLR2026"
if str(COSA_DIR) not in sys.path:
    sys.path.insert(0, str(COSA_DIR))

from config import get_cfg_defaults  # noqa: E402
from datasets.build import update_cfg_from_dataset  # noqa: E402
from datasets.build import Exchange  # noqa: E402
from models.build import build_model, load_best_model  # noqa: E402
from models.forecast import forecast  # noqa: E402
from models.optimizer import get_optimizer  # noqa: E402
from tta.cosa import SimpleOutputAdapter  # noqa: E402
from utils.misc import prepare_inputs, set_devices, set_seeds  # noqa: E402


@dataclass(frozen=True)
class Experiment:
    name: str
    anomaly_type: str
    severity: float


class SpikeAnomalyExchange(Exchange):
    """Exchange dataset with abrupt spike corruption on the second half of test data."""

    def __init__(self, *args, spike_alpha: float = 0.0, shift_fraction: float = 0.5, **kwargs):
        self.spike_alpha = float(spike_alpha)
        self.shift_fraction = float(shift_fraction)
        super().__init__(*args, **kwargs)

    def _normalize_data(self):
        super()._normalize_data()
        if self.split != "test" or self.spike_alpha <= 0:
            return

        sigma = np.std(self.test, axis=0, keepdims=True)
        shift_start = int(len(self.test) * self.shift_fraction)
        self.test = self.test.copy()
        self.test[shift_start:] += self.spike_alpha * sigma


def ensure_runtime_data() -> Path:
    shared = PROJECT_ROOT / "shared_data" / "exchange_rate" / "exchange_rate.csv"
    runtime = COSA_DIR / "data" / "exchange_rate" / "exchange_rate.csv"
    if not shared.exists():
        raise FileNotFoundError(f"Missing shared dataset: {shared}")
    runtime.parent.mkdir(parents=True, exist_ok=True)
    if not runtime.exists() or runtime.stat().st_size != shared.stat().st_size:
        shutil.copy2(shared, runtime)
    return runtime


def build_cfg(args: argparse.Namespace):
    cfg = get_cfg_defaults()
    update_cfg_from_dataset(cfg, "exchange_rate")

    cfg.VISIBLE_DEVICES = args.visible_devices
    cfg.DATA.BASE_DIR = str(COSA_DIR / "data")
    cfg.DATA.NAME = "exchange_rate"
    cfg.DATA.PRED_LEN = args.pred_len
    cfg.MODEL.NAME = args.model
    cfg.MODEL.pred_len = args.pred_len
    cfg.MODEL.seq_len = cfg.DATA.SEQ_LEN
    cfg.MODEL.label_len = cfg.DATA.LABEL_LEN

    cfg.TRAIN.CHECKPOINT_DIR = str(
        COSA_DIR / "checkpoints" / "test_black_swan" / args.model / f"exchange_rate_{args.pred_len}"
    )
    cfg.RESULT_DIR = str(
        PROJECT_ROOT / "results" / "test_black_swan" / args.model / f"exchange_rate_{args.pred_len}"
    )

    cfg.SOLVER.MAX_EPOCH = args.epochs
    cfg.TRAIN.PRINT_FREQ = args.print_freq
    cfg.TRAIN.BATCH_SIZE = args.train_batch_size
    cfg.VAL.BATCH_SIZE = args.eval_batch_size
    cfg.TEST.BATCH_SIZE = args.eval_batch_size

    cfg.TTA.ENABLE = True
    cfg.TTA.COSA.BATCH_SIZE = args.cosa_batch_size
    cfg.TTA.COSA.STEPS = args.cosa_steps
    cfg.TTA.COSA.BUFFER_CONTEXT_SIZE = args.buffer_context_size
    cfg.TTA.COSA.FAST_ADAPTATION = True
    cfg.TTA.COSA.PER_BATCH_LR_RESET = True
    cfg.TTA.COSA.ADAPTIVE_LR = True
    cfg.TTA.COSA.PAAS = False
    cfg.TTA.COSA.VAR_WISE_GATING = True
    cfg.TTA.SOLVER.BASE_LR = args.cosa_lr
    cfg.TTA.SOLVER.WEIGHT_DECAY = args.cosa_weight_decay
    return cfg


def make_dataset(cfg, split: str, severity: float = 0.0, shift_fraction: float = 0.5):
    return SpikeAnomalyExchange(
        data_dir=Path(cfg.DATA.BASE_DIR) / cfg.DATA.NAME,
        n_var=cfg.DATA.N_VAR,
        seq_len=cfg.DATA.SEQ_LEN,
        label_len=cfg.DATA.LABEL_LEN,
        pred_len=cfg.DATA.PRED_LEN,
        features=cfg.DATA.FEATURES,
        timeenc=cfg.DATA.TIMEENC,
        freq=cfg.DATA.FREQ,
        date_idx=cfg.DATA.DATE_IDX,
        target_start_idx=cfg.DATA.TARGET_START_IDX,
        scale=cfg.DATA.SCALE,
        split=split,
        train_ratio=cfg.DATA.TRAIN_RATIO,
        test_ratio=cfg.DATA.TEST_RATIO,
        spike_alpha=severity,
        shift_fraction=shift_fraction,
    )


def make_loader(
    cfg,
    split: str,
    severity: float = 0.0,
    batch_size: int | None = None,
    shift_fraction: float = 0.5,
):
    if split == "train":
        shuffle = cfg.TRAIN.SHUFFLE
        drop_last = cfg.TRAIN.DROP_LAST
        batch_size = batch_size or cfg.TRAIN.BATCH_SIZE
    elif split == "val":
        shuffle = False
        drop_last = False
        batch_size = batch_size or cfg.VAL.BATCH_SIZE
    else:
        shuffle = False
        drop_last = False
        batch_size = batch_size or cfg.TEST.BATCH_SIZE

    return DataLoader(
        make_dataset(cfg, split, severity=severity, shift_fraction=shift_fraction),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )


def train_baseline_if_needed(cfg, force_train: bool = False) -> Path:
    checkpoint_dir = Path(cfg.TRAIN.CHECKPOINT_DIR)
    checkpoint_path = checkpoint_dir / "checkpoint_best.pth"
    if checkpoint_path.exists() and not force_train:
        return checkpoint_path

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.RESULT_DIR).mkdir(parents=True, exist_ok=True)
    model = build_model(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    train_loader = make_loader(cfg, "train")
    val_loader = make_loader(cfg, "val")

    best_mae = float("inf")
    best_state = None
    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        model.train()
        for inputs in train_loader:
            pred, ground_truth = forecast(cfg, inputs, model, None)
            loss = F.mse_loss(pred, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mae = evaluate_mae(cfg, model, val_loader)
        print(f"Epoch {epoch + 1}/{cfg.SOLVER.MAX_EPOCH} val_mae={val_mae:.6f}")
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {
                "epoch": epoch,
                "model_state": deepcopy(model.state_dict()),
                "optimizer_state": deepcopy(optimizer.state_dict()),
                "cfg": cfg.dump(),
                "best_val_mae": best_mae,
            }

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state")
    torch.save(best_state, checkpoint_path)
    return checkpoint_path


@torch.no_grad()
def evaluate_mae(cfg, model, loader) -> float:
    model.eval()
    values = []
    for inputs in loader:
        pred, ground_truth = forecast(cfg, inputs, model, None)
        mae = F.l1_loss(pred, ground_truth, reduction="none").mean(dim=(-2, -1))
        values.append(mae.detach().cpu())
    return float(torch.cat(values).mean().item())


@torch.no_grad()
def summarize_shift_segments(cfg, mse_arr: np.ndarray, shift_fraction: float = 0.5) -> dict:
    dataset = make_dataset(cfg, "test", severity=0.0, shift_fraction=shift_fraction)
    shift_data_index = int(len(dataset.test) * shift_fraction)
    target_start = np.arange(len(mse_arr)) + cfg.DATA.SEQ_LEN
    target_end = target_start + cfg.DATA.PRED_LEN

    before_mask = target_end <= shift_data_index
    transition_mask = (target_start < shift_data_index) & (target_end > shift_data_index)
    after_mask = target_start >= shift_data_index

    def masked_mean(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(mse_arr[mask].mean())

    return {
        "shift_data_index": int(shift_data_index),
        "before_window_count": int(before_mask.sum()),
        "transition_window_count": int(transition_mask.sum()),
        "after_window_count": int(after_mask.sum()),
        "before_mse": masked_mean(before_mask),
        "transition_mse": masked_mean(transition_mask),
        "after_mse": masked_mean(after_mask),
    }


def run_baseline(cfg, severity: float, output_dir: Path, shift_fraction: float = 0.5) -> dict:
    model = load_best_model(cfg, build_model(cfg))
    model.eval()
    loader = make_loader(
        cfg,
        "test",
        severity=severity,
        batch_size=cfg.TEST.BATCH_SIZE,
        shift_fraction=shift_fraction,
    )

    preds = []
    truths = []
    per_window_mse = []
    per_window_mae = []
    for inputs in loader:
        pred, ground_truth = forecast(cfg, inputs, model, None)
        mse = F.mse_loss(pred, ground_truth, reduction="none").mean(dim=(-2, -1))
        mae = F.l1_loss(pred, ground_truth, reduction="none").mean(dim=(-2, -1))
        preds.append(pred.detach().cpu().numpy())
        truths.append(ground_truth.detach().cpu().numpy())
        per_window_mse.append(mse.detach().cpu().numpy())
        per_window_mae.append(mae.detach().cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0)
    truth_arr = np.concatenate(truths, axis=0)
    mse_arr = np.concatenate(per_window_mse, axis=0)
    mae_arr = np.concatenate(per_window_mae, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "baseline_pred.npy", pred_arr)
    np.save(output_dir / "ground_truth.npy", truth_arr)
    np.save(output_dir / "baseline_mse_all.npy", mse_arr)
    np.save(output_dir / "baseline_mae_all.npy", mae_arr)

    return {
        "mse": float(mse_arr.mean()),
        "mae": float(mae_arr.mean()),
        "mse_all": mse_arr,
        **summarize_shift_segments(cfg, mse_arr, shift_fraction),
    }


def run_cosa(cfg, severity: float, output_dir: Path, shift_fraction: float = 0.5) -> dict:
    model = load_best_model(cfg, build_model(cfg))
    model.eval()
    test_size = len(make_dataset(cfg, "test", severity=severity, shift_fraction=shift_fraction))
    loader = make_loader(
        cfg,
        "test",
        severity=severity,
        batch_size=test_size,
        shift_fraction=shift_fraction,
    )

    output_adapter = SimpleOutputAdapter(
        pred_len=cfg.DATA.PRED_LEN,
        buffer_context_size=cfg.TTA.COSA.BUFFER_CONTEXT_SIZE,
        n_vars=cfg.DATA.N_VAR,
        var_wise_gating=cfg.TTA.COSA.VAR_WISE_GATING,
        num_layers=cfg.TTA.COSA.ADAPTER_LAYERS,
        hidden_dim=cfg.TTA.COSA.HIDDEN_DIM,
    )
    if torch.cuda.is_available():
        output_adapter = output_adapter.cuda()

    for param in model.parameters():
        param.requires_grad_(False)
    for param in output_adapter.parameters():
        param.requires_grad_(True)

    optimizer = get_optimizer(output_adapter.parameters(), cfg.TTA)
    sample_history = []
    mse_all = []
    mae_all = []
    base_preds = []
    cosa_preds = []
    truths = []
    n_adapt = 0

    model.eval()
    output_adapter.eval()
    for inputs in loader:
        enc_all, enc_stamp_all, dec_all, dec_stamp_all = prepare_inputs(inputs)
        batch_start = 0
        batch_idx = 0
        while batch_start < len(enc_all):
            batch_end = min(batch_start + cfg.TTA.COSA.BATCH_SIZE, len(enc_all))
            batch_size = batch_end - batch_start
            batch_inputs = (
                enc_all[batch_start:batch_end],
                enc_stamp_all[batch_start:batch_end],
                dec_all[batch_start:batch_end],
                dec_stamp_all[batch_start:batch_end],
            )

            pred, ground_truth = forecast(cfg, batch_inputs, model, None)
            original_pred = pred.detach()
            if sample_history:
                context = make_context(sample_history, cfg.TTA.COSA.BUFFER_CONTEXT_SIZE, batch_size)
                with torch.no_grad():
                    pred = output_adapter(pred, context)

            mse = F.mse_loss(pred, ground_truth, reduction="none").mean(dim=(-2, -1))
            mae = F.l1_loss(pred, ground_truth, reduction="none").mean(dim=(-2, -1))
            mse_all.append(mse.detach().cpu().numpy())
            mae_all.append(mae.detach().cpu().numpy())
            base_preds.append(original_pred.detach().cpu().numpy())
            cosa_preds.append(pred.detach().cpu().numpy())
            truths.append(ground_truth.detach().cpu().numpy())

            sample_history.append(float(ground_truth.mean().item()))
            context = make_context(sample_history, cfg.TTA.COSA.BUFFER_CONTEXT_SIZE, batch_size)

            output_adapter.train()
            effective_steps = min(cfg.TTA.COSA.STEPS, 5) if cfg.TTA.COSA.FAST_ADAPTATION else cfg.TTA.COSA.STEPS
            for _ in range(effective_steps):
                adapted_pred = output_adapter(original_pred, context)
                loss = F.mse_loss(adapted_pred, ground_truth)
                l2_reg = sum(p.pow(2).sum() for p in output_adapter.parameters() if p.requires_grad)
                loss = loss + 1e-4 * l2_reg
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(output_adapter.parameters(), max_norm=0.1)
                optimizer.step()
                n_adapt += 1
            output_adapter.eval()

            batch_start = batch_end
            batch_idx += 1

    mse_arr = np.concatenate(mse_all, axis=0)
    mae_arr = np.concatenate(mae_all, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "baseline_pred_seen_by_cosa.npy", np.concatenate(base_preds, axis=0))
    np.save(output_dir / "cosa_pred.npy", np.concatenate(cosa_preds, axis=0))
    np.save(output_dir / "ground_truth.npy", np.concatenate(truths, axis=0))
    np.save(output_dir / "cosa_mse_all.npy", mse_arr)
    np.save(output_dir / "cosa_mae_all.npy", mae_arr)

    return {
        "mse": float(mse_arr.mean()),
        "mae": float(mae_arr.mean()),
        "mse_all": mse_arr,
        "adaptation_count": int(n_adapt),
        **summarize_shift_segments(cfg, mse_arr, shift_fraction),
    }


def make_context(history: list[float], context_size: int, batch_size: int):
    values = list(reversed(history[-context_size:]))
    if len(values) < context_size:
        values.extend([values[-1] if values else 0.0] * (context_size - len(values)))
    context = torch.tensor(values, dtype=torch.float32)
    if torch.cuda.is_available():
        context = context.cuda()
    return context.unsqueeze(0).expand(batch_size, -1)


def write_summary(rows: Iterable[dict], output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "experiment_summary.csv"
    fieldnames = [
        "Dataset",
        "Model",
        "Horizon_L",
        "Anomaly_Type",
        "Severity",
        "Baseline_MSE",
        "COSA_MSE",
        "Improvement_percent",
        "NAR_percent",
        "Shift_Data_Index",
        "Before_Window_Count",
        "Transition_Window_Count",
        "After_Window_Count",
        "Baseline_Before_Shift_MSE",
        "COSA_Before_Shift_MSE",
        "Before_Shift_Improvement_percent",
        "Baseline_Transition_MSE",
        "COSA_Transition_MSE",
        "Transition_Improvement_percent",
        "Baseline_After_Shift_MSE",
        "COSA_After_Shift_MSE",
        "After_Shift_Improvement_percent",
        "Baseline_Result_Path",
        "COSA_Result_Path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test-only Exchange black swan COSA experiments.")
    parser.add_argument("--model", default="DLinear")
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--visible-devices", default="0")
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--cosa-batch-size", type=int, default=48)
    parser.add_argument("--cosa-steps", type=int, default=3)
    parser.add_argument("--buffer-context-size", type=int, default=10)
    parser.add_argument("--cosa-lr", type=float, default=0.001)
    parser.add_argument("--cosa-weight-decay", type=float, default=0.0001)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--severity", type=float, nargs="+", default=[0.0, 5.0, 10.0])
    parser.add_argument("--shift-fraction", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_runtime_data()
    cfg = build_cfg(args)
    set_devices(cfg.VISIBLE_DEVICES)
    set_seeds(cfg.SEED)

    checkpoint_path = train_baseline_if_needed(cfg, force_train=args.force_train)
    print(f"Using checkpoint: {checkpoint_path}")

    output_root = Path(cfg.RESULT_DIR)
    rows = []
    for severity in args.severity:
        experiment = Experiment(
            name="original" if severity == 0 else f"abrupt_spike_{severity:g}sigma",
            anomaly_type="None" if severity == 0 else "AbruptSpike",
            severity=severity,
        )
        exp_dir = output_root / experiment.name
        baseline_dir = exp_dir / "baseline"
        cosa_dir = exp_dir / "cosa"
        baseline = run_baseline(cfg, severity, baseline_dir, shift_fraction=args.shift_fraction)
        cosa = run_cosa(cfg, severity, cosa_dir, shift_fraction=args.shift_fraction)
        improvement = (baseline["mse"] - cosa["mse"]) / baseline["mse"] * 100.0
        before_improvement = (
            (baseline["before_mse"] - cosa["before_mse"]) / baseline["before_mse"] * 100.0
        )
        transition_improvement = (
            (baseline["transition_mse"] - cosa["transition_mse"])
            / baseline["transition_mse"]
            * 100.0
        )
        after_improvement = (
            (baseline["after_mse"] - cosa["after_mse"]) / baseline["after_mse"] * 100.0
        )
        nar = float((cosa["mse_all"] > baseline["mse_all"]).mean() * 100.0)
        row = {
            "Dataset": "exchange_rate",
            "Model": cfg.MODEL.NAME,
            "Horizon_L": cfg.DATA.PRED_LEN,
            "Anomaly_Type": experiment.anomaly_type,
            "Severity": f"{severity:g}sigma",
            "Baseline_MSE": baseline["mse"],
            "COSA_MSE": cosa["mse"],
            "Improvement_percent": improvement,
            "NAR_percent": nar,
            "Shift_Data_Index": baseline["shift_data_index"],
            "Before_Window_Count": baseline["before_window_count"],
            "Transition_Window_Count": baseline["transition_window_count"],
            "After_Window_Count": baseline["after_window_count"],
            "Baseline_Before_Shift_MSE": baseline["before_mse"],
            "COSA_Before_Shift_MSE": cosa["before_mse"],
            "Before_Shift_Improvement_percent": before_improvement,
            "Baseline_Transition_MSE": baseline["transition_mse"],
            "COSA_Transition_MSE": cosa["transition_mse"],
            "Transition_Improvement_percent": transition_improvement,
            "Baseline_After_Shift_MSE": baseline["after_mse"],
            "COSA_After_Shift_MSE": cosa["after_mse"],
            "After_Shift_Improvement_percent": after_improvement,
            "Baseline_Result_Path": str(baseline_dir),
            "COSA_Result_Path": str(cosa_dir),
        }
        rows.append(row)
        print(json.dumps(row, indent=2))

    write_summary(rows, output_root)
    print(f"Summary saved to: {output_root / 'experiment_summary.csv'}")


if __name__ == "__main__":
    main()
