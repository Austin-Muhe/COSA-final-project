#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "COSA_ICLR2026"
sys.path.insert(0, str(EXTERNAL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_cfg_defaults, get_norm_module_cfg
from datasets.build import update_cfg_from_dataset
from datasets.loader import get_val_dataloader
from models.build import build_model, build_norm_module, load_best_model
from models.forecast import forecast
from tta.cosa import DynamicRegimeAdapter, dynamic_regime_inference, train_regime_adapter
from utils.misc import prepare_inputs, set_devices, set_seeds


DATA_FILES = {
    "ETTh1": "ETTh1.csv",
    "weather": "weather.csv",
    "exchange_rate": "exchange_rate.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Meta-train DynamicRegimeAdapter on validation black-swan shifts.")
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--model", default="DLinear")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--shift_prob", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.7)
    parser.add_argument("--mae_weight", type=float, default=0.3)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means use all validation batches.")
    parser.add_argument("--output_dir", default="./results/regime_adapter")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--visible_devices", default="0")
    return parser.parse_args()


def build_cfg(args):
    cfg = get_cfg_defaults()
    cfg.VISIBLE_DEVICES = args.visible_devices
    cfg.DATA.NAME = args.dataset
    cfg.DATA.BASE_DIR = str(EXTERNAL_ROOT / "data")
    cfg.DATA.PRED_LEN = args.pred_len
    cfg.MODEL.NAME = args.model
    cfg.MODEL.pred_len = args.pred_len
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = False
    cfg.TTA.ENABLE = False
    cfg.VAL.BATCH_SIZE = args.batch_size
    cfg.VAL.SHUFFLE = False
    cfg.VAL.DROP_LAST = False

    update_cfg_from_dataset(cfg, args.dataset)

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = EXTERNAL_ROOT / "checkpoints" / args.model / f"{args.dataset}_{args.pred_len}"
    cfg.TRAIN.CHECKPOINT_DIR = str(checkpoint_dir)
    return cfg


def preflight_inputs(cfg):
    data_file = DATA_FILES.get(cfg.DATA.NAME, f"{cfg.DATA.NAME}.csv")
    data_path = Path(cfg.DATA.BASE_DIR) / cfg.DATA.NAME / data_file
    checkpoint_path = Path(cfg.TRAIN.CHECKPOINT_DIR) / "checkpoint_best.pth"

    missing = []
    if not data_path.exists():
        missing.append(f"data file: {data_path}")
    if not checkpoint_path.exists():
        missing.append(f"checkpoint: {checkpoint_path}")
    if missing:
        raise FileNotFoundError("Missing required input(s):\n  - " + "\n  - ".join(missing))


@torch.no_grad()
def build_meta_batches(cfg, model, norm_module, max_batches=0):
    model.eval()
    if norm_module is not None:
        norm_module.eval()

    meta_batches = []
    val_loader = get_val_dataloader(cfg)
    for batch_idx, inputs in enumerate(val_loader):
        if max_batches and batch_idx >= max_batches:
            break

        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        base_forecast, y_true = forecast(
            cfg,
            (enc_window, enc_window_stamp, dec_window, dec_window_stamp),
            model,
            norm_module,
        )
        meta_batches.append(
            {
                "context_window": enc_window.detach().cpu(),
                "base_forecast": base_forecast.detach().cpu(),
                "y_true": y_true.detach().cpu(),
            }
        )

    if not meta_batches:
        raise RuntimeError("No validation meta-training batches were generated.")
    return meta_batches


@torch.no_grad()
def evaluate_adapter(dynamic_adapter, meta_batches, device):
    dynamic_adapter.eval()
    losses = []
    base_losses = []

    for batch in meta_batches:
        context_window = batch["context_window"].to(device)
        base_forecast = batch["base_forecast"].to(device)
        y_true = batch["y_true"].to(device)

        robust_forecast, _, _ = dynamic_regime_inference(dynamic_adapter, base_forecast, context_window)
        losses.append(F.mse_loss(robust_forecast, y_true).item())
        base_losses.append(F.mse_loss(base_forecast, y_true).item())

    return {
        "base_val_mse": sum(base_losses) / len(base_losses),
        "adapter_val_mse": sum(losses) / len(losses),
    }


def save_checkpoint(path, dynamic_adapter, cfg, args, history, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "adapter_state": dynamic_adapter.state_dict(),
        "dataset": args.dataset,
        "model": args.model,
        "pred_len": args.pred_len,
        "feature_dim": cfg.DATA.N_VAR,
        "horizon": cfg.DATA.PRED_LEN,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "mse_weight": args.mse_weight,
        "mae_weight": args.mae_weight,
        "history": history,
        "metrics": metrics,
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    cfg = build_cfg(args)
    preflight_inputs(cfg)
    set_devices(cfg.VISIBLE_DEVICES)
    set_seeds(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    meta_batches = build_meta_batches(cfg, model, norm_module, max_batches=args.max_batches)
    dynamic_adapter = DynamicRegimeAdapter(
        feature_dim=cfg.DATA.N_VAR,
        horizon=cfg.DATA.PRED_LEN,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    before_metrics = evaluate_adapter(dynamic_adapter, meta_batches, device)
    history = train_regime_adapter(
        dynamic_adapter,
        meta_batches,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        shift_prob=args.shift_prob,
        mse_weight=args.mse_weight,
        mae_weight=args.mae_weight,
    )
    after_metrics = evaluate_adapter(dynamic_adapter, meta_batches, device)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    run_dir = output_dir / args.model / args.dataset / str(args.pred_len)
    checkpoint_path = run_dir / "dynamic_regime_adapter.pt"
    metrics = {
        "before": before_metrics,
        "after": after_metrics,
        "history": history,
        "num_meta_batches": len(meta_batches),
    }
    save_checkpoint(checkpoint_path, dynamic_adapter, cfg, args, history, metrics)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "meta_training_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
