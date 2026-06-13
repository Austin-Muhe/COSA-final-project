#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "COSA_ICLR2026"
sys.path.insert(0, str(EXTERNAL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_cfg_defaults, get_norm_module_cfg
from datasets.build import update_cfg_from_dataset
from datasets.loader import get_test_dataloader, get_val_dataloader
from models.build import build_model, build_norm_module, load_best_model
from models.forecast import forecast
from tta.cosa import DynamicRegimeAdapter, dynamic_regime_inference
from utils.misc import prepare_inputs, set_devices, set_seeds


DATA_FILES = {
    "ETTh1": "ETTh1.csv",
    "weather": "weather.csv",
    "exchange_rate": "exchange_rate.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DynamicRegimeAdapter with no test-time backprop.")
    parser.add_argument("--dataset", default="weather")
    parser.add_argument("--model", default="DLinear")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--regime_checkpoint", default="")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--output_dir", default="./results/regime_adapter_eval")
    parser.add_argument("--save_latents", action="store_true")
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
    cfg.TEST.BATCH_SIZE = args.batch_size
    cfg.VAL.SHUFFLE = False
    cfg.TEST.SHUFFLE = False
    cfg.VAL.DROP_LAST = False
    cfg.TEST.DROP_LAST = False

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


def build_dynamic_adapter(cfg, args, device):
    checkpoint_payload = None
    if args.regime_checkpoint:
        checkpoint_payload = torch.load(args.regime_checkpoint, map_location="cpu")
        args.latent_dim = int(checkpoint_payload.get("latent_dim", args.latent_dim))
        args.hidden_dim = int(checkpoint_payload.get("hidden_dim", args.hidden_dim))

    dynamic_adapter = DynamicRegimeAdapter(
        feature_dim=cfg.DATA.N_VAR,
        horizon=cfg.DATA.PRED_LEN,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    if checkpoint_payload is not None:
        state_dict = checkpoint_payload.get("adapter_state", checkpoint_payload)
        dynamic_adapter.load_state_dict(state_dict, strict=True)

    return dynamic_adapter


@torch.no_grad()
def evaluate(cfg, model, norm_module, dynamic_adapter, split, save_latents=False):
    model.eval()
    dynamic_adapter.eval()
    if norm_module is not None:
        norm_module.eval()

    dataloader = get_val_dataloader(cfg) if split == "val" else get_test_dataloader(cfg)
    base_mse_all = []
    base_mae_all = []
    dyn_mse_all = []
    dyn_mae_all = []
    latents = []
    gates = []

    for inputs in dataloader:
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        base_forecast, y_true = forecast(
            cfg,
            (enc_window, enc_window_stamp, dec_window, dec_window_stamp),
            model,
            norm_module,
        )
        robust_forecast, g, z = dynamic_regime_inference(dynamic_adapter, base_forecast, enc_window)

        base_mse = F.mse_loss(base_forecast, y_true, reduction="none").mean(dim=(-2, -1))
        base_mae = F.l1_loss(base_forecast, y_true, reduction="none").mean(dim=(-2, -1))
        dyn_mse = F.mse_loss(robust_forecast, y_true, reduction="none").mean(dim=(-2, -1))
        dyn_mae = F.l1_loss(robust_forecast, y_true, reduction="none").mean(dim=(-2, -1))

        base_mse_all.append(base_mse.cpu())
        base_mae_all.append(base_mae.cpu())
        dyn_mse_all.append(dyn_mse.cpu())
        dyn_mae_all.append(dyn_mae.cpu())
        if save_latents:
            latents.append(z.cpu())
            gates.append(g.cpu())

    base_mse_all = torch.cat(base_mse_all).numpy()
    base_mae_all = torch.cat(base_mae_all).numpy()
    dyn_mse_all = torch.cat(dyn_mse_all).numpy()
    dyn_mae_all = torch.cat(dyn_mae_all).numpy()

    metrics = {
        "split": split,
        "base_mse": float(base_mse_all.mean()),
        "base_mae": float(base_mae_all.mean()),
        "dynamic_mse": float(dyn_mse_all.mean()),
        "dynamic_mae": float(dyn_mae_all.mean()),
        "mse_delta": float(dyn_mse_all.mean() - base_mse_all.mean()),
        "mae_delta": float(dyn_mae_all.mean() - base_mae_all.mean()),
        "mse_relative_improvement_pct": float((base_mse_all.mean() - dyn_mse_all.mean()) / base_mse_all.mean() * 100.0),
        "mae_relative_improvement_pct": float((base_mae_all.mean() - dyn_mae_all.mean()) / base_mae_all.mean() * 100.0),
        "num_samples": int(len(base_mse_all)),
    }

    extras = {}
    if save_latents:
        extras["z"] = torch.cat(latents).numpy()
        extras["g"] = torch.cat(gates).numpy()
    return metrics, extras


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

    dynamic_adapter = build_dynamic_adapter(cfg, args, device)
    metrics, extras = evaluate(cfg, model, norm_module, dynamic_adapter, args.split, args.save_latents)
    metrics.update(
        {
            "dataset": args.dataset,
            "model": args.model,
            "pred_len": args.pred_len,
            "regime_checkpoint": args.regime_checkpoint,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
        }
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    run_name = "meta_trained" if args.regime_checkpoint else "untrained"
    run_dir = output_dir / run_name / args.model / args.dataset / str(args.pred_len)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / f"metrics_{args.split}.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    for name, value in extras.items():
        np.save(run_dir / f"{name}_{args.split}.npy", value)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
