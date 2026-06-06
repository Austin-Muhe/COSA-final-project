import json
import os
import time
from collections import defaultdict, deque
from copy import deepcopy
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .calr import CALR


class COSAPlus(nn.Module):
    """Output-space adapter with scalar/vector gate and mean/std context variants."""

    VALID_VARIANTS = {"original", "vec_gate", "rich_ctx", "ctx_std_only", "cosa_plus"}

    def __init__(
        self,
        pred_len: int,
        ctx_len: int = 10,
        variant: str = "original",
        vec_gate: Optional[bool] = None,
        rich_ctx: Optional[bool] = None,
        ctx_mode: Optional[str] = None,
    ):
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown COSA+ variant: {variant}")

        self.pred_len = pred_len
        self.ctx_len = ctx_len
        self.variant = variant
        self.vec_gate = variant in {"vec_gate", "cosa_plus"} if vec_gate is None else vec_gate

        if ctx_mode is None:
            if variant == "ctx_std_only":
                ctx_mode = "std"
            elif variant in {"rich_ctx", "cosa_plus"}:
                ctx_mode = "mean_std"
            else:
                ctx_mode = "mean"
        if rich_ctx is not None:
            ctx_mode = "mean_std" if rich_ctx else "mean"
        if ctx_mode not in {"mean", "std", "mean_std"}:
            raise ValueError(f"Unknown context mode: {ctx_mode}")
        self.ctx_mode = ctx_mode

        ctx_step_dim = 2 if self.ctx_mode == "mean_std" else 1
        self.ctx_dim = self.ctx_len * ctx_step_dim
        self.linear = nn.Linear(self.pred_len + self.ctx_dim, self.pred_len, bias=True)
        gate_dim = self.pred_len if self.vec_gate else 1
        self.g = nn.Parameter(torch.zeros(gate_dim))
        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

    def build_context(self, gt_buffer: Iterable[torch.Tensor], batch_size: int = 1) -> torch.Tensor:
        values = []
        device = self.g.device
        dtype = self.g.dtype

        for batch in list(gt_buffer)[-self.ctx_len:]:
            batch = batch.detach().to(device=device, dtype=dtype)
            mu = batch.mean().reshape(1)
            std = (batch.std(unbiased=False) + 1e-8).reshape(1)
            if self.ctx_mode == "mean":
                values.append(mu)
            elif self.ctx_mode == "std":
                values.append(std)
            else:
                values.append(torch.cat([mu, std], dim=0))

        step_dim = 2 if self.ctx_mode == "mean_std" else 1
        while len(values) < self.ctx_len:
            values.insert(0, torch.zeros(step_dim, device=device, dtype=dtype))

        context = torch.cat(values, dim=0)
        return context.unsqueeze(0).expand(batch_size, -1)

    def forward(self, y_base: torch.Tensor, context_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        if context_data is None:
            return y_base

        if y_base.dim() == 1:
            combined_input = torch.cat([y_base, context_data], dim=-1)
            correction = self.linear(combined_input)
            return y_base + torch.tanh(self.g) * correction

        batch_size, pred_len, n_vars = y_base.shape
        if pred_len != self.pred_len:
            raise ValueError(f"Expected pred_len={self.pred_len}, got {pred_len}")

        y_flat = y_base.transpose(1, 2).contiguous().view(batch_size * n_vars, pred_len)
        context_repeated = context_data.unsqueeze(1).repeat(1, n_vars, 1).view(batch_size * n_vars, -1)
        combined_input = torch.cat([y_flat, context_repeated], dim=-1)
        correction = self.linear(combined_input)
        correction = correction.view(batch_size, n_vars, pred_len).transpose(1, 2)
        gate = torch.tanh(self.g).view(1, pred_len, 1) if self.vec_gate else torch.tanh(self.g)
        return y_base + gate * correction


class COSAPlusAdapter(nn.Module):
    """Official COSA streaming loop using COSAPlus as the output adapter."""

    def __init__(self, cfg, model: nn.Module, norm_module=None, variant: str = "original"):
        super().__init__()
        from config import get_norm_method
        from datasets.loader import get_test_dataloader
        from models.optimizer import get_optimizer

        self.cfg = cfg
        self.model = model
        self.norm_module = norm_module
        self.norm_method = get_norm_method(cfg)
        self.variant = variant
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test

        self.buffer_context_size = getattr(cfg.TTA.COSA, "BUFFER_CONTEXT_SIZE", 10)
        self.adapt_steps = getattr(cfg.TTA.COSA, "STEPS", 3)
        self.fast_adaptation = getattr(cfg.TTA.COSA, "FAST_ADAPTATION", True)
        self.adaptive_lr = getattr(cfg.TTA.COSA, "ADAPTIVE_LR", True)
        self.convergence_threshold = getattr(cfg.TTA.COSA, "CONVERGENCE_THRESHOLD", 1e-4)

        self.output_adapter = COSAPlus(
            pred_len=cfg.DATA.PRED_LEN,
            ctx_len=self.buffer_context_size,
            variant=variant,
        )
        if torch.cuda.is_available():
            self.output_adapter = self.output_adapter.cuda()

        self._freeze_all_model_params()
        self.optimizer = get_optimizer(self.output_adapter.parameters(), cfg.TTA)
        self.calr = CALR(
            base_lr=getattr(cfg.TTA.SOLVER, "BASE_LR", 0.001),
            min_lr=getattr(cfg.TTA.COSA, "MIN_LR", 0.0001),
            max_lr=getattr(cfg.TTA.COSA, "MAX_LR", 0.005),
            adapt_steps=self.adapt_steps,
            per_batch_lr_reset=getattr(cfg.TTA.COSA, "PER_BATCH_LR_RESET", True),
        )

        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()
        cfg.TEST.BATCH_SIZE = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(cfg)
        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.n_adapt = 0
        self.step_count = 0
        self.adapters_enabled = False
        self.sample_history = deque(maxlen=200)
        self.current_time_idx = 0
        self.mse_all = []
        self.mae_all = []
        self.time_stats = defaultdict(float)
        self.time_counts = defaultdict(int)

    def _copy_model_and_optimizer(self):
        return deepcopy(self.model.state_dict()), deepcopy(self.optimizer.state_dict())

    def _freeze_all_model_params(self):
        for param in self.model.parameters():
            param.requires_grad_(False)
        if self.norm_module is not None:
            for param in self.norm_module.parameters():
                param.requires_grad_(False)
        for param in self.output_adapter.parameters():
            param.requires_grad_(True)

    def switch_model_to_train(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.train()

    def switch_model_to_eval(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.eval()

    def update_memory_buffer(self, targets: torch.Tensor):
        self.sample_history.append(targets.detach())
        self.current_time_idx += targets.shape[0]
        self.step_count += 1
        self.adapters_enabled = True

    def _get_context_for_batch(self, batch_size: int) -> torch.Tensor:
        return self.output_adapter.build_context(self.sample_history, batch_size=batch_size)

    def _set_optimizer_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @torch.enable_grad()
    def adapt(self):
        from datasets.loader import get_test_dataloader
        from models.forecast import forecast
        from utils.misc import prepare_inputs

        batch_start = 0
        batch_end = 0
        batch_idx = 0
        total_start_time = time.time()
        self.switch_model_to_eval()

        for _, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
            batch_start = 0
            batch_end = 0

            while batch_end < len(enc_window_all):
                batch_size = getattr(self.cfg.TTA.COSA, "BATCH_SIZE", 48)
                batch_end = min(batch_start + batch_size, len(enc_window_all))
                batch_size = batch_end - batch_start
                self.cur_step += batch_size

                batch_inputs = (
                    enc_window_all[batch_start:batch_end],
                    enc_window_stamp_all[batch_start:batch_end],
                    dec_window_all[batch_start:batch_end],
                    dec_window_stamp_all[batch_start:batch_end],
                )

                pred_start = time.time()
                pred, ground_truth = forecast(self.cfg, batch_inputs, self.model, self.norm_module)
                original_pred = pred.clone()
                self.time_stats["base_prediction"] += time.time() - pred_start
                self.time_counts["base_prediction"] += 1

                if self.adapters_enabled:
                    context_data = self._get_context_for_batch(batch_size)
                    with torch.no_grad():
                        pred = self.output_adapter(pred, context_data)

                mse = F.mse_loss(pred, ground_truth, reduction="none").mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, ground_truth, reduction="none").mean(dim=(-2, -1)).detach().cpu().numpy()
                self.mse_all.append(mse)
                self.mae_all.append(mae)

                self.update_memory_buffer(ground_truth)

                context_data = self._get_context_for_batch(batch_size)
                effective_steps = min(self.adapt_steps, 5) if self.fast_adaptation else self.adapt_steps
                self.calr.reset_batch()

                for step in range(effective_steps):
                    self.n_adapt += 1
                    self.switch_model_to_train()
                    adapted_pred = self.output_adapter(original_pred, context_data)
                    loss = F.mse_loss(adapted_pred, ground_truth)
                    l2_reg = sum(p.pow(2).sum() for p in self.output_adapter.parameters() if p.requires_grad)
                    loss = loss + getattr(self.cfg.TTA.SOLVER, "WEIGHT_DECAY", 1e-4) * l2_reg

                    if self.adaptive_lr and self.fast_adaptation:
                        self._set_optimizer_lr(self.calr.step(loss.item(), step))

                    self.optimizer.zero_grad()
                    loss.backward()
                    max_norm = max(0.05, min(0.5, loss.item())) if self.fast_adaptation else 0.1
                    torch.nn.utils.clip_grad_norm_(self.output_adapter.parameters(), max_norm=max_norm)
                    self.optimizer.step()
                    self.switch_model_to_eval()

                    if (
                        self.fast_adaptation
                        and step > 2
                        and len(self.calr.loss_history) >= 2
                        and abs(self.calr.loss_history[-1] - self.calr.loss_history[-2]) < self.convergence_threshold
                    ):
                        break

                batch_start = batch_end
                batch_idx += 1

        self.time_stats["total_time"] = time.time() - total_start_time
        assert self.cur_step == len(self.test_data) - self.cfg.DATA.PRED_LEN - 1
        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)
        return self.get_results()

    def get_results(self):
        return {
            "variant": self.variant,
            "test_mse": float(self.mse_all.mean()),
            "test_mae": float(self.mae_all.mean()),
            "adaptation_count": int(self.n_adapt),
        }

    def save_gate(self, output_dir: str, dataset: str, model_name: str, pred_len: int) -> str:
        os.makedirs(output_dir, exist_ok=True)
        gate = torch.tanh(self.output_adapter.g).detach().cpu().numpy()
        path = os.path.join(output_dir, f"gate_{self.variant}_{dataset}_{model_name}_{pred_len}.npy")
        np.save(path, gate)
        return path

    def print_results(self):
        results = self.get_results()
        print(json.dumps(results, indent=2))
        print(f"MSE: {results['test_mse']:.6f}")
        print(f"MAE: {results['test_mae']:.6f}")
