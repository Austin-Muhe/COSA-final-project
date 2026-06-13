import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "COSA_ICLR2026"))

from torch.utils.data import DataLoader, TensorDataset

from tta.cosa import (
    DynamicRegimeAdapter,
    SimpleOutputAdapter,
    dynamic_regime_inference,
    train_regime_adapter,
)


def test_dynamic_regime_adapter_forward_shapes():
    torch.manual_seed(0)
    batch_size = 4
    context_length = 96
    feature_dim = 7
    horizon = 24
    latent_dim = 8

    adapter = DynamicRegimeAdapter(
        feature_dim=feature_dim,
        horizon=horizon,
        latent_dim=latent_dim,
        hidden_dim=16,
        dropout=0.0,
    )
    context_window = torch.randn(batch_size, context_length, feature_dim)

    g, z = adapter(context_window)

    assert g.shape == (batch_size, horizon)
    assert z.shape == (batch_size, latent_dim)
    assert adapter.gate_generator[-1].bias.abs().max().item() == 0.0
    assert adapter.gate_generator[-1].weight.std().item() < 5e-4


def test_simple_output_adapter_dynamic_forward_is_near_identity_at_init():
    torch.manual_seed(0)
    batch_size = 4
    context_length = 96
    feature_dim = 7
    horizon = 24

    output_adapter = SimpleOutputAdapter(
        pred_len=horizon,
        buffer_context_size=5,
        n_vars=feature_dim,
        hidden_dim=16,
        latent_dim=8,
        dynamic_dropout=0.0,
    )
    base_forecast = torch.randn(batch_size, horizon, feature_dim)
    historical_context_data = torch.randn(batch_size, context_length, feature_dim)

    adapted_forecast = output_adapter(
        base_forecast,
        historical_context_data=historical_context_data,
    )

    assert adapted_forecast.shape == base_forecast.shape
    assert (adapted_forecast - base_forecast).abs().max().item() < 1e-2


def test_train_regime_adapter_updates_only_dynamic_adapter():
    torch.manual_seed(0)
    batch_size = 4
    context_length = 32
    feature_dim = 3
    horizon = 12

    dynamic_adapter = DynamicRegimeAdapter(
        feature_dim=feature_dim,
        horizon=horizon,
        latent_dim=8,
        hidden_dim=16,
        dropout=0.0,
    )
    context_window = torch.randn(batch_size, context_length, feature_dim)
    base_forecast = torch.randn(batch_size, horizon, feature_dim)
    y_true = base_forecast * 0.8

    loader = DataLoader(
        TensorDataset(context_window, base_forecast, y_true),
        batch_size=batch_size,
    )
    before = {name: param.detach().clone() for name, param in dynamic_adapter.named_parameters()}

    history = train_regime_adapter(
        dynamic_adapter,
        loader,
        epochs=2,
        lr=1e-2,
        shift_prob=1.0,
    )

    assert len(history) == 2
    assert all(param.requires_grad for param in dynamic_adapter.parameters())
    assert any(
        not torch.allclose(before[name], param.detach())
        for name, param in dynamic_adapter.named_parameters()
    )


def test_dynamic_regime_inference_has_no_backward_graph():
    torch.manual_seed(0)
    batch_size = 4
    context_length = 32
    feature_dim = 3
    horizon = 12

    dynamic_adapter = DynamicRegimeAdapter(
        feature_dim=feature_dim,
        horizon=horizon,
        latent_dim=8,
        hidden_dim=16,
        dropout=0.0,
    )
    dynamic_adapter.train()
    base_forecast = torch.randn(batch_size, horizon, feature_dim, requires_grad=True)
    realtime_context = torch.randn(batch_size, context_length, feature_dim, requires_grad=True)

    robust_forecast, g, z = dynamic_regime_inference(
        dynamic_adapter,
        base_forecast,
        realtime_context,
    )

    assert dynamic_adapter.training
    assert robust_forecast.shape == base_forecast.shape
    assert g.shape == (batch_size, horizon)
    assert z.shape == (batch_size, 8)
    assert not robust_forecast.requires_grad
    assert not g.requires_grad
    assert not z.requires_grad


if __name__ == "__main__":
    test_dynamic_regime_adapter_forward_shapes()
    test_simple_output_adapter_dynamic_forward_is_near_identity_at_init()
    test_train_regime_adapter_updates_only_dynamic_adapter()
    test_dynamic_regime_inference_has_no_backward_graph()
    print("DynamicRegimeAdapter tests passed.")
