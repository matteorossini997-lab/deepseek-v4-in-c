from __future__ import annotations

from pathlib import Path

import torch

from tools.dsv4_mini.config import load_config
from tools.dsv4_mini.initialization import initialize_reference_model
from tools.dsv4_mini.reference import HashRouter, LearnedRouter, sqrtsoftplus


CONFIG_PATH = Path(__file__).resolve().parents[2] / "tools" / "dsv4_mini" / "tiny_config.json"


def test_sqrtsoftplus_is_positive_and_monotonic() -> None:
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    y = sqrtsoftplus(x)

    assert torch.all(y > 0)
    assert torch.all(y[1:] > y[:-1])


def test_hash_router_returns_six_unique_layer_specific_experts() -> None:
    cfg = load_config(CONFIG_PATH)
    router0 = HashRouter(cfg, layer_idx=0)
    router1 = HashRouter(cfg, layer_idx=1)
    token_ids = torch.tensor([[0, 1, 17, 255]])
    hidden = torch.randn(1, 4, cfg.hidden_size)

    _, _, route0 = router0(hidden, token_ids)
    _, _, route1 = router1(hidden, token_ids)

    assert route0.shape == (4, cfg.num_experts_per_token)
    assert all(len(set(row.tolist())) == cfg.num_experts_per_token for row in route0)
    assert not torch.equal(route0, route1)


def test_learned_router_bias_changes_selection_but_not_weight_source() -> None:
    cfg = load_config(CONFIG_PATH)
    router = LearnedRouter(cfg)
    with torch.no_grad():
        router.weight.zero_()
        router.weight[:, 0] = torch.linspace(-0.02, 0.02, cfg.num_routed_experts)
        router.correction_bias.copy_(torch.linspace(0.5, -0.5, cfg.num_routed_experts))
    hidden = torch.zeros(1, 2, cfg.hidden_size)
    hidden[..., 0] = 1.0

    logits, weights, indices = router(hidden)
    scores = sqrtsoftplus(logits)
    expected_indices = torch.topk(
        scores + router.correction_bias,
        cfg.num_experts_per_token,
        dim=-1,
        sorted=False,
    ).indices
    expected_weights = scores.gather(1, expected_indices)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = expected_weights * cfg.routed_scaling_factor

    assert torch.equal(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights)
    assert not torch.equal(indices, torch.topk(scores, cfg.num_experts_per_token, dim=-1, sorted=False).indices)


def test_adversarial_initialization_exercises_correction_bias() -> None:
    cfg = load_config(CONFIG_PATH)
    model = initialize_reference_model(cfg, device="cpu")
    token_ids = torch.arange(32).view(1, -1) % cfg.vocab_size
    hidden = model.embed_tokens(token_ids)
    learned = model.layers[cfg.num_hash_layers].moe.router

    logits, _, biased = learned(hidden)
    unbiased = torch.topk(sqrtsoftplus(logits), cfg.num_experts_per_token, dim=-1, sorted=False).indices

    assert torch.any(torch.sort(biased, dim=-1).values != torch.sort(unbiased, dim=-1).values)
