from __future__ import annotations

from pathlib import Path

import torch

from tools.dsv4_mini.config import load_config
from tools.dsv4_mini.reference import HyperConnection


CONFIG_PATH = Path(__file__).resolve().parents[2] / "tools" / "dsv4_mini" / "tiny_config.json"


def test_hyperconnection_sinkhorn_is_doubly_stochastic() -> None:
    cfg = load_config(CONFIG_PATH)
    torch.manual_seed(7)
    hc = HyperConnection(cfg)
    streams = torch.randn(2, 3, cfg.hc_mult, cfg.hidden_size)

    post, comb, collapsed = hc(streams)

    assert post.shape == (2, 3, cfg.hc_mult)
    assert comb.shape == (2, 3, cfg.hc_mult, cfg.hc_mult)
    assert collapsed.shape == (2, 3, cfg.hidden_size)
    torch.testing.assert_close(
        comb.sum(dim=-1),
        torch.ones_like(comb.sum(dim=-1)),
        rtol=2e-4,
        atol=2e-4,
    )
    torch.testing.assert_close(
        comb.sum(dim=-2),
        torch.ones_like(comb.sum(dim=-2)),
        rtol=2e-4,
        atol=2e-4,
    )
