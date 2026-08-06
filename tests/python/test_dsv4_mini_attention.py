from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools.dsv4_mini.config import load_config
from tools.dsv4_mini.reference import MiniAttention


CONFIG_PATH = Path(__file__).resolve().parents[2] / "tools" / "dsv4_mini" / "tiny_config.json"


def _hidden(cfg, length: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234 + length)
    return torch.randn(1, length, cfg.hidden_size, generator=generator) * 0.1


def test_csa_emits_one_entry_after_each_four_source_tokens() -> None:
    cfg = load_config(CONFIG_PATH)
    attention = MiniAttention(cfg, layer_idx=2)
    state = attention.new_state(batch_size=1, device="cpu")
    hidden = _hidden(cfg, 9)
    counts = []
    index_counts = []

    for position in range(hidden.shape[1]):
        _, trace = attention.step(hidden[:, position : position + 1], position, state)
        counts.append(trace.compressed_count)
        index_counts.append(trace.indexer_compressed_count)

    assert counts == [0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert index_counts == counts


def test_hca_emits_first_entry_at_position_127() -> None:
    cfg = load_config(CONFIG_PATH)
    attention = MiniAttention(cfg, layer_idx=3)
    state = attention.new_state(batch_size=1, device="cpu")
    hidden = _hidden(cfg, 130)
    counts = []

    for position in range(hidden.shape[1]):
        _, trace = attention.step(hidden[:, position : position + 1], position, state)
        counts.append(trace.compressed_count)

    assert counts[126] == 0
    assert counts[127] == 1
    assert counts[128] == 1
    assert counts[129] == 1


def test_sliding_cache_never_exceeds_window() -> None:
    cfg = load_config(CONFIG_PATH)
    attention = MiniAttention(cfg, layer_idx=0)
    state = attention.new_state(batch_size=1, device="cpu")
    hidden = _hidden(cfg, cfg.sliding_window + 5)

    lengths = []
    for position in range(hidden.shape[1]):
        _, trace = attention.step(hidden[:, position : position + 1], position, state)
        lengths.append(trace.raw_cache_length)

    assert lengths[:3] == [1, 2, 3]
    assert lengths[cfg.sliding_window - 1] == cfg.sliding_window
    assert lengths[-1] == cfg.sliding_window


@pytest.mark.parametrize(
    ("layer_idx", "length", "positions"),
    [
        (0, 18, [15, 16, 17]),
        (2, 18, [3, 4, 15, 16, 17]),
        (3, 130, [126, 127, 128, 129]),
    ],
)
def test_attention_full_and_incremental_match_boundaries(
    layer_idx: int, length: int, positions: list[int]
) -> None:
    cfg = load_config(CONFIG_PATH)
    torch.manual_seed(41 + layer_idx)
    attention = MiniAttention(cfg, layer_idx=layer_idx).eval()
    hidden = _hidden(cfg, length)

    full, full_trace = attention.forward_full(hidden)
    state = attention.new_state(batch_size=1, device="cpu")
    incremental_parts = []
    incremental_trace = []
    for position in range(length):
        output, trace = attention.step(hidden[:, position : position + 1], position, state)
        incremental_parts.append(output)
        incremental_trace.append(trace)
    incremental = torch.cat(incremental_parts, dim=1)

    torch.testing.assert_close(full, incremental, rtol=1e-5, atol=1e-6)
    for position in positions:
        assert full_trace[position].compressed_count == incremental_trace[position].compressed_count
        assert full_trace[position].indexer_compressed_count == incremental_trace[position].indexer_compressed_count
        assert full_trace[position].index_indices == incremental_trace[position].index_indices
