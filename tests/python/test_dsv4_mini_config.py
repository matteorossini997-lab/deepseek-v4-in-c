from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.dsv4_mini.config import MiniConfig, load_config


CONFIG_PATH = Path(__file__).resolve().parents[2] / "tools" / "dsv4_mini" / "tiny_config.json"


def test_canonical_config_derives_all_layer_categories() -> None:
    cfg = load_config(CONFIG_PATH)

    assert cfg.layer_types == (
        "sliding_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
    )
    assert cfg.mlp_layer_types == (
        "hash_moe",
        "hash_moe",
        "hash_moe",
        "learned_moe",
        "learned_moe",
        "learned_moe",
        "learned_moe",
    )
    assert cfg.mtp_compress_rate == 0


def test_config_rejects_topk_larger_than_expert_count() -> None:
    raw = json.loads(CONFIG_PATH.read_text())
    raw["num_experts_per_token"] = raw["num_routed_experts"] + 1

    with pytest.raises(ValueError, match="num_experts_per_token"):
        MiniConfig.from_dict(raw)


def test_config_rejects_rope_dimension_not_even() -> None:
    raw = json.loads(CONFIG_PATH.read_text())
    raw["rope_head_dim"] = 7

    with pytest.raises(ValueError, match="rope_head_dim"):
        MiniConfig.from_dict(raw)


def test_config_rejects_missing_mtp_rate() -> None:
    raw = json.loads(CONFIG_PATH.read_text())
    raw["compress_rates"] = raw["compress_rates"][:-1]

    with pytest.raises(ValueError, match="compress_rates"):
        MiniConfig.from_dict(raw)
