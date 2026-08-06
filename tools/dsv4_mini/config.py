from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping


_LAYER_TYPE_BY_RATE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}


@dataclass(frozen=True, slots=True)
class MiniConfig:
    schema_version: int
    vocab_size: int
    max_seq_len: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_hash_layers: int
    num_mtp_layers: int
    num_attention_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    sliding_window: int
    compress_rates: tuple[int, ...]
    num_routed_experts: int
    num_shared_experts: int
    num_experts_per_token: int
    scoring_func: str
    routed_scaling_factor: float
    swiglu_limit: float
    index_num_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    rms_norm_eps: float
    rope_theta: float
    compress_rope_theta: float
    seed: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MiniConfig":
        expected = {field.name for field in fields(cls)}
        missing = sorted(expected - raw.keys())
        extra = sorted(raw.keys() - expected)
        if missing:
            raise ValueError(f"missing config fields: {', '.join(missing)}")
        if extra:
            raise ValueError(f"unknown config fields: {', '.join(extra)}")
        values = dict(raw)
        values["compress_rates"] = tuple(int(value) for value in values["compress_rates"])
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        positive_ints = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "hidden_size": self.hidden_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "q_lora_rank": self.q_lora_rank,
            "o_groups": self.o_groups,
            "o_lora_rank": self.o_lora_rank,
            "sliding_window": self.sliding_window,
            "num_routed_experts": self.num_routed_experts,
            "num_shared_experts": self.num_shared_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "index_num_heads": self.index_num_heads,
            "index_head_dim": self.index_head_dim,
            "index_topk": self.index_topk,
            "hc_mult": self.hc_mult,
            "hc_sinkhorn_iters": self.hc_sinkhorn_iters,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.num_hash_layers <= self.num_hidden_layers:
            raise ValueError("num_hash_layers must be between zero and num_hidden_layers")
        if self.num_mtp_layers != 1:
            raise ValueError("this fixture requires exactly one num_mtp_layers")
        if self.num_experts_per_token > self.num_routed_experts:
            raise ValueError("num_experts_per_token cannot exceed num_routed_experts")
        if self.rope_head_dim <= 0 or self.rope_head_dim % 2:
            raise ValueError("rope_head_dim must be a positive even number")
        if self.rope_head_dim > self.head_dim:
            raise ValueError("rope_head_dim cannot exceed head_dim")
        if self.num_attention_heads * self.head_dim % self.o_groups:
            raise ValueError("num_attention_heads * head_dim must be divisible by o_groups")
        if self.hidden_size * self.hc_mult <= 0:
            raise ValueError("hidden_size * hc_mult must be positive")
        expected_rates = self.num_hidden_layers + self.num_mtp_layers
        if len(self.compress_rates) != expected_rates:
            raise ValueError(
                f"compress_rates must contain {expected_rates} entries "
                f"({self.num_hidden_layers} base + {self.num_mtp_layers} MTP)"
            )
        unsupported = sorted(set(self.compress_rates) - _LAYER_TYPE_BY_RATE.keys())
        if unsupported:
            raise ValueError(f"unsupported compress_rates: {unsupported}")
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError("scoring_func must be sqrtsoftplus")
        if self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be positive")
        if self.swiglu_limit <= 0:
            raise ValueError("swiglu_limit must be positive")
        if self.hc_eps <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("normalization epsilons must be positive")
        if self.index_topk > self.max_seq_len // 4:
            raise ValueError("index_topk is larger than the maximum CSA entry count")

    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple(_LAYER_TYPE_BY_RATE[rate] for rate in self.compress_rates[: self.num_hidden_layers])

    @property
    def mlp_layer_types(self) -> tuple[str, ...]:
        return tuple(
            "hash_moe" if index < self.num_hash_layers else "learned_moe"
            for index in range(self.num_hidden_layers)
        )

    @property
    def mtp_compress_rate(self) -> int:
        return self.compress_rates[self.num_hidden_layers]

    def to_dict(self) -> dict[str, Any]:
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["compress_rates"] = list(self.compress_rates)
        return result


def load_config(path: str | Path) -> MiniConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON object")
    return MiniConfig.from_dict(raw)
