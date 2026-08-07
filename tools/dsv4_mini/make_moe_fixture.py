from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .config import MiniConfig
from .reference import SparseMoE

SOURCE_COMMIT = "1106f248ccea9bc3b06cb031e8ca58542e23c8c7"


def moe_config() -> MiniConfig:
    config = MiniConfig(
        schema_version=1,
        vocab_size=16,
        max_seq_len=16,
        hidden_size=8,
        moe_intermediate_size=6,
        num_hidden_layers=2,
        num_hash_layers=1,
        num_mtp_layers=1,
        num_attention_heads=2,
        head_dim=4,
        rope_head_dim=2,
        q_lora_rank=4,
        o_groups=2,
        o_lora_rank=3,
        sliding_window=4,
        compress_rates=(0, 4, 0),
        num_routed_experts=8,
        num_shared_experts=1,
        num_experts_per_token=2,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        index_num_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        hc_sinkhorn_iters=4,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        seed=0xD54F4321,
    )
    config.validate()
    return config


def pattern(shape: torch.Size | tuple[int, ...], tag: int, denominator: int = 128) -> torch.Tensor:
    count = math.prod(shape)
    index = torch.arange(count, dtype=torch.int64)
    integer = ((index * 13 + tag * 17) % 29) - 14
    return (integer.to(torch.float32) / float(denominator)).reshape(shape)


def fill_moe(moe: SparseMoE, layer_idx: int) -> None:
    base = 50 * layer_idx + 3
    with torch.no_grad():
        moe.router.weight.copy_(pattern(moe.router.weight.shape, base + 1))
        if hasattr(moe.router, "correction_bias"):
            moe.router.correction_bias.copy_(pattern(moe.router.correction_bias.shape, base + 2, 64))
        for expert_idx, expert in enumerate(moe.experts):
            expert.gate_proj.weight.copy_(pattern(expert.gate_proj.weight.shape, base + 10 + 3 * expert_idx))
            expert.up_proj.weight.copy_(pattern(expert.up_proj.weight.shape, base + 11 + 3 * expert_idx))
            expert.down_proj.weight.copy_(pattern(expert.down_proj.weight.shape, base + 12 + 3 * expert_idx))
        moe.shared_expert.gate_proj.weight.copy_(pattern(moe.shared_expert.gate_proj.weight.shape, base + 40))
        moe.shared_expert.up_proj.weight.copy_(pattern(moe.shared_expert.up_proj.weight.shape, base + 41))
        moe.shared_expert.down_proj.weight.copy_(pattern(moe.shared_expert.down_proj.weight.shape, base + 42))


def hidden_case(config: MiniConfig, layer_idx: int, case_idx: int) -> torch.Tensor:
    index = torch.arange(config.hidden_size, dtype=torch.int64)
    integer = ((index * 7 + case_idx * 5 + layer_idx * 11) % 19) - 9
    return (integer.to(torch.float32) / 16.0).view(1, 1, config.hidden_size)


def c_float(value: float | torch.Tensor) -> str:
    number = float(value)
    return f"{number.hex()}f"


def emit_float_array(name: str, values: list[float], per_line: int = 4) -> str:
    lines = [f"static const float {name}[{len(values)}] = {{"]
    for start in range(0, len(values), per_line):
        chunk = values[start : start + per_line]
        suffix = "," if start + per_line < len(values) else ""
        lines.append("    " + ", ".join(c_float(value) for value in chunk) + suffix)
    lines.append("};")
    return "\n".join(lines)


def emit_uint_array(name: str, values: list[int], per_line: int = 8) -> str:
    lines = [f"static const unsigned {name}[{len(values)}] = {{"]
    for start in range(0, len(values), per_line):
        chunk = values[start : start + per_line]
        suffix = "," if start + per_line < len(values) else ""
        lines.append("    " + ", ".join(f"{value}u" for value in chunk) + suffix)
    lines.append("};")
    return "\n".join(lines)


def build_header() -> str:
    torch.set_num_threads(1)
    config = moe_config()
    cases = [
        (0, 0, 3),
        (0, 1, 9),
        (1, 2, 5),
        (1, 3, 12),
    ]
    outputs: list[float] = []
    route_indices: list[int] = []
    route_weights: list[float] = []
    layers: list[int] = []
    case_ids: list[int] = []
    token_ids: list[int] = []

    modules: dict[int, SparseMoE] = {}
    for layer_idx, case_idx, token_id in cases:
        if layer_idx not in modules:
            modules[layer_idx] = SparseMoE(config, layer_idx).eval()
            fill_moe(modules[layer_idx], layer_idx)
        hidden = hidden_case(config, layer_idx, case_idx)
        ids = torch.tensor([[token_id]], dtype=torch.long)
        output, indices, weights = modules[layer_idx](hidden, ids)
        outputs.extend(float(value) for value in output.detach().reshape(-1))
        pairs = sorted(
            zip(indices.reshape(-1).tolist(), weights.reshape(-1).tolist(), strict=True),
            key=lambda pair: int(pair[0]),
        )
        route_indices.extend(int(pair[0]) for pair in pairs)
        route_weights.extend(float(pair[1]) for pair in pairs)
        layers.append(layer_idx)
        case_ids.append(case_idx)
        token_ids.append(token_id)

    preamble = f"""/* Generated by tools/dsv4_mini/make_moe_fixture.py.
 * Canonical source: tools/dsv4_mini/reference.py SparseMoE
 * Source commit: {SOURCE_COMMIT}
 * Do not edit by hand.
 */
#ifndef DSV4_MOE_VECTORS_H
#define DSV4_MOE_VECTORS_H

#define DSV4_MOE_HIDDEN 8u
#define DSV4_MOE_INTERMEDIATE 6u
#define DSV4_MOE_EXPERTS 8u
#define DSV4_MOE_TOPK 2u
#define DSV4_MOE_VOCAB 16u
#define DSV4_MOE_HASH_LAYERS 1u
#define DSV4_MOE_CASES 4u
#define DSV4_MOE_ROUTE_SCALE 0x1.8000000000000p+0f
#define DSV4_MOE_SWIGLU_LIMIT 0x1.4000000000000p+3f
"""
    arrays = [
        emit_uint_array("dsv4_moe_case_layer", layers),
        emit_uint_array("dsv4_moe_case_id", case_ids),
        emit_uint_array("dsv4_moe_case_token", token_ids),
        emit_float_array("dsv4_moe_expected_output", outputs),
        emit_uint_array("dsv4_moe_expected_indices", route_indices),
        emit_float_array("dsv4_moe_expected_weights", route_weights),
    ]
    return preamble + "\n\n" + "\n\n".join(arrays) + "\n\n#endif\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_header(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
