from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .config import MiniConfig
from .reference import MiniAttention, MiniCompressor

SOURCE_COMMIT = "70c79636aa92af42552da035290348e9e504ffa0"


def tiny_runtime_config() -> MiniConfig:
    config = MiniConfig(
        schema_version=1,
        vocab_size=32,
        max_seq_len=132,
        hidden_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=3,
        num_hash_layers=1,
        num_mtp_layers=1,
        num_attention_heads=2,
        head_dim=4,
        rope_head_dim=2,
        q_lora_rank=4,
        o_groups=2,
        o_lora_rank=3,
        sliding_window=4,
        compress_rates=(0, 4, 128, 0),
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
        seed=0xD54F1234,
    )
    config.validate()
    return config


def _pattern(shape: torch.Size | tuple[int, ...], tag: int, denominator: int = 128) -> torch.Tensor:
    count = math.prod(shape)
    index = torch.arange(count, dtype=torch.int64)
    integer = ((index * 17 + tag * 11) % 31) - 15
    return (integer.to(torch.float32) / float(denominator)).reshape(shape)


def _norm_pattern(shape: torch.Size | tuple[int, ...], tag: int) -> torch.Tensor:
    count = math.prod(shape)
    index = torch.arange(count, dtype=torch.int64)
    integer = ((index * 5 + tag * 3) % 9) - 4
    return (1.0 + integer.to(torch.float32) / 64.0).reshape(shape)


def _fill_linear(weight: torch.Tensor, tag: int) -> None:
    weight.copy_(_pattern(weight.shape, tag))


def _fill_compressor(compressor: MiniCompressor, tag: int) -> None:
    _fill_linear(compressor.kv_proj.weight, tag + 1)
    _fill_linear(compressor.gate_proj.weight, tag + 2)
    compressor.position_bias.copy_(_pattern(compressor.position_bias.shape, tag + 3, 256))
    compressor.norm.weight.copy_(_norm_pattern(compressor.norm.weight.shape, tag + 4))


def fill_attention(attention: MiniAttention, layer_idx: int) -> None:
    base = 40 * layer_idx + 1
    with torch.no_grad():
        _fill_linear(attention.q_a_proj.weight, base + 1)
        attention.q_a_norm.weight.copy_(_norm_pattern(attention.q_a_norm.weight.shape, base + 2))
        _fill_linear(attention.q_b_proj.weight, base + 3)
        _fill_linear(attention.kv_proj.weight, base + 4)
        attention.kv_norm.weight.copy_(_norm_pattern(attention.kv_norm.weight.shape, base + 5))
        attention.o_a_proj.weight.copy_(_pattern(attention.o_a_proj.weight.shape, base + 6))
        _fill_linear(attention.o_b_proj.weight, base + 7)
        attention.sinks.copy_(_pattern(attention.sinks.shape, base + 8, 32))
        if attention.compressor is not None:
            _fill_compressor(attention.compressor, base + 10)
        if attention.indexer is not None:
            _fill_linear(attention.indexer.q_proj.weight, base + 20)
            _fill_linear(attention.indexer.weight_proj.weight, base + 21)
            _fill_compressor(attention.indexer.compressor, base + 22)


def hidden_token(config: MiniConfig, layer_idx: int, position: int) -> torch.Tensor:
    index = torch.arange(config.hidden_size, dtype=torch.int64)
    integer = ((index * 7 + position * 5 + layer_idx * 13) % 23) - 11
    return (integer.to(torch.float32) / 32.0).view(1, 1, config.hidden_size)


def c_float(value: float | torch.Tensor) -> str:
    number = float(value)
    if math.isnan(number):
        return "NAN"
    if math.isinf(number):
        return "INFINITY" if number > 0 else "-INFINITY"
    return f"{number.hex()}f"


def emit_float_array(name: str, values: torch.Tensor, per_line: int = 4) -> str:
    flat = values.detach().cpu().reshape(-1).tolist()
    lines = [f"static const float {name}[{len(flat)}] = {{"]
    for start in range(0, len(flat), per_line):
        chunk = flat[start : start + per_line]
        suffix = "," if start + per_line < len(flat) else ""
        lines.append("    " + ", ".join(c_float(value) for value in chunk) + suffix)
    lines.append("};")
    return "\n".join(lines)


def emit_uint_array(name: str, values: list[int], per_line: int = 12) -> str:
    lines = [f"static const unsigned {name}[{len(values)}] = {{"]
    for start in range(0, len(values), per_line):
        chunk = values[start : start + per_line]
        suffix = "," if start + per_line < len(values) else ""
        lines.append("    " + ", ".join(f"{value}u" for value in chunk) + suffix)
    lines.append("};")
    return "\n".join(lines)


def run_layer(config: MiniConfig, layer_idx: int, steps: int) -> tuple[torch.Tensor, list[int], list[int], list[int], list[int], list[int]]:
    attention = MiniAttention(config, layer_idx).eval()
    fill_attention(attention, layer_idx)
    state = attention.new_state(1, "cpu", torch.float32)
    outputs: list[torch.Tensor] = []
    raw_counts: list[int] = []
    compressed_counts: list[int] = []
    indexer_counts: list[int] = []
    selected_counts: list[int] = []
    selected_slots: list[int] = []
    for position in range(steps):
        output, trace = attention.step(hidden_token(config, layer_idx, position), position, state)
        outputs.append(output.detach().cpu())
        raw_counts.append(trace.raw_cache_length)
        compressed_counts.append(trace.compressed_count)
        indexer_counts.append(trace.indexer_compressed_count)
        selected_counts.append(len(trace.index_indices))
        slots = list(trace.index_indices[: config.index_topk])
        slots.extend([0xFFFFFFFF] * (config.index_topk - len(slots)))
        selected_slots.extend(slots)
    return (
        torch.cat(outputs, dim=1),
        raw_counts,
        compressed_counts,
        indexer_counts,
        selected_counts,
        selected_slots,
    )


def build_header() -> str:
    torch.set_num_threads(1)
    config = tiny_runtime_config()
    cases = [
        ("sliding", 0, 7),
        ("csa", 1, 9),
        ("hca", 2, 129),
    ]
    sections: list[str] = []
    for name, layer_idx, steps in cases:
        outputs, raw, compressed, indexer, selected_count, selected = run_layer(config, layer_idx, steps)
        prefix = f"dsv4_runtime_{name}"
        sections.extend(
            [
                emit_float_array(f"{prefix}_output", outputs),
                emit_uint_array(f"{prefix}_raw_count", raw),
                emit_uint_array(f"{prefix}_compressed_count", compressed),
                emit_uint_array(f"{prefix}_indexer_count", indexer),
                emit_uint_array(f"{prefix}_selected_count", selected_count),
                emit_uint_array(f"{prefix}_selected", selected),
            ]
        )

    preamble = f"""/* Generated by tools/dsv4_mini/make_attention_runtime_fixture.py.
 * Canonical source: tools/dsv4_mini/reference.py MiniAttention.step
 * Source commit: {SOURCE_COMMIT}
 * Do not edit by hand.
 */
#ifndef DSV4_ATTENTION_RUNTIME_VECTORS_H
#define DSV4_ATTENTION_RUNTIME_VECTORS_H

#define DSV4_RUNTIME_HIDDEN 8u
#define DSV4_RUNTIME_Q_RANK 4u
#define DSV4_RUNTIME_HEADS 2u
#define DSV4_RUNTIME_HEAD_DIM 4u
#define DSV4_RUNTIME_ROPE_DIM 2u
#define DSV4_RUNTIME_GROUPS 2u
#define DSV4_RUNTIME_O_RANK 3u
#define DSV4_RUNTIME_WINDOW 4u
#define DSV4_RUNTIME_INDEX_HEADS 2u
#define DSV4_RUNTIME_INDEX_DIM 4u
#define DSV4_RUNTIME_INDEX_TOPK 2u
#define DSV4_RUNTIME_SLIDING_STEPS 7u
#define DSV4_RUNTIME_CSA_STEPS 9u
#define DSV4_RUNTIME_HCA_STEPS 129u
#define DSV4_RUNTIME_RMS_EPS 0x1.0c6f7a0b5ed8dp-20f
#define DSV4_RUNTIME_ROPE_THETA 0x1.3880000000000p+13f
#define DSV4_RUNTIME_COMPRESS_THETA 0x1.3880000000000p+17f
#define DSV4_RUNTIME_SENTINEL 0xffffffffu
"""
    return preamble + "\n\n" + "\n\n".join(sections) + "\n\n#endif\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_header(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
