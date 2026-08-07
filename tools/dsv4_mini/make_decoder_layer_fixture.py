from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .config import MiniConfig
from .make_attention_runtime_fixture import fill_attention
from .make_moe_fixture import fill_moe
from .reference import DecoderLayer, HyperConnection

SOURCE_COMMIT = "7195f1e3ff54e310ee35fb5eacdee446c31a8c78"
SENTINEL = 0xFFFFFFFF


def decoder_config() -> MiniConfig:
    config = MiniConfig(
        schema_version=1,
        vocab_size=16,
        max_seq_len=132,
        hidden_size=8,
        moe_intermediate_size=6,
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
        seed=0xD54F9876,
    )
    config.validate()
    return config


def pattern(shape: torch.Size | tuple[int, ...], tag: int, denominator: int = 128) -> torch.Tensor:
    count = math.prod(shape)
    index = torch.arange(count, dtype=torch.int64)
    integer = ((index * 11 + tag * 7) % 27) - 13
    return (integer.to(torch.float32) / float(denominator)).reshape(shape)


def norm_pattern(shape: torch.Size | tuple[int, ...], tag: int) -> torch.Tensor:
    count = math.prod(shape)
    index = torch.arange(count, dtype=torch.int64)
    integer = ((index * 5 + tag * 2) % 11) - 5
    return (1.0 + integer.to(torch.float32) / 64.0).reshape(shape)


def scale_pattern(tag: int) -> torch.Tensor:
    index = torch.arange(3, dtype=torch.int64)
    integer = ((index * 3 + tag) % 7) - 3
    return 1.0 + integer.to(torch.float32) / 16.0


def fill_hc(module: HyperConnection, tag: int) -> None:
    with torch.no_grad():
        module.fn.copy_(pattern(module.fn.shape, tag + 1))
        module.base.copy_(pattern(module.base.shape, tag + 2, 256))
        module.scale.copy_(scale_pattern(tag + 3))


def fill_layer(layer: DecoderLayer, layer_idx: int) -> None:
    fill_attention(layer.attention, layer_idx)
    fill_moe(layer.moe, layer_idx)
    base = 80 * layer_idx + 5
    with torch.no_grad():
        layer.input_norm.weight.copy_(norm_pattern(layer.input_norm.weight.shape, base + 1))
        layer.post_attention_norm.weight.copy_(
            norm_pattern(layer.post_attention_norm.weight.shape, base + 2)
        )
    fill_hc(layer.attention_hc, base + 10)
    fill_hc(layer.ffn_hc, base + 20)


def stream_input(config: MiniConfig, layer_idx: int, position: int) -> torch.Tensor:
    stream = torch.arange(config.hc_mult, dtype=torch.int64).view(config.hc_mult, 1)
    dim = torch.arange(config.hidden_size, dtype=torch.int64).view(1, config.hidden_size)
    integer = ((stream * 13 + dim * 7 + position * 5 + layer_idx * 17) % 31) - 15
    return (integer.to(torch.float32) / 32.0).view(
        1, 1, config.hc_mult, config.hidden_size
    )


def token_for(layer_idx: int, position: int, vocab_size: int) -> int:
    return (3 + layer_idx * 5 + position * 7) % vocab_size


def c_float(value: float | torch.Tensor) -> str:
    return f"{float(value).hex()}f"


def emit_float_array(name: str, values: list[float], per_line: int = 4) -> str:
    lines = [f"static const float {name}[{len(values)}] = {{"]
    for start in range(0, len(values), per_line):
        chunk = values[start : start + per_line]
        suffix = "," if start + per_line < len(values) else ""
        lines.append("    " + ", ".join(c_float(value) for value in chunk) + suffix)
    lines.append("};")
    return "\n".join(lines)


def emit_uint_array(name: str, values: list[int], per_line: int = 10) -> str:
    lines = [f"static const unsigned {name}[{len(values)}] = {{"]
    for start in range(0, len(values), per_line):
        chunk = values[start : start + per_line]
        suffix = "," if start + per_line < len(values) else ""
        lines.append("    " + ", ".join(f"{value}u" for value in chunk) + suffix)
    lines.append("};")
    return "\n".join(lines)


def run_layer(
    config: MiniConfig,
    layer_idx: int,
    steps: int,
) -> tuple[list[float], list[int], list[int], list[int], list[int], list[int], list[int], list[float]]:
    layer = DecoderLayer(config, layer_idx).eval()
    fill_layer(layer, layer_idx)
    state = layer.new_state(1, "cpu", torch.float32)

    outputs: list[float] = []
    raw: list[int] = []
    compressed: list[int] = []
    indexer: list[int] = []
    attention_selected_count: list[int] = []
    attention_selected: list[int] = []
    route_indices: list[int] = []
    route_weights: list[float] = []

    for position in range(steps):
        token = token_for(layer_idx, position, config.vocab_size)
        ids = torch.tensor([[token]], dtype=torch.long)
        result, trace = layer.step(stream_input(config, layer_idx, position), ids, position, state)
        outputs.extend(float(value) for value in result.detach().reshape(-1))
        raw.append(trace.attention.raw_cache_length)
        compressed.append(trace.attention.compressed_count)
        indexer.append(trace.attention.indexer_compressed_count)

        selected = list(trace.attention.index_indices[: config.index_topk])
        attention_selected_count.append(len(selected))
        selected.extend([SENTINEL] * (config.index_topk - len(selected)))
        attention_selected.extend(selected)

        pairs = sorted(
            zip(trace.route_indices, trace.route_weights, strict=True),
            key=lambda pair: int(pair[0]),
        )
        route_indices.extend(int(pair[0]) for pair in pairs)
        route_weights.extend(float(pair[1]) for pair in pairs)

    return (
        outputs,
        raw,
        compressed,
        indexer,
        attention_selected_count,
        attention_selected,
        route_indices,
        route_weights,
    )


def build_header() -> str:
    torch.set_num_threads(1)
    config = decoder_config()
    cases = [
        ("sliding", 0, 6),
        ("csa", 1, 9),
        ("hca", 2, 129),
    ]
    sections: list[str] = []
    for name, layer_idx, steps in cases:
        values = run_layer(config, layer_idx, steps)
        prefix = f"dsv4_decoder_{name}"
        sections.extend(
            [
                emit_float_array(f"{prefix}_output", values[0]),
                emit_uint_array(f"{prefix}_raw_count", values[1]),
                emit_uint_array(f"{prefix}_compressed_count", values[2]),
                emit_uint_array(f"{prefix}_indexer_count", values[3]),
                emit_uint_array(f"{prefix}_attention_selected_count", values[4]),
                emit_uint_array(f"{prefix}_attention_selected", values[5]),
                emit_uint_array(f"{prefix}_route_indices", values[6]),
                emit_float_array(f"{prefix}_route_weights", values[7]),
            ]
        )

    preamble = f"""/* Generated by tools/dsv4_mini/make_decoder_layer_fixture.py.
 * Canonical source: tools/dsv4_mini/reference.py DecoderLayer.step
 * Source commit: {SOURCE_COMMIT}
 * Do not edit by hand.
 */
#ifndef DSV4_DECODER_LAYER_VECTORS_H
#define DSV4_DECODER_LAYER_VECTORS_H

#define DSV4_DECODER_HIDDEN 8u
#define DSV4_DECODER_HC 2u
#define DSV4_DECODER_HC_MIX 8u
#define DSV4_DECODER_Q_RANK 4u
#define DSV4_DECODER_HEADS 2u
#define DSV4_DECODER_HEAD_DIM 4u
#define DSV4_DECODER_ROPE_DIM 2u
#define DSV4_DECODER_GROUPS 2u
#define DSV4_DECODER_O_RANK 3u
#define DSV4_DECODER_WINDOW 4u
#define DSV4_DECODER_INDEX_HEADS 2u
#define DSV4_DECODER_INDEX_DIM 4u
#define DSV4_DECODER_INDEX_TOPK 2u
#define DSV4_DECODER_INTERMEDIATE 6u
#define DSV4_DECODER_EXPERTS 8u
#define DSV4_DECODER_MOE_TOPK 2u
#define DSV4_DECODER_VOCAB 16u
#define DSV4_DECODER_HASH_LAYERS 1u
#define DSV4_DECODER_SINKHORN_ITERS 4
#define DSV4_DECODER_RMS_EPS 0x1.0c6f7a0b5ed8dp-20f
#define DSV4_DECODER_HC_EPS 0x1.0c6f7a0b5ed8dp-20f
#define DSV4_DECODER_ROPE_THETA 0x1.3880000000000p+13f
#define DSV4_DECODER_COMPRESS_THETA 0x1.3880000000000p+17f
#define DSV4_DECODER_ROUTE_SCALE 0x1.8000000000000p+0f
#define DSV4_DECODER_SWIGLU_LIMIT 0x1.4000000000000p+3f
#define DSV4_DECODER_SLIDING_STEPS 6u
#define DSV4_DECODER_CSA_STEPS 9u
#define DSV4_DECODER_HCA_STEPS 129u
#define DSV4_DECODER_SENTINEL 0xffffffffu
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
