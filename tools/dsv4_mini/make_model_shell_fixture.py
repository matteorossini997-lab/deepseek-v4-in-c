from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .make_decoder_layer_fixture import decoder_config, fill_layer
from .reference import MiniReferenceModel

SOURCE_COMMIT = "df79a91c4e24bf3629dd4bd6c57557163393a0ce"


def pattern(shape: torch.Size | tuple[int, ...], tag: int, denominator: int = 128) -> torch.Tensor:
    count = math.prod(shape)
    index = torch.arange(count, dtype=torch.int64)
    integer = ((index * 19 + tag * 5) % 33) - 16
    return (integer.to(torch.float32) / float(denominator)).reshape(shape)


def norm_pattern(shape: torch.Size | tuple[int, ...], tag: int) -> torch.Tensor:
    count = math.prod(shape)
    index = torch.arange(count, dtype=torch.int64)
    integer = ((index * 7 + tag * 3) % 13) - 6
    return (1.0 + integer.to(torch.float32) / 64.0).reshape(shape)


def fill_base_model(model: MiniReferenceModel) -> None:
    with torch.no_grad():
        model.embed_tokens.weight.copy_(pattern(model.embed_tokens.weight.shape, 301))
        for layer_idx, layer in enumerate(model.layers):
            fill_layer(layer, layer_idx)
        model.hc_head.fn.copy_(pattern(model.hc_head.fn.shape, 302))
        model.hc_head.base.copy_(pattern(model.hc_head.base.shape, 303, 256))
        model.hc_head.scale.copy_(pattern(model.hc_head.scale.shape, 304, 16))
        model.output_norm.weight.copy_(norm_pattern(model.output_norm.weight.shape, 305))
        model.lm_head.weight.copy_(pattern(model.lm_head.weight.shape, 306))


def token_for(position: int, vocab_size: int) -> int:
    return (3 + position * 7) % vocab_size


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


def emit_uint_array(name: str, values: list[int], per_line: int = 12) -> str:
    lines = [f"static const unsigned {name}[{len(values)}] = {{"]
    for start in range(0, len(values), per_line):
        chunk = values[start : start + per_line]
        suffix = "," if start + per_line < len(values) else ""
        lines.append("    " + ", ".join(f"{value}u" for value in chunk) + suffix)
    lines.append("};")
    return "\n".join(lines)


def floats(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1)]


def uints(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().reshape(-1)]


def emit_layer_weights(model: MiniReferenceModel, layer_idx: int) -> list[str]:
    layer = model.layers[layer_idx]
    attention = layer.attention
    moe = layer.moe
    prefix = f"dsv4_model_l{layer_idx}"
    sections = [
        emit_float_array(f"{prefix}_attn_hc_fn", floats(layer.attention_hc.fn)),
        emit_float_array(f"{prefix}_attn_hc_base", floats(layer.attention_hc.base)),
        emit_float_array(f"{prefix}_attn_hc_scale", floats(layer.attention_hc.scale)),
        emit_float_array(f"{prefix}_ffn_hc_fn", floats(layer.ffn_hc.fn)),
        emit_float_array(f"{prefix}_ffn_hc_base", floats(layer.ffn_hc.base)),
        emit_float_array(f"{prefix}_ffn_hc_scale", floats(layer.ffn_hc.scale)),
        emit_float_array(f"{prefix}_input_norm", floats(layer.input_norm.weight)),
        emit_float_array(f"{prefix}_post_attn_norm", floats(layer.post_attention_norm.weight)),
        emit_float_array(f"{prefix}_q_a", floats(attention.q_a_proj.weight)),
        emit_float_array(f"{prefix}_q_a_norm", floats(attention.q_a_norm.weight)),
        emit_float_array(f"{prefix}_q_b", floats(attention.q_b_proj.weight)),
        emit_float_array(f"{prefix}_kv", floats(attention.kv_proj.weight)),
        emit_float_array(f"{prefix}_kv_norm", floats(attention.kv_norm.weight)),
        emit_float_array(f"{prefix}_sinks", floats(attention.sinks)),
        emit_float_array(f"{prefix}_o_a", floats(attention.o_a_proj.weight)),
        emit_float_array(f"{prefix}_o_b", floats(attention.o_b_proj.weight)),
        emit_float_array(f"{prefix}_router", floats(moe.router.weight)),
        emit_float_array(
            f"{prefix}_expert_gate",
            floats(torch.stack([expert.gate_proj.weight for expert in moe.experts])),
        ),
        emit_float_array(
            f"{prefix}_expert_up",
            floats(torch.stack([expert.up_proj.weight for expert in moe.experts])),
        ),
        emit_float_array(
            f"{prefix}_expert_down",
            floats(torch.stack([expert.down_proj.weight for expert in moe.experts])),
        ),
        emit_float_array(f"{prefix}_shared_gate", floats(moe.shared_expert.gate_proj.weight)),
        emit_float_array(f"{prefix}_shared_up", floats(moe.shared_expert.up_proj.weight)),
        emit_float_array(f"{prefix}_shared_down", floats(moe.shared_expert.down_proj.weight)),
    ]
    if hasattr(moe.router, "tid2eid"):
        sections.append(emit_uint_array(f"{prefix}_hash", uints(moe.router.tid2eid)))
    else:
        sections.append(emit_float_array(f"{prefix}_bias", floats(moe.router.correction_bias)))

    if attention.compressor is not None:
        compressor = attention.compressor
        sections.extend(
            [
                emit_float_array(f"{prefix}_compress_kv", floats(compressor.kv_proj.weight)),
                emit_float_array(f"{prefix}_compress_gate", floats(compressor.gate_proj.weight)),
                emit_float_array(f"{prefix}_compress_bias", floats(compressor.position_bias)),
                emit_float_array(f"{prefix}_compress_norm", floats(compressor.norm.weight)),
            ]
        )
    if attention.indexer is not None:
        indexer = attention.indexer
        sections.extend(
            [
                emit_float_array(f"{prefix}_index_q", floats(indexer.q_proj.weight)),
                emit_float_array(f"{prefix}_index_head", floats(indexer.weight_proj.weight)),
                emit_float_array(
                    f"{prefix}_index_compress_kv", floats(indexer.compressor.kv_proj.weight)
                ),
                emit_float_array(
                    f"{prefix}_index_compress_gate", floats(indexer.compressor.gate_proj.weight)
                ),
                emit_float_array(
                    f"{prefix}_index_compress_bias", floats(indexer.compressor.position_bias)
                ),
                emit_float_array(
                    f"{prefix}_index_compress_norm", floats(indexer.compressor.norm.weight)
                ),
            ]
        )
    return sections


def build_header() -> str:
    torch.set_num_threads(1)
    config = decoder_config()
    model = MiniReferenceModel(config).eval()
    fill_base_model(model)
    layer_states = [layer.new_state(1, "cpu", torch.float32) for layer in model.layers]

    streams_values: list[float] = []
    logits_values: list[float] = []
    argmax_values: list[int] = []
    token_values: list[int] = []

    for position in range(config.max_seq_len - 3):
        token = token_for(position, config.vocab_size)
        ids = torch.tensor([[token]], dtype=torch.long)
        streams = model._expand_embeddings(ids)
        for layer, state in zip(model.layers, layer_states, strict=True):
            streams, _trace = layer.step(streams, ids, position, state)
        hidden = model.output_norm(model.hc_head(streams))
        logits = model.lm_head(hidden)
        streams_values.extend(floats(streams))
        logits_values.extend(floats(logits))
        argmax_values.append(int(logits.argmax(dim=-1).item()))
        token_values.append(token)

    steps = len(token_values)
    preamble = f"""/* Generated by tools/dsv4_mini/make_model_shell_fixture.py.
 * Canonical source: tools/dsv4_mini/reference.py MiniReferenceModel base path
 * Source commit: {SOURCE_COMMIT}
 * MTP is deliberately not executed in this fixture.
 * Do not edit by hand.
 */
#ifndef DSV4_MODEL_SHELL_VECTORS_H
#define DSV4_MODEL_SHELL_VECTORS_H

#define DSV4_MODEL_HIDDEN 8u
#define DSV4_MODEL_HC 2u
#define DSV4_MODEL_LAYERS 3u
#define DSV4_MODEL_VOCAB 16u
#define DSV4_MODEL_STEPS {steps}u
#define DSV4_MODEL_MAX_SEQ 132u
#define DSV4_MODEL_Q_RANK 4u
#define DSV4_MODEL_HEADS 2u
#define DSV4_MODEL_HEAD_DIM 4u
#define DSV4_MODEL_ROPE_DIM 2u
#define DSV4_MODEL_GROUPS 2u
#define DSV4_MODEL_O_RANK 3u
#define DSV4_MODEL_WINDOW 4u
#define DSV4_MODEL_INDEX_HEADS 2u
#define DSV4_MODEL_INDEX_DIM 4u
#define DSV4_MODEL_INDEX_TOPK 2u
#define DSV4_MODEL_INTERMEDIATE 6u
#define DSV4_MODEL_EXPERTS 8u
#define DSV4_MODEL_MOE_TOPK 2u
#define DSV4_MODEL_HASH_LAYERS 1u
#define DSV4_MODEL_SINKHORN_ITERS 4
#define DSV4_MODEL_RMS_EPS 0x1.0c6f7a0b5ed8dp-20f
#define DSV4_MODEL_HC_EPS 0x1.0c6f7a0b5ed8dp-20f
#define DSV4_MODEL_ROPE_THETA 0x1.3880000000000p+13f
#define DSV4_MODEL_COMPRESS_THETA 0x1.3880000000000p+17f
#define DSV4_MODEL_ROUTE_SCALE 0x1.8000000000000p+0f
#define DSV4_MODEL_SWIGLU_LIMIT 0x1.4000000000000p+3f
"""
    sections = [
        emit_float_array("dsv4_model_embedding", floats(model.embed_tokens.weight)),
        emit_float_array("dsv4_model_hc_head_fn", floats(model.hc_head.fn)),
        emit_float_array("dsv4_model_hc_head_base", floats(model.hc_head.base)),
        emit_float_array("dsv4_model_hc_head_scale", floats(model.hc_head.scale)),
        emit_float_array("dsv4_model_output_norm", floats(model.output_norm.weight)),
        emit_float_array("dsv4_model_lm_head", floats(model.lm_head.weight)),
    ]
    for layer_idx in range(config.num_hidden_layers):
        sections.extend(emit_layer_weights(model, layer_idx))
    sections.extend(
        [
            emit_uint_array("dsv4_model_token", token_values),
            emit_float_array("dsv4_model_expected_streams", streams_values),
            emit_float_array("dsv4_model_expected_logits", logits_values),
            emit_uint_array("dsv4_model_expected_argmax", argmax_values),
        ]
    )
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
