from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .make_decoder_layer_fixture import decoder_config, fill_layer
from .make_model_shell_fixture import (
    emit_float_array,
    emit_uint_array,
    fill_base_model,
    floats,
    norm_pattern,
    pattern,
    token_for,
)
from .reference import MiniReferenceModel

SOURCE_COMMIT = "e67a8f6b1203c417724f8e47f74dd5f9af56dab5"
STEPS = 20


def fill_mtp(model: MiniReferenceModel) -> None:
    mtp = model.mtp
    fill_layer(mtp.decoder, model.config.num_hidden_layers)
    with torch.no_grad():
        mtp.embedding_norm.weight.copy_(norm_pattern(mtp.embedding_norm.weight.shape, 401))
        mtp.hidden_norm.weight.copy_(norm_pattern(mtp.hidden_norm.weight.shape, 402))
        mtp.embedding_proj.weight.copy_(pattern(mtp.embedding_proj.weight.shape, 403))
        mtp.hidden_proj.weight.copy_(pattern(mtp.hidden_proj.weight.shape, 404))
        mtp.hc_head.fn.copy_(pattern(mtp.hc_head.fn.shape, 405))
        mtp.hc_head.base.copy_(pattern(mtp.hc_head.base.shape, 406, 256))
        mtp.hc_head.scale.copy_(pattern(mtp.hc_head.scale.shape, 407, 16))
        mtp.output_norm.weight.copy_(norm_pattern(mtp.output_norm.weight.shape, 408))


def emit_mtp_decoder_weights(model: MiniReferenceModel) -> list[str]:
    layer = model.mtp.decoder
    attention = layer.attention
    moe = layer.moe
    prefix = "dsv4_mtp_decoder"
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
        emit_float_array(f"{prefix}_bias", floats(moe.router.correction_bias)),
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
    if attention.compressor is not None or attention.indexer is not None:
        raise AssertionError("canonical P1-J MTP layer must use compression rate zero")
    return sections


def build_header() -> str:
    torch.set_num_threads(1)
    config = decoder_config()
    model = MiniReferenceModel(config).eval()
    fill_base_model(model)
    fill_mtp(model)

    base_states = [layer.new_state(1, "cpu", torch.float32) for layer in model.layers]
    mtp_state = model.mtp.new_state(1, "cpu", torch.float32)

    token_values: list[int] = []
    base_stream_values: list[float] = []
    mtp_stream_values: list[float] = []
    logits_values: list[float] = []
    argmax_values: list[int] = []
    route_index_values: list[int] = []
    route_weight_values: list[float] = []
    raw_cache_values: list[int] = []

    for position in range(STEPS):
        token = token_for(position, config.vocab_size)
        ids = torch.tensor([[token]], dtype=torch.long)
        streams = model._expand_embeddings(ids)
        for layer, state in zip(model.layers, base_states, strict=True):
            streams, _ = layer.step(streams, ids, position, state)
        base_stream_values.extend(floats(streams))

        mtp_input = model.mtp._inputs(streams, ids)
        mtp_streams, trace = model.mtp.decoder.step(mtp_input, ids, position, mtp_state)
        logits = model.mtp.head(model.mtp.output_norm(model.mtp.hc_head(mtp_streams)))

        token_values.append(token)
        mtp_stream_values.extend(floats(mtp_streams))
        logits_values.extend(floats(logits))
        argmax_values.append(int(logits.argmax(dim=-1).item()))
        route_index_values.extend(int(value) for value in trace.route_indices)
        route_weight_values.extend(float(value) for value in trace.route_weights)
        raw_cache_values.append(trace.attention.raw_cache_length)

    mtp = model.mtp
    preamble = f"""/* Generated by tools/dsv4_mini/make_mtp_fixture.py.
 * Canonical source: tools/dsv4_mini/reference.py MTPBlock.
 * Base source commit: {SOURCE_COMMIT}
 * Shared embedding and LM-head arrays come from dsv4_model_shell_vectors.h.
 * Do not edit by hand.
 */
#ifndef DSV4_MTP_VECTORS_H
#define DSV4_MTP_VECTORS_H

#define DSV4_MTP_HIDDEN 8u
#define DSV4_MTP_HC 2u
#define DSV4_MTP_VOCAB 16u
#define DSV4_MTP_STEPS {STEPS}u
#define DSV4_MTP_MAX_SEQ 132u
#define DSV4_MTP_LAYER_IDX 3u
#define DSV4_MTP_Q_RANK 4u
#define DSV4_MTP_HEADS 2u
#define DSV4_MTP_HEAD_DIM 4u
#define DSV4_MTP_ROPE_DIM 2u
#define DSV4_MTP_GROUPS 2u
#define DSV4_MTP_O_RANK 3u
#define DSV4_MTP_WINDOW 4u
#define DSV4_MTP_INTERMEDIATE 6u
#define DSV4_MTP_EXPERTS 8u
#define DSV4_MTP_TOPK 2u
#define DSV4_MTP_HASH_LAYERS 1u
#define DSV4_MTP_SINKHORN_ITERS 4
#define DSV4_MTP_RMS_EPS 0x1.0c6f7a0b5ed8dp-20f
#define DSV4_MTP_HC_EPS 0x1.0c6f7a0b5ed8dp-20f
#define DSV4_MTP_ROPE_THETA 0x1.3880000000000p+13f
#define DSV4_MTP_COMPRESS_THETA 0x1.3880000000000p+17f
#define DSV4_MTP_ROUTE_SCALE 0x1.8000000000000p+0f
#define DSV4_MTP_SWIGLU_LIMIT 0x1.4000000000000p+3f
"""

    sections = [
        emit_float_array("dsv4_mtp_embedding_norm", floats(mtp.embedding_norm.weight)),
        emit_float_array("dsv4_mtp_hidden_norm", floats(mtp.hidden_norm.weight)),
        emit_float_array("dsv4_mtp_embedding_proj", floats(mtp.embedding_proj.weight)),
        emit_float_array("dsv4_mtp_hidden_proj", floats(mtp.hidden_proj.weight)),
        emit_float_array("dsv4_mtp_hc_head_fn", floats(mtp.hc_head.fn)),
        emit_float_array("dsv4_mtp_hc_head_base", floats(mtp.hc_head.base)),
        emit_float_array("dsv4_mtp_hc_head_scale", floats(mtp.hc_head.scale)),
        emit_float_array("dsv4_mtp_output_norm", floats(mtp.output_norm.weight)),
    ]
    sections.extend(emit_mtp_decoder_weights(model))
    sections.extend(
        [
            emit_uint_array("dsv4_mtp_token", token_values),
            emit_float_array("dsv4_mtp_base_streams", base_stream_values),
            emit_float_array("dsv4_mtp_expected_streams", mtp_stream_values),
            emit_float_array("dsv4_mtp_expected_logits", logits_values),
            emit_uint_array("dsv4_mtp_expected_argmax", argmax_values),
            emit_uint_array("dsv4_mtp_expected_route_indices", route_index_values),
            emit_float_array("dsv4_mtp_expected_route_weights", route_weight_values),
            emit_uint_array("dsv4_mtp_expected_raw_cache", raw_cache_values),
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
