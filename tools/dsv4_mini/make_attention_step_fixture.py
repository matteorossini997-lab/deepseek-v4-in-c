from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .reference import (
    GroupedLinear,
    MiniAttention,
    _apply_partial_rope,
    _rope_cos_sin,
)


def c_float(value: float) -> str:
    return float(torch.tensor(value, dtype=torch.float32).item()).hex() + "f"


def emit_array(
    name: str,
    values: torch.Tensor,
    ctype: str = "float",
    per_line: int = 4,
) -> str:
    flattened = values.detach().cpu().flatten().tolist()
    lines = [f"static const {ctype} {name}[{len(flattened)}] = {{"]
    for start in range(0, len(flattened), per_line):
        chunk = flattened[start : start + per_line]
        if ctype == "float":
            rendered = ", ".join(c_float(value) for value in chunk)
        else:
            rendered = ", ".join(f"{int(value)}u" for value in chunk)
        suffix = "," if start + per_line < len(flattened) else ""
        lines.append(f"    {rendered}{suffix}")
    lines.append("};\n")
    return "\n".join(lines)


def apply_rope(
    values: torch.Tensor,
    rope_dim: int,
    position: int,
    theta: float,
) -> torch.Tensor:
    positions = torch.tensor([[position]], dtype=torch.long)
    cosine, sine = _rope_cos_sin(
        positions,
        rope_dim,
        theta,
        torch.float32,
    )
    shaped = values.view(1, values.shape[0], 1, values.shape[1])
    return _apply_partial_rope(shaped, cosine, sine, rope_dim)[0, :, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    heads = 4
    head_dim = 4
    rope_dim = 4
    raw_tokens = 3
    compressed_tokens = 4
    index_heads = 2
    index_dim = 4
    index_rope_dim = 4
    index_top_k = 2
    output_groups = 2
    output_rank = 3
    hidden_size = 6
    position = 5
    theta = 160000.0

    q = torch.linspace(-0.6, 0.8, heads * head_dim, dtype=torch.float32).view(
        heads, head_dim
    )
    raw = (
        ((torch.arange(raw_tokens * head_dim).view(raw_tokens, head_dim) * 5 + 2) % 17)
        - 8
    ).float() * 0.07
    compressed = (
        (
            (
                torch.arange(compressed_tokens * head_dim).view(
                    compressed_tokens, head_dim
                )
                * 7
                + 1
            )
            % 19
        )
        - 9
    ).float() * 0.06
    index_q = torch.linspace(
        -0.5, 0.7, index_heads * index_dim, dtype=torch.float32
    ).view(index_heads, index_dim)
    index_compressed = (
        (
            (
                torch.arange(compressed_tokens * index_dim).view(
                    compressed_tokens, index_dim
                )
                * 3
                + 4
            )
            % 13
        )
        - 6
    ).float() * 0.08
    index_head_weights = torch.tensor([0.9, -0.4], dtype=torch.float32)
    sinks = torch.tensor([-0.2, 0.05, 0.3, -0.1], dtype=torch.float32)

    input_per_group = heads * head_dim // output_groups
    o_a_weight = (
        (
            (
                torch.arange(
                    output_groups * output_rank * input_per_group
                ).view(output_groups, output_rank, input_per_group)
                * 11
                + 3
            )
            % 23
        )
        - 11
    ).float() * 0.025
    o_b_weight = (
        (
            (
                torch.arange(hidden_size * output_groups * output_rank).view(
                    hidden_size, output_groups * output_rank
                )
                * 7
                + 2
            )
            % 17
        )
        - 8
    ).float() * 0.035

    q_rotated = apply_rope(q, rope_dim, position, theta)
    index_q_rotated = apply_rope(index_q, index_rope_dim, position, theta)
    index_dots = torch.einsum(
        "hd,td->ht", index_q_rotated.float(), index_compressed.float()
    )
    index_scores = (
        F.relu(index_dots)
        * (index_dim**-0.5)
        * (index_head_weights * (index_heads**-0.5)).unsqueeze(-1)
    ).sum(dim=0)
    sorted_scores, selected = torch.topk(index_scores, index_top_k, sorted=True)
    all_scores = torch.sort(index_scores, descending=True).values
    if not torch.all(all_scores[:-1] - all_scores[1:] > 1e-6):
        raise RuntimeError("canonical fixture contains a sparse-index tie")

    grouped = GroupedLinear(output_groups, input_per_group, output_rank)
    with torch.no_grad():
        grouped.weight.copy_(o_a_weight)

    attention_proxy = SimpleNamespace(
        sinks=sinks,
        num_heads=heads,
        scaling=head_dim**-0.5,
    )

    def finish(kv: torch.Tensor) -> torch.Tensor:
        attention = MiniAttention._attention_with_sink(
            attention_proxy,
            q_rotated.view(1, heads, 1, head_dim),
            kv.view(1, 1, kv.shape[0], head_dim),
        )[0, :, 0]
        attention = apply_rope(attention, rope_dim, -position, theta)
        grouped_output = grouped(
            attention.reshape(1, output_groups, input_per_group)
        ).flatten()
        return F.linear(grouped_output, o_b_weight)

    expected_csa = finish(torch.cat([raw, compressed.index_select(0, selected)], dim=0))
    expected_hca = finish(torch.cat([raw, compressed], dim=0))
    expected_sliding = finish(raw)

    content = "\n".join(
        [
            "/* Generated by tools/dsv4_mini/make_attention_step_fixture.py.",
            " * Canonical sources: MiniAttention, GroupedLinear and RoPE helpers.",
            " * Do not edit by hand.",
            " */",
            "#ifndef DSV4_ATTENTION_STEP_VECTORS_H",
            "#define DSV4_ATTENTION_STEP_VECTORS_H",
            "",
            "#include <stddef.h>",
            "",
            f"#define DSV4_STEP_HEADS {heads}u",
            f"#define DSV4_STEP_HEAD_DIM {head_dim}u",
            f"#define DSV4_STEP_ROPE_DIM {rope_dim}u",
            f"#define DSV4_STEP_RAW_TOKENS {raw_tokens}u",
            f"#define DSV4_STEP_COMPRESSED_TOKENS {compressed_tokens}u",
            f"#define DSV4_STEP_INDEX_HEADS {index_heads}u",
            f"#define DSV4_STEP_INDEX_DIM {index_dim}u",
            f"#define DSV4_STEP_INDEX_ROPE_DIM {index_rope_dim}u",
            f"#define DSV4_STEP_TOPK {index_top_k}u",
            f"#define DSV4_STEP_GROUPS {output_groups}u",
            f"#define DSV4_STEP_RANK {output_rank}u",
            f"#define DSV4_STEP_HIDDEN {hidden_size}u",
            f"#define DSV4_STEP_POSITION {position}",
            "",
            emit_array("dsv4_step_q", q),
            emit_array("dsv4_step_raw", raw),
            emit_array("dsv4_step_compressed", compressed),
            emit_array("dsv4_step_index_q", index_q),
            emit_array("dsv4_step_index_compressed", index_compressed),
            emit_array("dsv4_step_index_head_weights", index_head_weights),
            emit_array("dsv4_step_sinks", sinks),
            emit_array("dsv4_step_oa", o_a_weight),
            emit_array("dsv4_step_ob", o_b_weight),
            emit_array("dsv4_step_selected", selected, "size_t"),
            emit_array("dsv4_step_selected_scores", sorted_scores),
            emit_array("dsv4_step_expected_csa", expected_csa),
            emit_array("dsv4_step_expected_hca", expected_hca),
            emit_array("dsv4_step_expected_sliding", expected_sliding),
            "#endif",
            "",
        ]
    )
    Path(args.output).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
