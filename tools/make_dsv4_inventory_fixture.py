#!/usr/bin/env python3
"""Generate a tiny synthetic DeepSeek-style Safetensors inventory fixture.

The fixture contains no model values. It exists only to exercise index/header,
absolute-offset, dtype-width and quantization-geometry validation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Any

REFERENCE_REVISION = "7d0cce91a2ba9b7738b3b33163a53eccb1cbe47c"

BITS = {
    "BF16": 16,
    "I32": 32,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
}

SHARDS: dict[str, list[tuple[str, str, list[int]]]] = {
    "model-00001-of-00002.safetensors": [
        ("model.embed_tokens.weight", "BF16", [8, 8]),
        ("model.layers.0.mlp.gate.tid2eid", "I32", [4, 6]),
        ("model.layers.3.self_attn.q_a_proj.weight", "F8_E4M3", [128, 128]),
        (
            "model.layers.3.self_attn.q_a_proj.weight_scale_inv",
            "F8_E8M0",
            [1, 1],
        ),
    ],
    "model-00002-of-00002.safetensors": [
        ("model.layers.3.mlp.experts.7.gate_proj.weight", "I8", [8, 16]),
        (
            "model.layers.3.mlp.experts.7.gate_proj.weight_scale_inv",
            "F8_E8M0",
            [8, 1],
        ),
        ("model.layers.43.mlp.experts.2.down_proj.weight", "I8", [8, 16]),
        (
            "model.layers.43.mlp.experts.2.down_proj.weight_scale_inv",
            "F8_E8M0",
            [8, 1],
        ),
        ("lm_head.weight", "BF16", [8, 8]),
    ],
}


def tensor_nbytes(dtype: str, shape: list[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    bits = numel * BITS[dtype]
    if bits % 8:
        raise ValueError(f"{dtype} {shape} is not byte aligned")
    return bits // 8


def write_shard(path: Path, tensors: list[tuple[str, str, list[int]]]) -> int:
    header: dict[str, Any] = {
        "__metadata__": {
            "format": "synthetic-inventory-only",
            "reference_revision": REFERENCE_REVISION,
        }
    }
    cursor = 0
    for name, dtype, shape in tensors:
        size = tensor_nbytes(dtype, shape)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    raw = json.dumps(header, separators=(",", ":"), sort_keys=False).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(bytes(cursor))
    return cursor


def generate(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    total = 0
    for shard_name, tensors in SHARDS.items():
        total += write_shard(output / shard_name, tensors)
        for name, _, _ in tensors:
            weight_map[name] = shard_name
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "generator": "tools/make_dsv4_inventory_fixture.py",
        "reference_repository": "deepseek-ai/DeepSeek-V4-Flash",
        "reference_weight_revision": REFERENCE_REVISION,
        "scope": "synthetic index/header and quantization geometry only",
        "shards": len(SHARDS),
        "tensors": len(weight_map),
        "total_size": total,
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
