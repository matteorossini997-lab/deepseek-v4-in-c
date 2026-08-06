#!/usr/bin/env python3
"""Independently verify dsv4-inventory JSON against index and shard headers."""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

BITS = {
    "BOOL": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}


def expected_nbytes(dtype: str, shape: list[int]) -> int:
    if dtype not in BITS:
        raise ValueError(f"unsupported dtype {dtype}")
    numel = math.prod(shape) if shape else 1
    bits = numel * BITS[dtype]
    if bits % 8:
        raise ValueError(f"sub-byte tensor is not byte aligned: {dtype} {shape}")
    return bits // 8


def parse_shard(path: Path) -> dict[str, dict[str, Any]]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated length prefix")
        header_len = struct.unpack("<Q", prefix)[0]
        raw = handle.read(header_len)
        if len(raw) != header_len:
            raise ValueError(f"{path}: truncated header")
    header = json.loads(raw)
    data_base = 8 + header_len
    tensors: dict[str, dict[str, Any]] = {}
    spans: list[tuple[int, int, str]] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if name in tensors:
            raise ValueError(f"{path}: duplicate tensor {name}")
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        start, end = descriptor["data_offsets"]
        nbytes = end - start
        if start < 0 or end < start:
            raise ValueError(f"{path}: invalid span for {name}")
        if nbytes != expected_nbytes(dtype, shape):
            raise ValueError(f"{path}: shape/dtype mismatch for {name}")
        tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "offset": data_base + start,
            "nbytes": nbytes,
            "shard": path.name,
        }
        spans.append((start, end, name))
    cursor = 0
    for start, end, name in sorted(spans):
        if start != cursor:
            relation = "overlap" if start < cursor else "hole"
            raise ValueError(f"{path}: {relation} before {name}")
        cursor = end
    if data_base + cursor != size:
        raise ValueError(f"{path}: unindexed trailing bytes")
    return tensors


def verify(model_dir: Path, report_path: Path) -> None:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text("utf-8"))
    report = json.loads(report_path.read_text("utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    truth: dict[str, dict[str, Any]] = {}
    for shard_name in sorted(set(weight_map.values())):
        for name, descriptor in parse_shard(model_dir / shard_name).items():
            if name in truth:
                raise ValueError(f"duplicate tensor across shards: {name}")
            truth[name] = descriptor
    if set(truth) != set(weight_map):
        missing = sorted(set(weight_map) - set(truth))
        extra = sorted(set(truth) - set(weight_map))
        raise ValueError(f"index/header mismatch missing={missing[:3]} extra={extra[:3]}")
    for name, shard in weight_map.items():
        if truth[name]["shard"] != shard:
            raise ValueError(f"{name}: index maps {shard}, header is {truth[name]['shard']}")

    reported = {entry["name"]: entry for entry in report["tensor_inventory"]}
    if set(reported) != set(truth):
        raise ValueError("C report tensor set differs from independent parse")
    for name, expected in truth.items():
        got = reported[name]
        for field in ("shard", "dtype", "shape", "offset", "nbytes"):
            if got[field] != expected[field]:
                raise ValueError(
                    f"{name}: {field} C={got[field]!r} Python={expected[field]!r}"
                )
    total = sum(item["nbytes"] for item in truth.values())
    if total != index["metadata"]["total_size"]:
        raise ValueError("index metadata.total_size differs from header descriptors")
    if report["total_data_bytes"] != total or report["declared_total_size"] != total:
        raise ValueError("C report byte totals differ from independent parse")
    if report["tensors"] != len(truth):
        raise ValueError("C report tensor count differs from independent parse")
    if report["shards"] != len(set(weight_map.values())):
        raise ValueError("C report shard count differs from independent parse")
    print(f"verified {len(truth)} tensors across {report['shards']} shards ({total} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    verify(args.model_dir, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
