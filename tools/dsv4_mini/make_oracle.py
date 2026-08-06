from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .oracle import build_oracle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the deterministic DeepSeek V4 Flash mini-oracle")
    parser.add_argument("--config", type=Path, required=True, help="Path to tiny_config.json")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sequence-length", type=int, default=130)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")
    config = load_config(args.config)
    oracle = build_oracle(config, device=args.device, sequence_length=args.sequence_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    device_name = oracle["environment"].get("device_name", "CPU")
    print(
        f"wrote {args.output} | device={device_name} | params={oracle['model']['parameter_count']:,} "
        f"| max_abs_error={oracle['parity']['max_abs_error']:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
