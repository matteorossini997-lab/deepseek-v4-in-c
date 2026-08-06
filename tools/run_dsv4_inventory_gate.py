#!/usr/bin/env python3
"""Run the synthetic DeepSeek inventory gate for Make, CTest and CI."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], *, stdout=None) -> None:
    completed = subprocess.run(command, check=False, stdout=stdout)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()

    fixture = args.workdir / "fixture"
    report = args.workdir / "inventory.json"
    args.workdir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(args.generator), "--output", str(fixture)])
    with report.open("wb") as handle:
        run([
  str(args.cli), "--json", "--tensors", "--config",
  str(args.config), str(fixture)
        ], stdout=handle)
    run([sys.executable, str(args.verifier), str(fixture), str(report)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
