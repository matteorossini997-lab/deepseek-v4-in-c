from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools.dsv4_mini.config import load_config
from tools.dsv4_mini.make_oracle import main as make_oracle_main
from tools.dsv4_mini.oracle import build_oracle


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "tools" / "dsv4_mini" / "tiny_config.json"


@pytest.fixture(scope="module")
def long_oracle() -> dict:
    cfg = load_config(CONFIG_PATH)
    return build_oracle(cfg, device="cpu", sequence_length=130)


def test_oracle_covers_required_boundaries_and_paths(long_oracle: dict) -> None:
    assert long_oracle["schema_version"] == 1
    assert long_oracle["model"]["architecture"] == "deepseek_v4_flash_mini"
    assert long_oracle["model"]["parameter_count"] > 0
    assert long_oracle["parity"]["argmax_equal"] is True
    assert long_oracle["parity"]["routes_equal"] is True
    assert long_oracle["parity"]["max_abs_error"] <= 2e-6
    assert long_oracle["selected_positions"] == [0, 1, 3, 4, 15, 16, 17, 126, 127, 128, 129]

    position_127 = long_oracle["trace"]["127"]
    position_126 = long_oracle["trace"]["126"]
    assert position_126["layers"][3]["attention"]["compressed_count"] == 0
    assert position_127["layers"][3]["attention"]["compressed_count"] == 1
    assert len(position_127["layers"][0]["route_indices"]) == 6
    assert len(position_127["mtp"]["route_indices"]) == 6


def test_short_oracle_is_deterministic() -> None:
    cfg = load_config(CONFIG_PATH)
    first = build_oracle(cfg, device="cpu", sequence_length=18)
    second = build_oracle(cfg, device="cpu", sequence_length=18)

    assert first["checksums"] == second["checksums"]
    assert first["generated_ids"] == second["generated_ids"]
    assert first["trace"] == second["trace"]


def test_cli_writes_valid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "oracle.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_oracle",
            "--config",
            str(CONFIG_PATH),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--sequence-length",
            "18",
        ],
    )
    assert make_oracle_main() == 0

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["sequence_length"] == 18
    assert document["parity"]["argmax_equal"] is True
    assert document["checksums"]["full_logits"]
