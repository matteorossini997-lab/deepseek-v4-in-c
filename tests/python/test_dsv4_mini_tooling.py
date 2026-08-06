from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from tools.dsv4_mini import run_tests


ROOT = Path(__file__).resolve().parents[2]


def test_test_runner_disables_external_plugins_and_thread_oversubscription(monkeypatch) -> None:
    captured = {}

    def fake_run(command, *, cwd, env, check):
        captured.update(command=command, cwd=cwd, env=env, check=check)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_tests.subprocess, "run", fake_run)
    assert run_tests.main() == 0
    assert captured["command"] == [sys.executable, "-m", "pytest", "-q", "tests/python"]
    assert captured["cwd"] == ROOT
    assert captured["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert captured["env"]["OMP_NUM_THREADS"] == "1"
    assert captured["env"]["MKL_NUM_THREADS"] == "1"
    assert captured["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert captured["check"] is False


def test_workflow_uses_isolated_test_runner() -> None:
    workflow = (ROOT / ".github" / "workflows" / "dsv4-mini.yml").read_text(encoding="utf-8")
    assert "python tools/dsv4_mini/run_tests.py" in workflow
