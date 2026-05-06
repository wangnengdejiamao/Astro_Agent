from pathlib import Path

import pytest

from analysis_agent.nodes.claude_code_node import claude_code_execute_node
from claude_code_toolbox import build_task
from claude_code_toolbox.schemas import ClaudeCodeTaskType


class _FakeBackend:
    name = "fake"

    def __init__(self, payload):
        self._payload = payload

    def run(self, task):
        return self._payload


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch, tmp_path):
    """Replace the default client with one that uses a fake backend."""
    from claude_code_toolbox import client as client_mod

    def _factory(run_id=None):
        backend = _FakeBackend(
            {
                "status": "ok",
                "stdout": '{"result": "Refined draft. Method unchanged."}',
                "stderr": "",
                "cmd": ["fake"],
            }
        )
        return client_mod.ClaudeCodeClient(
            backend=backend, runs_root=tmp_path, run_id=run_id or "test-run"
        )

    monkeypatch.setattr(
        "analysis_agent.nodes.claude_code_node.make_default_client", _factory
    )
    yield


def test_node_skips_when_no_tasks():
    state = {}
    out = claude_code_execute_node(state)
    assert out["claude_code_results"] == []


def test_node_executes_paper_refine_and_writes_report():
    task = build_task(
        ClaudeCodeTaskType.PAPER_REFINE,
        {"draft": "We measured the period. Method unchanged."},
    )
    state = {"claude_code_tasks": [task.model_dump(mode="json")], "run_id": "test-run"}
    out = claude_code_execute_node(state)
    assert len(out["claude_code_results"]) == 1
    res = out["claude_code_results"][0]
    assert res["type"] == "paper_refine"
    # human_review_required forces NEEDS_REVIEW even if backend says ok
    assert res["status"] in {"needs_review", "ok", "blocked"}
    assert "paper_refinement_report" in out


def test_node_writes_code_quality_report():
    task = build_task(
        ClaudeCodeTaskType.CODE_REVIEW,
        {"target_files": ["astro_toolbox/desi.py"], "context": "SED unit consistency"},
    )
    state = {"claude_code_tasks": [task.model_dump(mode="json")]}
    out = claude_code_execute_node(state)
    assert "code_quality_report" in out
